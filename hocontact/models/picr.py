import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manopth.rodrigues_layer import batch_rodrigues

from hocontact.hodatasets.hoquery import TransQueries
from hocontact.models.bases import hourglass
from hocontact.models.contacthead import PointNetContactHead
from hocontact.models.honet import HONet
from hocontact.utils.netutils import freeze_batchnorm_stats


class PicrHourglassPointNet(nn.Module):
    def __init__(
        self,
        hg_stacks=2,
        hg_blocks=1,
        hg_classes=64,
        obj_scale_factor=0.0001,
        honet_resnet_version=18,
        honet_center_idx=9,
        honet_mano_lambda_recov_joints3d=0.5,
        honet_mano_lambda_recov_verts3d=0,
        honet_mano_lambda_shape=5e-07,
        honet_mano_lambda_pose_reg=5e-06,
        honet_obj_lambda_recov_verts3d=0.5,
        honet_obj_trans_factor=100,
        honet_mano_fhb_hand=False,
        mean_offset=0.010,
        std_offset=0.005,
        maximal_angle=math.pi / 24,
    ):
        super(PicrHourglassPointNet, self).__init__()
        self.obj_scale_factor = obj_scale_factor

        # ================ CREATE BASE NET >>>>>>>>>>>>>>>>>>>>
        self.ho_net = HONet(
            resnet_version=honet_resnet_version,
            mano_center_idx=honet_center_idx,
            mano_lambda_recov_joints3d=honet_mano_lambda_recov_joints3d,
            mano_lambda_recov_verts3d=honet_mano_lambda_recov_verts3d,
            mano_lambda_shape=honet_mano_lambda_shape,
            mano_lambda_pose_reg=honet_mano_lambda_pose_reg,
            obj_lambda_recov_verts3d=honet_obj_lambda_recov_verts3d,
            obj_trans_factor=honet_obj_trans_factor,
            obj_scale_factor=obj_scale_factor,
            mano_fhb_hand=honet_mano_fhb_hand,
        )
        for param in self.ho_net.parameters():
            param.requires_grad = False
        self.ho_net.eval()
        freeze_batchnorm_stats(self.ho_net)
        self.base_net = hourglass.StackedHourglass(hg_stacks, hg_blocks, hg_classes)
        self.intermediate_feature_size = hg_classes
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ================ CREATE HEADERS >>>>>>>>>>>>>>>>>>>>>
        self.contact_head = PointNetContactHead(feat_dim=self.intermediate_feature_size + 1, n_region=17, n_anchor=4)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.mean_offset = mean_offset
        self.std_offset = std_offset
        self.maximal_angle = maximal_angle

    @staticmethod
    def generate_random_direction():
        # first get azimuthal angle
        azi = torch.rand(1) * 2 * math.pi
        # next get inclination angle
        cos_inc = 1 - 2 * torch.rand(1)
        sin_inc = torch.sqrt(1 - cos_inc ** 2)
        # gen vec
        vec = torch.tensor([sin_inc * torch.cos(azi), sin_inc * torch.sin(azi), cos_inc])
        return vec.float()

    @staticmethod
    def generate_random_rotation(max_angle):
        axisang = PicrHourglassPointNet.generate_random_direction()
        angle = torch.rand(1) * max_angle
        axisang = axisang * angle
        rot_mat = batch_rodrigues(axisang.unsqueeze(0)).view(1, 3, 3)
        return rot_mat

    def forward(self, sample, rank=None):
        # first compute ho_net
        self.ho_net.eval()
        with torch.no_grad():
            honet_results = self.ho_net(sample, rank=rank)

        # get device
        if rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{rank}")

        ls_results = []

        image = sample[TransQueries.IMAGE]
        image_resolution = torch.from_numpy(np.array([image.shape[3], image.shape[2]])).float()  # TENSOR[2]
        image = image.to(device)
        image_resolution = image_resolution.to(device)

        ls_hg_feature, _ = self.base_net(image)  # prefix [ ls_ ] = list
        has_contact_supv = True

        # * if block rot, then TransQueries.OBJ_VERTS_3D is equal to BaseQueries.OBJ_VERTS_3D
        objverts3d = honet_results["recov_obj_verts3d"]
        cam_intr = sample[TransQueries.CAM_INTR].float()
        cam_intr = cam_intr.to(device)

        if has_contact_supv:
            for i_stack in range(self.base_net.nstacks):  # RANGE 2
                i_hg_feature = ls_hg_feature[i_stack]  # TENSOR[NBATCH, 64, 1/4 ?, 1/4 ?]
                i_contact_results = self.picr_forward(cam_intr, objverts3d, i_hg_feature, image_resolution)
                ls_results.append(i_contact_results)

        # ====== get required fields from honet_results
        extra_results = {
            "hand_tsl": honet_results["hand_center3d"],
            "hand_joints_3d": honet_results["recov_joints3d"],  # in fhb format if appliable
            "hand_verts_3d": honet_results["recov_hand_verts3d"],
            "hand_full_pose": honet_results["full_pose"],  # in axisang
            "hand_shape": honet_results["shape"],
            "obj_tsl": honet_results["obj_center3d"],
            "obj_rot": honet_results["obj_prerot"],  # in axisang
            "obj_verts_3d": honet_results["recov_obj_verts3d"],
        }
        ls_results[-1].update(extra_results)

        # ====== for evalutils
        evalutil_results = {
            "recov_obj_verts3d": honet_results["recov_obj_verts3d"],
            "obj_verts2d": honet_results["obj_verts2d"],
        }
        ls_results[-1].update(evalutil_results)

        return ls_results

    # ? ========= name it picr [ 'Pee-Ker'] Pixel-wise Implicity function for Contact Region Recovery >>>>>>>>>>>>>>>>>
    def picr_forward(self, cam_intr, object_vert_3d, low_level_feature_map, image_resolution):
        """
        low_level_feature_map = TENSOR[NBATCH, 64, 1/4 IMGH, 1/4 IMGW]
        object_vert_3d = TENSOR[NBATCH, NPOINT, 3]
        image_resolution = TENSOR[2]
        """
        results = {}

        # ? ================= STAGE 1, index the features >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.training:
            # generate a random_rotation
            batch_size = object_vert_3d.shape[0]
            rand_rot = (
                self.generate_random_rotation(self.maximal_angle).expand(batch_size, -1, -1).to(object_vert_3d.device)
            )
            mean_obj_v = torch.mean(object_vert_3d, dim=1, keepdim=True)  # TENSOR[NBATCH, 1, 3]
            object_vert_3d = (
                torch.bmm(rand_rot, (object_vert_3d - mean_obj_v).permute(0, 2, 1)).permute(0, 2, 1) + mean_obj_v
            )

            # generate a random_direction
            dir_vec = self.generate_random_direction()
            rand_dist = torch.normal(torch.Tensor([self.mean_offset]), torch.Tensor([self.std_offset]))
            offset = rand_dist * dir_vec
            offset = offset.to(object_vert_3d.device)
            object_vert_3d = object_vert_3d + offset

        reprojected_vert = torch.bmm(cam_intr, object_vert_3d.transpose(1, 2)).transpose(1, 2)
        reprojected_vert = reprojected_vert[:, :, :2] / reprojected_vert[:, :, 2:]  # TENSOR[NBATCH, NPOINT, 2]

        image_center_coord = image_resolution / 2  # TENSOR[2]
        image_resolution = image_resolution.view((1, 1, 2))  # TENSOR[1, 1, 2]
        image_center_coord = image_center_coord.view((1, 1, 2))  # TENSOR[1, 1, 2]
        reprojected_grid = (reprojected_vert - image_center_coord) / image_center_coord  # TENSOR[NBATCH, NPOINT, 2]
        # compute the in image mask, so that the points fall out of the image can be filtered when calculating loss
        in_image_mask = (
            (reprojected_grid[:, :, 0] >= -1.0)
            & (reprojected_grid[:, :, 0] <= 1.0)
            & (reprojected_grid[:, :, 1] >= -1.0)
            & (reprojected_grid[:, :, 1] <= 1.0)
        )
        in_image_mask = in_image_mask.float()
        # reshape reprojected_grid so that it fits the torch grid_sample interface
        reprojected_grid = reprojected_grid.unsqueeze(2)  # TENSOR[NBATCH, NPOINT, 1, 2]
        # by default. grid sampling have zero padding
        # those points get outside of current featmap will have feature vector all zeros
        collected_features = F.grid_sample(
            low_level_feature_map, reprojected_grid, align_corners=True
        )  # TENSOR[NBATCH, 64, NPOINT, 1]

        # ? =============== STAGE 2, concate the geometry features >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # z_normed = (z - 0.4)/focal, we uses the focal normalized z value.
        focal = cam_intr[:, :1, :1]
        object_vert_3d_z = object_vert_3d[:, :, 2:]  # TENSOR(B, N, 1)
        normed_object_vert_3d_z = ((object_vert_3d_z - 0.4) / focal) / self.obj_scale_factor
        normed_object_vert_3d_z = normed_object_vert_3d_z.unsqueeze(1)  # TENSOR(B, 1, N, 1)
        collected_features = torch.cat((collected_features, normed_object_vert_3d_z), dim=1)  # TENSOR(B, 65, N, 1)

        # ? ============== STAGE 3, pass to contact head for vertex contact, region classfication & elasticity >>>>>>>>>
        vertex_contact_pred, contact_region_pred, anchor_elasti_pred = self.contact_head(collected_features)
        results.update(
            {
                "recov_vertex_contact": vertex_contact_pred,
                "recov_contact_in_image_mask": in_image_mask,
                "recov_contact_region": contact_region_pred,
                "recov_anchor_elasti": anchor_elasti_pred,
            }
        )

        return results
