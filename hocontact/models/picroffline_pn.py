import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manopth.rodrigues_layer import batch_rodrigues

from hocontact.hodatasets.hoquery import BaseQueries, TransQueries
from hocontact.models.bases import hourglass
from hocontact.models.contacthead import PointNetContactHead


class PicrOfflineHourglassPointNet(nn.Module):
    def __init__(
        self,
        picr_use_hand_pose=False,
        hg_stacks=2,
        hg_blocks=1,
        hg_classes=64,
        obj_scale_factor=1.0,
        mean_offset=0.020,
        std_offset=0.005,
        maximal_angle=math.pi / 12,
    ):
        super(PicrOfflineHourglassPointNet, self).__init__()
        self.obj_scale_factor = obj_scale_factor
        self.picr_use_hand_pose = picr_use_hand_pose

        # ================ CREATE BASE NET >>>>>>>>>>>>>>>>>>>>
        self.base_net = hourglass.StackedHourglass(hg_stacks, hg_blocks, hg_classes)
        self.intermediate_feature_size = hg_classes
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ================ CREATE HEADERS >>>>>>>>>>>>>>>>>>>>>
        if self.picr_use_hand_pose:
            self.contact_head = PointNetContactHead(
                feat_dim=self.intermediate_feature_size + 48 + 1, n_region=17, n_anchor=4
            )
        else:
            self.contact_head = PointNetContactHead(feat_dim=self.intermediate_feature_size + 1, n_region=17, n_anchor=4)
        self.mean_offset = mean_offset
        self.std_offset = std_offset
        self.maximal_angle = maximal_angle
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
        axisang = PicrOfflineHourglassPointNet.generate_random_direction()
        angle = torch.rand(1) * max_angle
        axisang = axisang * angle
        rot_mat = batch_rodrigues(axisang.unsqueeze(0)).view(1, 3, 3)
        return rot_mat

    def forward(self, sample, rank=None):
        ls_results = []
        if rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{rank}")

        image = sample[TransQueries.IMAGE]
        image_resolution = torch.from_numpy(np.array([image.shape[3], image.shape[2]])).float()  # TENSOR[2]
        image = image.to(device)
        image_resolution = image_resolution.to(device)

        ls_hg_feature, _ = self.base_net(image)  # prefix [ ls_ ] = list
        has_contact_supv = True

        # * if block rot, then TransQueries.OBJ_VERTS_3D is equal to BaseQueries.OBJ_VERTS_3D
        objverts3d = sample[TransQueries.OBJ_VERTS_3D].float()
        handposewrtcam = sample[BaseQueries.HAND_POSE_WRT_CAM].float()  # TENSOR[NBATCH, 48]
        cam_intr = sample[TransQueries.CAM_INTR].float()
        objverts3d = objverts3d.to(device)
        cam_intr = cam_intr.to(device)

        if has_contact_supv:
            for i_stack in range(self.base_net.nstacks):  # RANGE 2
                i_hg_feature = ls_hg_feature[i_stack]  # TENSOR[NBATCH, 64, 1/4 ?, 1/4 ?]
                i_contact_results = self.picr_forward(
                    cam_intr, objverts3d, i_hg_feature, image_resolution, handposewrtcam
                )
                ls_results.append(i_contact_results)

        return ls_results

    # ? ========= name it picr [ 'Pee-Ker'] Pixel-wise Implicity function for Contact Region Recovery >>>>>>>>>>>>>>>>>
    def picr_forward(self, cam_intr, object_vert_3d, low_level_feature_map, image_resolution, hand_pose_wrt_cam=None):
        """
        low_level_feature_map = TENSOR[NBATCH, 64, 1/4 IMGH, 1/4 IMGW]
        object_vert_3d = TENSOR[NBATCH, NPOINT, 3]
        hand_pose_wrt_cam = TENSOR[NBATCH, 48]
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
        # object_vert_3d_xy = object_vert_3d[:, :, :2]  # TENSOR(B, N, 2)
        # object_vert_3d_xy = object_vert_3d_xy.unsqueeze(-1).permute(0, 2, 1, 3)  # TENSOR(B, 2, N, 1)
        object_vert_3d_z = object_vert_3d[:, :, 2:]  # TENSOR(B, N, 1)
        normed_object_vert_3d_z = ((object_vert_3d_z - 0.4) / focal) / self.obj_scale_factor
        normed_object_vert_3d_z = normed_object_vert_3d_z.unsqueeze(1)  # TENSOR(B, 1, N, 1)
        to_concat_list = [collected_features, normed_object_vert_3d_z]

        if self.picr_use_hand_pose:
            # hand features
            num_points = object_vert_3d.shape[1]
            hand_pose_wrt_cam = hand_pose_wrt_cam.unsqueeze(-1).unsqueeze(-1)  # TENSOR(B, 48, 1, 1)
            hand_pose_wrt_cam = hand_pose_wrt_cam.expand(-1, -1, num_points, 1)
            to_concat_list.append(hand_pose_wrt_cam)

        collected_features = torch.cat(to_concat_list, dim=1)  # TENSOR(B, 113, N, 1)

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

