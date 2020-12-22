import pickle

import torch
import torch.nn as nn
from manopth import rodrigues_layer

from hocontact.hodatasets.hoquery import TransQueries, BaseQueries, one_query_in
from hocontact.models.bases import resnet
from hocontact.models.manobranch import ManoBranch
from hocontact.models.transhead import TransHead, recover_3d_proj
from hocontact.utils import handutils
from hocontact.utils import netutils
from hocontact.utils.logger import logger


class ManoAdaptor(torch.nn.Module):
    def __init__(self, mano_layer, load_path=None):
        super().__init__()
        self.adaptor = torch.nn.Linear(778, 21, bias=False)
        if load_path is not None:
            with open(load_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
                weights = exp_data["adaptor"]
            regressor = torch.Tensor(weights)
            self.register_buffer("J_regressor", regressor)
        else:
            regressor = mano_layer._buffers["th_J_regressor"]
            tip_reg = regressor.new_zeros(5, regressor.shape[1])
            tip_reg[0, 745] = 1
            tip_reg[1, 317] = 1
            tip_reg[2, 444] = 1
            tip_reg[3, 556] = 1
            tip_reg[4, 673] = 1
            reordered_reg = torch.cat([regressor, tip_reg])[
                [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            ]
            self.register_buffer("J_regressor", reordered_reg)
        self.adaptor.weight.data = self.J_regressor

    def forward(self, inp):
        fix_idxs = [0, 4, 8, 12, 16, 20]
        for idx in fix_idxs:
            self.adaptor.weight.data[idx] = self.J_regressor[idx]
        return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


class HONet(nn.Module):
    def __init__(
        self,
        fc_dropout: int = 0,
        resnet_version: int = 18,
        mano_neurons: list = [512, 512],
        mano_comps: int = 15,
        mano_use_pca: bool = True,
        mano_use_shape: bool = True,
        mano_center_idx: int = 9,
        mano_root: str = "assets/mano",
        mano_pose_coeff: int = 1,
        mano_fhb_hand: bool = False,
        ## all lambdas :
        mano_lambda_recov_joints3d=None,
        mano_lambda_recov_verts3d=None,
        mano_lambda_shape=0,
        mano_lambda_pose_reg=0,
        obj_lambda_recov_verts3d=None,
        obj_lambda_recov_verts2d=None,
        obj_trans_factor=1,
        obj_scale_factor=1,
    ):
        """
        Args:
            mano_fhb_hand: Use pre-computed mapping from MANO joints to First Person
            Hand Action Benchmark hand skeleton
            mano_root (path): dir containing mano pickle files
            mano_neurons: number of neurons in each layer of base mano decoder
            mano_use_pca: predict pca parameters directly instead of rotation
                angles
            mano_comps (int): number of principal components to use if
                mano_use_pca
            mano_lambda_pca: weight to supervise hand pose in PCA space
            mano_lambda_pose_reg: weight to supervise hand pose in axis-angle
                space
            mano_lambda_verts: weight to supervise vertex distances
            mano_lambda_joints3d: weight to supervise distances
            adapt_atlas_decoder: add layer between encoder and decoder, usefull
                when finetuning from separately pretrained encoder and decoder
        """
        super().__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(pretrained=True)
        else:
            logger.error("Resnet {} not supported".format(resnet_version))
            raise NotImplementedError()

        mano_base_neurons = [img_feature_size] + mano_neurons
        self.mano_fhb_hand = mano_fhb_hand
        self.base_net = base_net
        # Predict translation and scaling for hand
        self.mano_transhead = TransHead(base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=3)
        # Predict translation, scaling and rotation for object
        self.obj_transhead = TransHead(base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=6)

        self.obj_scale_factor = obj_scale_factor
        self.obj_trans_factor = obj_trans_factor

        self.mano_branch = ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=mano_pose_coeff,
            mano_root=mano_root,
            center_idx=mano_center_idx,
            use_pca=mano_use_pca,
            use_shape=mano_use_shape,
        )
        self.mano_center_idx = mano_center_idx

        self.adaptor = None
        if self.mano_fhb_hand:
            load_fhb_path = f"assets/mano/fhb_skel_centeridx{mano_center_idx}.pkl"
            with open(load_fhb_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
            self.register_buffer("fhb_shape", torch.Tensor(exp_data["shape"]))
            self.adaptor = ManoAdaptor(self.mano_branch.mano_layer, load_fhb_path)
            netutils.rec_freeze(self.adaptor)

        self.mano_lambdas = False
        if mano_lambda_recov_joints3d or mano_lambda_recov_verts3d:
            self.mano_lambdas = True

        self.obj_lambdas = False
        if obj_lambda_recov_verts3d or obj_lambda_recov_verts2d:
            self.obj_lambdas = True

        self.mano_lambda_recov_joints3d = mano_lambda_recov_joints3d
        self.mano_lambda_recov_verts3d = mano_lambda_recov_verts3d
        self.obj_lambda_recov_verts3d = obj_lambda_recov_verts3d
        self.obj_lambda_recov_verts2d = obj_lambda_recov_verts2d

    def recover_mano(self, sample, features, rank=None):
        # Get hand projection, centered
        device = torch.device("cuda") if rank is None else torch.device(f"cuda:{rank}")

        mano_results = self.mano_branch(features)
        if self.adaptor:
            adapt_joints, _ = self.adaptor(mano_results["verts3d"])
            adapt_joints = adapt_joints.transpose(1, 2)
            mano_results["joints3d"] = adapt_joints - adapt_joints[:, self.mano_center_idx].unsqueeze(1)
            mano_results["verts3d"] = mano_results["verts3d"] - adapt_joints[:, self.mano_center_idx].unsqueeze(1)

        # Recover hand position in camera coordinates
        if self.mano_lambda_recov_joints3d or self.mano_lambda_recov_verts3d:
            scaletrans = self.mano_transhead(features)
            trans = scaletrans[:, 1:]
            scale = scaletrans[:, :1]

            final_trans = trans.unsqueeze(1) * self.obj_trans_factor
            final_scale = scale.view(scale.shape[0], 1, 1) * self.obj_scale_factor
            height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
            camintr = sample[TransQueries.CAM_INTR].to(device)
            recov_joints3d, hand_center3d = recover_3d_proj(
                mano_results["joints3d"], camintr, final_scale, final_trans, input_res=(width, height)
            )
            recov_hand_verts3d = mano_results["verts3d"] + hand_center3d
            proj_joints2d = handutils.batch_proj2d(recov_joints3d, camintr)
            proj_verts2d = handutils.batch_proj2d(recov_hand_verts3d, camintr)

            # * @Xinyu: mano_results["recov_joints3d"] = mano_results["joints3d"] + mano_results["hand_center3d"]
            mano_results["joints2d"] = proj_joints2d
            mano_results["hand_center3d"] = hand_center3d  # ===== To PICR =====
            mano_results["recov_joints3d"] = recov_joints3d  # ===== To PICR =====
            mano_results["recov_hand_verts3d"] = recov_hand_verts3d  # ===== To PICR =====
            mano_results["verts2d"] = proj_verts2d
            mano_results["hand_pretrans"] = trans
            mano_results["hand_prescale"] = scale
            mano_results["hand_trans"] = final_trans
            mano_results["hand_scale"] = final_scale
            # * @Xinyu:  mano_results["full_pose"]  ===== To PICR =====
            # * @Xinyu:  mano_results["shape"]  ===== To PICR =====

        return mano_results

    def recover_object(self, sample, features, rank=None):
        """
        Compute object vertex and corner positions in camera coordinates by predicting object translation
        and scaling, and recovering 3D positions given known object model
        """
        device = torch.device("cuda") if rank is None else torch.device(f"cuda:{rank}")

        scaletrans_obj = self.obj_transhead(features)
        batch_size = scaletrans_obj.shape[0]
        scale = scaletrans_obj[:, :1]
        trans = scaletrans_obj[:, 1:3]
        rotaxisang = scaletrans_obj[:, 3:]

        rotmat = rodrigues_layer.batch_rodrigues(rotaxisang).view(rotaxisang.shape[0], 3, 3)
        can_obj_verts = sample[BaseQueries.OBJ_CAN_VERTS].to(device)
        rot_obj_verts = rotmat.bmm(can_obj_verts.float().transpose(1, 2)).transpose(1, 2)

        final_trans = trans.unsqueeze(1) * self.obj_trans_factor
        final_scale = scale.view(batch_size, 1, 1) * self.obj_scale_factor
        height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
        camintr = sample[TransQueries.CAM_INTR].to(device)
        recov_obj_verts3d, obj_center3d = recover_3d_proj(
            rot_obj_verts, camintr, final_scale, final_trans, input_res=(width, height)
        )

        # Recover 2D positions given camera intrinsic parameters and object vertex
        # coordinates in camera coordinate reference
        pred_obj_verts2d = handutils.batch_proj2d(recov_obj_verts3d, camintr)
        if BaseQueries.OBJ_CORNERS_3D in sample:
            can_obj_corners = sample[BaseQueries.OBJ_CAN_CORNERS].to(device)
            rot_obj_corners = rotmat.bmm(can_obj_corners.float().transpose(1, 2)).transpose(1, 2)
            recov_obj_corners3d = rot_obj_corners + obj_center3d
            pred_obj_corners2d = handutils.batch_proj2d(recov_obj_corners3d, camintr)
        else:
            pred_obj_corners2d = None
            recov_obj_corners3d = None
            rot_obj_corners = None

        #  @Xinyu: obj_results["recov_obj_verts3d"] = \
        #      obj_results["rotaxisang"] @  OBJ_CAN_VERTS + obj_results["obj_center3d"]
        obj_results = {
            "obj_verts2d": pred_obj_verts2d,
            "obj_verts3d": rot_obj_verts,
            "obj_center3d": obj_center3d,  # ===== To PICR =====
            "recov_obj_verts3d": recov_obj_verts3d,  # ===== To PICR =====
            "recov_obj_corners3d": recov_obj_corners3d,
            "obj_scale": final_scale,
            "obj_prescale": scale,
            "obj_prerot": rotaxisang,  # ===== To PICR =====
            "obj_trans": final_trans,
            "obj_pretrans": trans,
            "obj_corners2d": pred_obj_corners2d,
            "obj_corners3d": rot_obj_corners,
        }

        return obj_results

    def forward(self, sample, rank=None):
        if rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{rank}")
        results = {}
        image = sample[TransQueries.IMAGE].to(device)
        features, _ = self.base_net(image)

        has_mano_supv = one_query_in(
            sample.keys(),
            [TransQueries.JOINTS_3D, TransQueries.JOINTS_2D, TransQueries.HAND_VERTS_2D, TransQueries.HAND_VERTS_3D,],
        )

        has_obj_supv = one_query_in(sample.keys(), [TransQueries.OBJ_VERTS_2D, TransQueries.OBJ_VERTS_3D])

        if has_mano_supv and self.mano_lambdas:
            mano_results = self.recover_mano(sample, features, rank)
            results.update(mano_results)

        if has_obj_supv and self.obj_lambdas:
            obj_results = self.recover_object(sample, features, rank)
            results.update(obj_results)

        return results
