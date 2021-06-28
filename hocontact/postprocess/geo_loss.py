from math import pi

import numpy as np
import torch
from manopth.quatutils import quaternion_norm_squared, quaternion_inv, quaternion_mul

from hocontact.utils.collisionutils import pairwise_dist

# original name: WorldLoss
class FieldLoss:
    @staticmethod
    def contact_loss(apos, vpos, e, e_k):
        # apos, vpos = TENSOR[NVALID, 3]
        # e = TENSOR[NVALID, ]
        dist = torch.sum(torch.pow(vpos - apos, 2), dim=1)  # TENSOR[NVALID, ]
        res = torch.mean(e_k * e * dist, dim=0)
        return res

    @staticmethod
    def repulsion_loss(
        pred_hand_verts,
        concat_hand_vert_idx,
        concat_obj_vert_3d,
        concat_obj_normal,
        constant=0.05,
        threshold=0.015,
    ):
        # pred_hand_verts = TENSOR[NHANDVERTS, 3]
        selected_hand_verts = pred_hand_verts[concat_hand_vert_idx, :]  # TENSOR[NCC, 3]
        # compute offset vector from object to hand
        offset_vectors = selected_hand_verts - concat_obj_vert_3d  # TENSOR[NCC, 3]
        # compute inner product (not normalized)
        inner_product = torch.einsum("bi,bi->b", offset_vectors, concat_obj_normal)
        thresholded_value = constant * torch.pow(
            torch.exp(torch.clamp(-inner_product, -threshold, threshold)), 2
        )  # TENSOR[NCC, ]
        # res = torch.mean(torch.pow(thresholded_value, 2), dim=0)
        res = torch.sum(thresholded_value, dim=0)
        return res

    @staticmethod
    def full_repulsion_loss(
        pred_hand_verts,
        pred_full_obj_verts,
        pred_full_obj_normal,
        query_candidate=50,
        query=0.020,
        constant=5e-4,
        threshold=0.080,
        offset=0.000,
    ):
        # get basic dim
        n_points_obj = pred_full_obj_verts.shape[0]
        # pairwise dist
        dist_mat = pairwise_dist(pred_full_obj_verts, pred_hand_verts)
        # sort in axis 1 and get candidates
        sort_idx = torch.argsort(dist_mat, dim=1)[:, 0:query_candidate]  # TENSOR[NPO, CANDI]
        # dist_mask
        dist_mask_bool = dist_mat[torch.arange(n_points_obj)[:, None], sort_idx] < query * query
        calc_mask = torch.any(dist_mask_bool, dim=1).long()
        if torch.sum(calc_mask) > 0:
            dist_mask = dist_mask_bool.float()
            # index and offset
            indexed_hand = pred_hand_verts[sort_idx]  # TENSOR[NPO, CANDI, 3]
            offset_vec = indexed_hand - pred_full_obj_verts.unsqueeze(1)  # TENSOR[NPO, CANDI, 3]; TENSOR[NPO, 1, 3]
            # inner product
            inner_prod = torch.einsum("bni,bi->bn", offset_vec, pred_full_obj_normal)  # TENSOR[NPO, CANDI]
            thresholded_value = constant * torch.pow(
                torch.exp(torch.clamp(-inner_prod - offset, -threshold - offset, threshold - offset)),
                2,
            )
            thresholded_value = thresholded_value * dist_mask
            res = torch.sum(thresholded_value) / torch.sum(calc_mask)
        else:
            res = torch.Tensor([0.0]).float().to(pred_hand_verts.device)
        return res


class ObjectLoss:
    @staticmethod
    def obj_transf_loss(vars_obj_tsl, vars_obj_rot, init_obj_tsl, init_obj_rot):
        tsl_loss = torch.pow((vars_obj_tsl - init_obj_tsl), 2)
        rot_loss = torch.pow((vars_obj_rot - init_obj_rot), 2)
        return torch.sum(tsl_loss, dim=0) + torch.sum(rot_loss, dim=0)


class HandLoss:
    @staticmethod
    def get_edge_idx(face_idx_tensor: torch.Tensor) -> list:
        device = face_idx_tensor.device
        res = []
        face_idx_tensor = face_idx_tensor.long()
        face_idx_list = face_idx_tensor.tolist()
        for item in face_idx_list:
            v_idx_0, v_idx_1, v_idx_2 = item
            if {v_idx_0, v_idx_1} not in res:
                res.append({v_idx_0, v_idx_1})
            if {v_idx_1, v_idx_2} not in res:
                res.append({v_idx_1, v_idx_2})
            if {v_idx_0, v_idx_2} not in res:
                res.append({v_idx_0, v_idx_2})
        res = [list(e) for e in res]
        res = torch.tensor(res).long().to(device)
        return res

    @staticmethod
    def get_edge_len(verts: torch.Tensor, edge_idx: torch.Tensor):
        # verts: TENSOR[NVERT, 3]
        # edge_idx: TENSOR[NEDGE, 2]
        return torch.norm(verts[edge_idx[:, 0], :] - verts[edge_idx[:, 1], :], p=2, dim=1)

    @staticmethod
    def pose_quat_norm_loss(var_pose):
        """ this is the only loss accepts unnormalized quats """
        reshaped_var_pose = var_pose.reshape((16, 4))  # TENSOR[16, 4]
        quat_norm_sq = quaternion_norm_squared(reshaped_var_pose)  # TENSOR[16, ]
        squared_norm_diff = quat_norm_sq - 1.0  # TENSOR[16, ]
        res = torch.mean(torch.pow(squared_norm_diff, 2), dim=0)
        return res

    @staticmethod
    def pose_reg_loss(var_pose_normed, var_pose_init):
        # the format of quat is [w, x, y, z]
        # to regularize
        # just to make sure w is close to 1.0
        # working aside with self.pose_quat_norm_loss defined above
        inv_var_pose_init = quaternion_inv(var_pose_init)
        combined_pose = quaternion_mul(var_pose_normed, inv_var_pose_init)
        w = combined_pose[..., 0]  # get w
        diff = w - 1.0  # TENSOR[16, ]
        res = torch.mean(torch.pow(diff, 2), dim=0)
        return res

    @staticmethod
    def shape_reg_loss(var_shape, shape_init):
        return torch.sum(torch.pow(var_shape - shape_init, 2), dim=0)

    @staticmethod
    def edge_len_loss(rebuild_verts, hand_edges, static_edge_len):
        pred_edge_len = HandLoss.get_edge_len(rebuild_verts, hand_edges)
        diff = pred_edge_len - static_edge_len  # TENSOR[NEDGE, ]
        return torch.mean(torch.pow(diff, 2), dim=0)

    # **** axis order right hand

    #         14-13-12-\
    #                   \
    #    2-- 1 -- 0 -----*
    #   5 -- 4 -- 3 ----/
    #   11 - 10 - 9 ---/
    #    8-- 7 -- 6 --/

    @staticmethod
    def joint_b_axis_loss(b_axis, axis):
        b_soft_idx = [0, 3, 9, 6, 14]
        b_thumb_soft_idx = [12, 13]
        b_axis = b_axis.squeeze(0)  # [15, 3]

        b_axis_cos = torch.einsum("bi,bi->b", b_axis, axis)
        restrict_cos = b_axis_cos[[i for i in range(15) if i not in b_soft_idx and i not in b_thumb_soft_idx]]
        soft_loss = torch.relu(torch.abs(b_axis_cos[b_soft_idx]) - np.cos(pi / 2 - pi / 36))  # [-5, 5]
        thumb_soft_loss = torch.relu(torch.abs(b_axis_cos[b_thumb_soft_idx]) - np.cos(pi / 2 - pi / 3))  # [-60, 60]

        res = (
            torch.mean(torch.pow(restrict_cos, 2), dim=0)
            + torch.mean(torch.pow(soft_loss, 2), dim=0)
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2), dim=0)
        )
        return res

    @staticmethod
    def joint_u_axis_loss(u_axis, axis):
        u_soft_idx = [0, 3, 9, 6, 14]
        u_thumb_soft_idx = [12, 13]
        u_axis = u_axis.squeeze(0)  # [15, 3]

        u_axis_cos = torch.einsum("bi,bi->b", u_axis, axis)
        restrict_cos = u_axis_cos[[i for i in range(15) if i not in u_soft_idx and i not in u_thumb_soft_idx]]
        soft_loss = torch.relu(torch.abs(u_axis_cos[u_soft_idx]) - np.cos(pi / 2 - pi / 18))  # [-10, 10]
        thumb_soft_loss = torch.relu(torch.abs(u_axis_cos[u_thumb_soft_idx]) - np.cos(pi / 2 - pi / 3))  # [-60, 60]

        res = (
            torch.mean(torch.pow(restrict_cos, 2), dim=0)
            + torch.mean(torch.pow(soft_loss, 2), dim=0)
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2), dim=0)
        )
        return res

    @staticmethod
    def joint_l_limit_loss(l_axis, axis):
        l_soft_idx = [0, 3, 9, 6, 14]
        l_thumb_soft_idx = [12, 13]
        l_axis = l_axis.squeeze(0)  # [15, 3]
        l_axis_cos = torch.einsum("bi,bi->b", l_axis, axis)
        restrict_cos = l_axis_cos[[i for i in range(15) if i not in l_soft_idx and i not in l_thumb_soft_idx]]
        soft_loss = torch.relu(-l_axis_cos[l_soft_idx] + 1 - np.cos(pi / 2 - pi / 9))  # [-20, 20]
        thumb_soft_loss = torch.relu(-l_axis_cos[l_thumb_soft_idx] + 1 - np.cos(pi / 2 - pi / 3))

        res = (
            torch.mean(torch.pow(restrict_cos - 1, 2), dim=0)
            + torch.mean(torch.pow(soft_loss, 2), dim=0)
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2), dim=0)
        )
        return res

    @staticmethod
    def rotation_angle_loss(angle, limit_angle=pi / 2, eps=1e-10):
        angle_new = torch.zeros_like(angle)  # TENSOR[15, ]
        nonzero_mask = torch.abs(angle) > eps  # TENSOR[15, ], bool
        angle_new[nonzero_mask] = angle[nonzero_mask]  # if angle is too small, pick them out of backward graph
        angle_over_limit = torch.relu(angle_new - limit_angle)  # < pi/2, 0; > pi/2, linear | Tensor[16, ]
        angle_over_limit_squared = torch.pow(angle_over_limit, 2)  # TENSOR[15, ]
        res = torch.mean(angle_over_limit_squared, dim=0)
        return res

    @staticmethod
    def hand_tsl_loss(var_hand_tsl, init_hand_tsl):
        return torch.sum(torch.pow(var_hand_tsl - init_hand_tsl, 2))