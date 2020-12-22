import numpy as np
import torch

import hocontact.config as cfg


def get_annot_scale(annots, visibility=None, scale_factor=2.3):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s


def get_annot_center(annots, visibility=None):
    # Get scale
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])


def kin_chain_from_joint(joint):
    kin_chain = [
        joint[:, i, :] - joint[:, cfg.SNAP_PARENT[i], :]
        for i in range(21)
    ]
    kin_chain = kin_chain[1:]  # id 0's parent is itself
    kin_chain = torch.stack(kin_chain, dim=1)  # (B, 20, 3)
    kin_len = torch.norm(kin_chain, dim=-1, keepdim=True)  # (B, 20, 1)
    kin_chain = kin_chain / kin_len
    return kin_chain, kin_len


def get_joint_bone_len(joint, ref_bone_link=None):
    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if (
            not torch.is_tensor(joint)
            and not isinstance(joint, np.ndarray)
    ):
        raise TypeError('joint should be ndarray or torch tensor. Got {}'.format(type(joint)))
    if (
            len(joint.shape) != 3
            or joint.shape[1] != 21
            or joint.shape[2] != 3
    ):
        raise TypeError('joint should have shape (B, njoint, 3), Got {}'.format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += torch.norm(
                joint[:, jid, :] - joint[:, nextjid, :],
                dim=1, keepdim=True
            )  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += np.linalg.norm(
                (joint[:, jid, :] - joint[:, nextjid, :]),
                ord=2, axis=1, keepdims=True
            )  # (B, 1)
    return bone


def batch_persp_proj(joint, intr):
    joint_homo = torch.matmul(joint, intr.transpose(1, 2))
    joint2d = joint_homo / joint_homo[:, :, 2:]
    joint2d = joint2d[:, :, :2]
    return joint2d


def batch_proj2d(verts, camintr, camextr=None):
    # Project 3d vertices on image plane
    if camextr is not None:
        verts = camextr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def flip_hand_side(target_side, hand_side):
    # Flip if needed
    if target_side == "right" and hand_side == "left":
        flip = True
        hand_side = "right"
    elif target_side == "left" and hand_side == "right":
        flip = True
        hand_side = "left"
    else:
        flip = False
    return hand_side, flip
