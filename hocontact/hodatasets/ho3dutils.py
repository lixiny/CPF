import os

import cv2
import numpy as np
import torch
import trimesh
from manopth import manolayer
from scipy.spatial.distance import cdist


def load_objects(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_simple_ds.obj")
        mesh = trimesh.load(obj_path)
        objects[obj_name] = {"verts": mesh.vertices, "faces": mesh.faces}
    return objects


def load_obj_normals(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    obj_normals = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_normal.obj")
        if not os.path.isfile(obj_path):
            continue
        mesh = trimesh.load(obj_path)
        obj_normals[obj_name] = mesh.vertex_normals
    return obj_normals


def load_objects_reduced(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "ds_plus.obj")
        mesh = trimesh.load(obj_path)
        objects[obj_name] = {"verts": mesh.vertices, "faces": mesh.faces}
    return objects


def load_objects_voxel(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "solid.binvox")
        vox = trimesh.load(obj_path)
        objects[obj_name] = {
            "points": np.array(vox.points),
            "matrix": np.array(vox.matrix),
            "element_volume": vox.element_volume,
        }
    return objects


def load_corners(corner_root):
    obj_corners = {}
    for objname in os.listdir(corner_root):
        filepath = os.path.join(corner_root, objname, "corners.npy")
        corners = np.load(filepath)
        obj_corners[objname] = corners
    return obj_corners


def get_offi_frames(name, split, root, trainval_idx=60000, filter_no_grasp=True):
    offi_train_seqs = {}  # remove train sequences since we only release a test version
    offi_test_seqs = {"SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13"}
    grasp_list = {  # Test sequences are filtered as we mentioned in supplementary materials C.1
        "SM1": [i for i in range(0, 889 + 1)],
        "MPM10": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
        "MPM11": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
        "MPM12": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
        "MPM13": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
        "MPM14": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
        "SB11": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
        "SB13": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    }
    if name != "HO3D":
        root = root.replace(name, "HO3D")
    if split in ["train", "trainval", "val"]:
        info_path = os.path.join(root, "train.txt")
        subfolder = "train"
    elif split == "test":
        info_path = os.path.join(root, "evaluation.txt")
        subfolder = "evaluation"
    else:
        assert False
    with open(info_path, "r") as f:
        lines = f.readlines()
    txt_seq_frames = [line.strip().split("/") for line in lines]
    if split == "trainval":
        txt_seq_frames = txt_seq_frames[:trainval_idx]
    elif split == "val":
        txt_seq_frames = txt_seq_frames[trainval_idx:]
    seqs = {}
    for sf in txt_seq_frames:
        if sf[0] not in offi_train_seqs and sf[0] not in offi_test_seqs:
            continue
        if filter_no_grasp and not (int(sf[1]) in grasp_list[sf[0]]):
            continue
        if sf[0] in seqs:
            seqs[sf[0]].append(sf[1])
        else:
            seqs[sf[0]] = [sf[1]]
    seq_frames = []
    for s in seqs:
        seqs[s].sort()
        for f in range(len(seqs[s])):
            seq_frames.append([s, seqs[s][f]])
    return seq_frames, subfolder


def min_contact_dis(annot, obj_meshes, vid):
    cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if not hasattr(min_contact_dis, "layer"):
        min_contact_dis.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang", use_pca=False, mano_root="assets/mano", center_idx=None, flat_hand_mean=True,
        )
    rot = cv2.Rodrigues(annot["objRot"])[0]
    trans = annot["objTrans"]
    obj_id = annot["objName"]
    verts = obj_meshes[obj_id]["verts"]
    trans_verts = rot.dot(verts.transpose()).transpose() + trans
    trans_verts = cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
    obj_verts = np.array(trans_verts).astype(np.float32)

    handpose = annot["handPose"]
    handtrans = annot["handTrans"]
    handshape = annot["handBeta"]

    handverts, handjoints = min_contact_dis.layer(
        torch.Tensor(handpose).unsqueeze(0), torch.Tensor(handshape).unsqueeze(0)
    )
    handverts = handverts[0].numpy() + handtrans
    trans_handverts = cam_extr[:3, :3].dot(handverts.transpose()).transpose()

    all_dist = cdist(trans_handverts[vid], obj_verts) * 1000

    return all_dist.min()


def get_object_seqs(split, like_v1, name):
    if split == "train":
        if like_v1:
            seqs = {"SM5", "MC6", "MC4", "SM3", "SM4", "SS3", "SS2", "SM2", "SS1", "MC5", "MC1"}
        else:
            seqs = {
                "ABF11",
                "ABF12",
                "ABF13",
                "ABF14",
                "BB10",
                "BB12",
                "BB13",
                "BB14",
                "GPMF10",
                "GPMF11",
                "GPMF13",
                "GPMF14",
                "GSF10",
                "GSF11",
                "GSF12",
                "GSF14",
                "MC2",
                "MC4",
                "MC5",
                "MC6",
                "MDF10",
                "MDF11",
                "MDF12",
                "MDF13",
                "SB10",
                "SB12",
                "ShSu12",
                "ShSu13",
                "ShSu14",
                "SiBF10",
                "SiBF12",
                "SiBF13",
                "SiBF14",
                "SM2",
                "SM4",
                "SM5",
                "SMu40",
                "SMu41",
                "SS1",
                "SS3",
                "SMu42",
            }
        subfolder = "train"
    elif split == "trainval":
        seqs = {
            "ABF12",
            "ABF13",
            "ABF14",
            "BB10",
            "BB13",
            "BB14",
            "GPMF10",
            "GPMF11",
            "GPMF14",
            "GSF10",
            "GSF11",
            "GSF12",
            "MC2",
            "MC4",
            "MC5",
            "MC6",
            "MDF10",
            "MDF11",
            "MDF12",
            "MDF13",
            "SB10",
            "SB12",
            "ShSu12",
            "ShSu13",
            "ShSu14",
            "SiBF10",
            "SiBF12",
            "SiBF13",
            "SiBF14",
            "SM2",
            "SM4",
            "SM5",
            "SMu40",
            "SMu41",
            "SS1",
            "SS3",
            "SMu42",
        }
        subfolder = "train"
    elif split == "val":
        seqs = {"ABF11", "BB12", "GPMF13", "GSF14"}
        subfolder = "train"
    elif split == "test":
        if like_v1:
            seqs = {"MC2"}
        else:
            seqs = {
                "ABF10",
                "MC1",
                "MDF14",
                "BB11",
                "GPMF12",
                "GSF13",
                "SB14",
                "ShSu10",
                "SM3",
                "SMu1",
                "SiBF11",
                "SS2",
            }
        subfolder = "train"
        print(f"Using seqs {seqs} for evaluation")
    elif split == "all":
        if like_v1:
            seqs = {"MC1", "MC2", "MC4", "MC5", "MC6", "SM2", "SM3", "SM4", "SM5", "SS1", "SS2", "SS3"}
        else:
            seqs = {
                "ABF10",
                "ABF11",
                "ABF12",
                "ABF13",
                "ABF14",
                "BB10",
                "BB11",
                "BB12",
                "BB13",
                "BB14",
                "GPMF10",
                "GPMF11",
                "GPMF12",
                "GPMF13",
                "GPMF14",
                "GSF10",
                "GSF11",
                "GSF12",
                "GSF13",
                "GSF14",
                "MC1",
                "MC2",
                "MC4",
                "MC5",
                "MC6",
                "MDF10",
                "MDF11",
                "MDF12",
                "MDF13",
                "MDF14",
                "ND2",  # new in v2
                "SB10",
                "SB12",
                "SB14",
                "SM2",
                "SM3",
                "SM4",
                "SM5",
                "SMu1",
                "SMu40",
                "SMu41",
                "SMu42",
                "SS1",
                "SS2",
                "SS3",
                "ShSu10",
                "ShSu12",
                "ShSu13",
                "ShSu14",
                "SiBF10",
                "SiBF11",
                "SiBF12",
                "SiBF13",
                "SiBF14",
                "SiS1",  # new in v2
            }
        subfolder = "train"
        version_descriptor = "v1" if like_v1 else "v2"
        print(f"Using seqs {seqs} for all, version {version_descriptor}")
    # ! Following splits have nothing to do with like_v1 switch
    # ! Only depend on split name
    elif split == "all_all":
        seqs = {
            "ABF10",
            "ABF11",
            "ABF12",
            "ABF13",
            "ABF14",
            "BB10",
            "BB11",
            "BB12",
            "BB13",
            "BB14",
            "GPMF10",
            "GPMF11",
            "GPMF12",
            "GPMF13",
            "GPMF14",
            "GSF10",
            "GSF11",
            "GSF12",
            "GSF13",
            "GSF14",
            "MC1",
            "MC2",
            "MC4",
            "MC5",
            "MC6",
            "MDF10",
            "MDF11",
            "MDF12",
            "MDF13",
            "MDF14",
            "SB10",
            "SB12",
            "SB14",
            "SM2",
            "SM3",
            "SM4",
            "SM5",
            "SMu1",
            "SMu40",
            "SMu41",
            "SMu42",
            "SS1",
            "SS2",
            "SS3",
            "ShSu10",
            "ShSu12",
            "ShSu13",
            "ShSu14",
            "SiBF10",
            "SiBF11",
            "SiBF12",
            "SiBF13",
            "SiBF14",
        }
        subfolder = "train"
        print(f"Using seqs {seqs} for total_dataset, regardless of version")
    else:
        assert False, "split mode not found!"
    return seqs, subfolder


def get_seq_object(seq):
    mapping = {
        "ABF10": "021_bleach_cleanser",
        "ABF11": "021_bleach_cleanser",
        "ABF12": "021_bleach_cleanser",
        "ABF13": "021_bleach_cleanser",
        "ABF14": "021_bleach_cleanser",
        "BB10": "011_banana",
        "BB11": "011_banana",
        "BB12": "011_banana",
        "BB13": "011_banana",
        "BB14": "011_banana",
        "GPMF10": "010_potted_meat_can",
        "GPMF11": "010_potted_meat_can",
        "GPMF12": "010_potted_meat_can",
        "GPMF13": "010_potted_meat_can",
        "GPMF14": "010_potted_meat_can",
        "GSF10": "037_scissors",
        "GSF11": "037_scissors",
        "GSF12": "037_scissors",
        "GSF13": "037_scissors",
        "GSF14": "037_scissors",
        "MC1": "003_cracker_box",
        "MC2": "003_cracker_box",
        "MC4": "003_cracker_box",
        "MC5": "003_cracker_box",
        "MC6": "003_cracker_box",
        "MDF10": "035_power_drill",
        "MDF11": "035_power_drill",
        "MDF12": "035_power_drill",
        "MDF13": "035_power_drill",
        "MDF14": "035_power_drill",
        "ND2": "035_power_drill",
        "SB10": "021_bleach_cleanser",
        "SB12": "021_bleach_cleanser",
        "SB14": "021_bleach_cleanser",
        "SM2": "006_mustard_bottle",
        "SM3": "006_mustard_bottle",
        "SM4": "006_mustard_bottle",
        "SM5": "006_mustard_bottle",
        "SMu1": "025_mug",
        "SMu40": "025_mug",
        "SMu41": "025_mug",
        "SMu42": "025_mug",
        "SS1": "004_sugar_box",
        "SS2": "004_sugar_box",
        "SS3": "004_sugar_box",
        "ShSu10": "004_sugar_box",
        "ShSu12": "004_sugar_box",
        "ShSu13": "004_sugar_box",
        "ShSu14": "004_sugar_box",
        "SiBF10": "011_banana",
        "SiBF11": "011_banana",
        "SiBF12": "011_banana",
        "SiBF13": "011_banana",
        "SiBF14": "011_banana",
        "SiS1": "004_sugar_box",
        # test
        "SM1": "006_mustard_bottle",
        "MPM10": "010_potted_meat_can",
        "MPM11": "010_potted_meat_can",
        "MPM12": "010_potted_meat_can",
        "MPM13": "010_potted_meat_can",
        "MPM14": "010_potted_meat_can",
        "SB11": "021_bleach_cleanser",
        "SB13": "021_bleach_cleanser",
        "AP10": "019_pitcher_base",
        "AP11": "019_pitcher_base",
        "AP12": "019_pitcher_base",
        "AP13": "019_pitcher_base",
        "AP14": "019_pitcher_base",
    }
    obj_set = set()
    for s in seq:
        obj_set.add(mapping[s])
    return list(obj_set)

