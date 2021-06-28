import os
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from liegroups import SO3
from manopth.anchorutils import anchor_load_driver
from tqdm import tqdm

import hocontact.hodatasets.hodata as hodata
from hocontact.hodatasets import ho3dutils
from hocontact.hodatasets.hodata import HOdata
from hocontact.hodatasets.hoquery import BaseQueries, get_trans_queries
from hocontact.utils import meshutils
from hocontact.utils.logger import logger


class HO3D(hodata.HOdata):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        njoints=21,
        use_cache=True,
        enable_contact=False,
        filter_no_contact=True,
        filter_thresh=10.0,
        mini_factor=1.0,
        center_idx=9,
        scale_jittering=0.0,
        center_jittering=0.0,
        block_rot=False,
        max_rot=0.0 * np.pi,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        query=None,
        sides="right",
        # *======== HO3D >>>>>>>>>>>>>>>>>>
        split_mode="objects",
        like_v1=True,
        full_image=True,
        full_sequences=False,
        contact_pad_vertex=True,
        contact_pad_anchor=True,
        contact_range_th=1000.0,
        contact_elasti_th=0.00,
        load_objects_reduced=False,
        load_objects_color=False,
        load_objects_voxel=False,
        # *<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        **kwargs,
    ):
        super().__init__(
            data_root,
            data_split,
            njoints,
            use_cache,
            filter_no_contact,
            filter_thresh,
            mini_factor,
            center_idx,
            scale_jittering,
            center_jittering,
            block_rot,
            max_rot,
            hue,
            saturation,
            contrast,
            brightness,
            blur_radius,
            query,
            sides,
        )
        self.name = "HO3D"
        self.split_mode = split_mode
        self.like_v1 = like_v1
        self.full_image = full_image
        self.full_sequences = full_sequences
        self.load_objects_reduced = load_objects_reduced
        self.load_objects_color = load_objects_color
        self.load_objects_voxel = load_objects_voxel
        self.enable_contact = enable_contact

        self.image_size = [640, 480]
        self.inp_res = self.image_size
        self.root_extra_info = os.path.normpath("assets")

        self.contact_pad_vertex = contact_pad_vertex
        self.contact_pad_anchor = contact_pad_anchor
        self.contact_range_th = contact_range_th
        self.contact_elasti_th = contact_elasti_th

        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

        self.all_queries.update(
            {
                BaseQueries.OBJ_VIS_2D,
                BaseQueries.OBJ_CORNERS_2D,
                BaseQueries.OBJ_CORNERS_3D,
                BaseQueries.OBJ_CAN_CORNERS,
                BaseQueries.OBJ_FACES,
                BaseQueries.HAND_VIS_2D,
            }
        )
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.update(trans_queries)

        # Fix dataset split
        valid_splits = ["train", "trainval", "val", "test", "all", "all_all"]
        assert self.data_split in valid_splits, "{} not in {}".format(self.data_split, valid_splits)

        self.cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # this camera extrinsic has no translation
        # and this is the reason transforms in following code just use rotation part

    def _preload(self):
        # ! ALL PATH SETTING SHOULD IN THIS PRE-LOAD
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")

        self.cache_path = os.path.join("common", "cache", self.name)
        self.cache_path = os.path.join(
            self.cache_path,
            f"{self.data_split}_{self.split_mode}_mf{round(self.mini_factor)}"
            f"_likev1{''if self.like_v1 else '(x)'}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
            f".pkl",
        )
        self.reduce_factor = 1.0

    def load_dataset(self):

        self._preload()
        self.obj_meshes = ho3dutils.load_objects(os.path.join(self.data_root, "YCB_models_supp"))
        self.obj_normals = ho3dutils.load_obj_normals(os.path.join(self.data_root, "YCB_models_supp"))

        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if self.split_mode == "objects":
            seqs, subfolder = ho3dutils.get_object_seqs(self.data_split, self.like_v1, self.name)
            logger.info(f"{self.name} {self.data_split} set has sequence {seqs}", "yellow")
            seq_frames, subfolder = self.load_seq_frames(subfolder, seqs)
        elif self.split_mode == "paper":
            seq_frames, subfolder = self.load_seq_frames()
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}", "yellow")
        elif self.split_mode == "official":
            seq_frames, subfolder = ho3dutils.get_offi_frames(self.name, self.data_split, self.root)
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}", "yellow")
        else:
            assert False

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache information for dataset {self.name} from {self.cache_path}")
        else:
            seq_map, idxs = self.load_annots(obj_meshes=self.obj_meshes, seq_frames=seq_frames, subfolder=subfolder)

            annotations = {
                "idxs": idxs,
                "seq_map": seq_map,
            }

            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)
            logger.info("Wrote cache for dataset {} to {}".format(self.name, self.cache_path), "yellow")

        self.idxs = annotations["idxs"]
        self.seq_map = annotations["seq_map"]
        self.hand_palm_vertex_index = np.loadtxt(os.path.join(self.root_extra_info, "hand_palm_full.txt"), dtype=np.int)
        self.n_palm_vert = self.get_n_hand_palm_vert(0)
        if self.enable_contact:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_extra_info)

        if self.load_objects_reduced:
            self.obj_meshes_reduced = ho3dutils.load_objects_reduced(os.path.join(self.data_root, "YCB_models_supp"))
        if self.load_objects_voxel:
            self.obj_voxels = ho3dutils.load_objects_voxel(os.path.join(self.data_root, "YCB_models_supp"))

        logger.info(f"{self.name} Got {len(self)} samples for data_split {self.data_split}")
        logger.warn(f"Got {len(self)} samples for data_split {self.data_split}")

    def load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.split_mode == "paper":
            if self.data_split in ["train", "trainval", "val"]:
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
            if self.data_split == "trainval":
                seq_frames = seq_frames[:trainval_idx]
            elif self.data_split == "val":
                seq_frames = seq_frames[trainval_idx:]
        elif self.split_mode == "objects":
            seq_frames = []
            for seq in sorted(seqs):
                seq_folder = os.path.join(self.root, subfolder, seq)
                meta_folder = os.path.join(seq_folder, "meta")
                img_nb = len(os.listdir(meta_folder))
                for img_idx in range(img_nb):
                    seq_frames.append([seq, f"{img_idx:04d}"])
        else:
            assert False
        return seq_frames, subfolder

    def load_annots(self, obj_meshes={}, seq_frames=[], subfolder="train"):
        """
        Args:
            split (str): HO3DV2 split in [train|trainval|val|test]
                train = trainval U(nion) val
            rand_size (int): synthetic data counts
                will be 0 if you want to use the vanilla data
        """

        vhand_path = os.path.join(self.root_extra_info, "hand_palm_full.txt")
        vid = np.loadtxt(vhand_path, dtype=np.int)

        idxs = []
        seq_map = defaultdict(list)
        seq_counts = defaultdict(int)
        for idx_count, (seq, frame_idx) in enumerate(tqdm(seq_frames)):
            if int(frame_idx) % round(self.mini_factor) != 0:
                continue
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")

            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)
                if annot["handJoints3D"].size == 3:
                    annot["handTrans"] = annot["handJoints3D"]
                    annot["handJoints3D"] = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)
                    annot["handPose"] = np.zeros(48, dtype=np.float32)
                    annot["handBeta"] = np.zeros(10, dtype=np.float32)

            # filter no contact
            if self.filter_no_contact and ho3dutils.min_contact_dis(annot, obj_meshes, vid) > self.filter_thresh:
                continue

            # ? this is the vanilla data
            img_path = os.path.join(rgb_folder, f"{frame_idx}.png")
            annot["img"] = img_path
            annot["frame_idx"] = frame_idx

            seq_map[seq].append(annot)
            idxs.append((seq, seq_counts[seq]))
            seq_counts[seq] += 1

        return seq_map, idxs

    def get_seq_frame(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        frame_idx = annot["frame_idx"]
        return seq, frame_idx

    def get_image_path(self, idx):
        seq, img_idx = self.idxs[idx]
        img_path = self.seq_map[seq][img_idx]["img"]
        return img_path

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = Image.open(img_path).convert("RGB")
        return img

    def get_joint_vis(self, idx):
        return np.ones(self.njoints)

    def get_joints2d(self, idx):
        joints3d = self.get_joints3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return self.project(joints3d, cam_intr)

    def get_joints3d(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        joints3d = annot["handJoints3D"]
        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        joints3d = joints3d[self.reorder_idxs]
        return joints3d.astype(np.float32)

    def get_obj_textures(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        textures = self.obj_meshes[obj_id]["textures"]
        return textures

    def _ho3d_get_hand_info(self, idx):
        """
        Get the hand annotation in the raw ho3d datasets.
        !!! This Mehthods shoudln't be called outside.
        :param idx:
        :return: raw hand pose, translate and shape coefficients
        """
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        # Retrieve hand info
        handpose = annot["handPose"]
        handtsl = annot["handTrans"]
        handshape = annot["handBeta"]
        return handpose, handtsl, handshape

    def get_hand_verts3d(self, idx):
        _handpose, _handtsl, _handshape = self._ho3d_get_hand_info(idx)
        handverts, handjoints = self.layer(
            torch.from_numpy(_handpose).unsqueeze(0), torch.from_numpy(_handshape).unsqueeze(0),
        )
        # important modify!!!!
        handverts = handverts[0].numpy() + _handtsl
        transf_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
        return transf_handverts.astype(np.float32)

    def get_hand_verts2d(self, idx):
        verts3d = self.get_hand_verts3d(idx)
        cam_intr = self.get_cam_intr(idx)
        verts2d = self.project(verts3d, cam_intr)
        return verts2d

    def get_hand_faces(self, idx):
        faces = np.array(self.layer.th_faces).astype(np.long)
        return faces

    def get_obj_faces(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        objfaces = self.obj_meshes[obj_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int32)
        return objfaces

    def get_obj_normal(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        obj_normal = self.obj_normals[obj_id]
        obj_normal = (self.cam_extr[:3, :3] @ np.array(obj_normal).T).T
        return obj_normal.astype(np.float32)

    def get_obj_name(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        return annot["objName"]

    def get_obj_verts_can(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()

        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can, bbox_center, bbox_scale

    # ? only used for render
    def get_obj_full_color_can(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        verts = self.obj_meshes_full_color[obj_id]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()

        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can, self.obj_meshes_full_color[obj_id]["faces"], self.obj_meshes_full_color[obj_id]["vc"]

    def get_obj_verts_transf(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        obj_id = annot["objName"]

        # This verts IS NOT EQUAL to the one in get_obj_verts_can,
        # since this verts is not translated to vertices center
        verts = self.obj_meshes[obj_id]["verts"]
        transf_verts = rot.dot(verts.transpose()).transpose() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(transf_verts.transpose()).transpose()
        return np.array(transf_verts).astype(np.float32)

    def get_obj_transf_wrt_cam(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]

        verts_can, v_0, _ = self.get_obj_verts_can(idx)  # (N, 3), (3, ), 1

        """ HACK
        v_{can} = E * v_{raw} - v_0
        v_{cam} = E * (R * v_{raw} + t)

        => v_{raw} = E^{-1} * (v_{can} + v_0)
        => v_{cam} = E * (R * (E^{-1} * (v_{can} + v_0)) + t)
        =>         = E*R*E^{-1} * v_{can} + E*R*E^{-1} * v_0 + E * t
        """

        ext_rot = self.cam_extr[:3, :3]
        ext_rot_inv = np.linalg.inv(ext_rot)

        rot_wrt_cam = ext_rot @ (rot @ ext_rot_inv)  # (3, 3)
        tsl_wrt_cam = (ext_rot @ (rot @ ext_rot_inv)).dot(v_0) + ext_rot.dot(tsl)  # (3,)
        tsl_wrt_cam = tsl_wrt_cam[:, np.newaxis]  # (3, 1)

        obj_transf = np.concatenate([rot_wrt_cam, tsl_wrt_cam], axis=1)  # (3, 4)
        obj_transf = np.concatenate([obj_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return obj_transf

    # ? for compatibility
    def get_obj_transf(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    # ? for compatibility
    def get_obj_pose(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_rot(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, :3]

    def get_obj_tsl(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, 3]

    def _get_obj_corners3d(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        corners = annot["objCorners3DRest"]
        trans_corners = rot.dot(corners.transpose()).transpose() + tsl
        trans_corners = self.cam_extr[:3, :3].dot(trans_corners.transpose()).transpose()
        obj_corners = np.array(trans_corners).astype(np.float32)
        return obj_corners

    def get_obj_corners3d(self, idx):
        corners = self.get_obj_corners_can(idx)
        obj_transf = self.get_obj_transf_wrt_cam(idx)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)
        obj_corners_transf = (obj_rot.dot(corners.transpose()) + obj_tsl).transpose()
        return obj_corners_transf.astype(np.float32)

    def get_obj_corners_can(self, idx):
        _, obj_cantrans, obj_canscale = self.get_obj_verts_can(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        corners = annot["objCorners3DRest"]
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        obj_cancorners = (corners - obj_cantrans) / obj_canscale
        return obj_cancorners.astype(np.float32)

    def get_obj_corners2d(self, idx):
        corners3d = self.get_obj_corners3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return self.project(corners3d, cam_intr)

    def get_obj_verts2d(self, idx):
        objpoints3d = self.get_obj_verts_transf(idx)
        cam_intr = self.get_cam_intr(idx)
        verts2d = self.project(objpoints3d, cam_intr)
        return verts2d

    def get_obj_vis2d(self, idx):
        objvis = np.ones_like(self.get_obj_verts2d(idx)[:, 0])
        return objvis

    def get_hand_vis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts2d(idx)[:, 0])
        return handvis

    def get_sides(self, idx):
        return "right"

    def get_cam_intr(self, idx):
        seq, img_idx = self.idxs[idx]
        cam_intr = self.seq_map[seq][img_idx]["camMat"]
        return cam_intr

    def get_hand_palm_vert_idx(self, _):
        return self.hand_palm_vertex_index

    def get_n_hand_palm_vert(self, _):
        return len(self.hand_palm_vertex_index)

    def __len__(self):
        return len(self.idxs)

    def get_center_scale(self, idx):
        if self.full_image:
            center = np.array([640 // 2, 480 // 2])
            scale = 640
        else:
            logger.error("Non full_image mode is not implements")
            raise NotImplementedError()
        return center, scale

    def get_annot(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        return annot

    def get_sample_identifier(self, idx):
        identifier = (
            f"{self.data_split}_{self.split_mode}_mf{round(self.mini_factor)}"
            f"_likev1{''if self.like_v1 else '(x)'}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
        )

        res = f"{self.name}/{identifier}/{idx}"
        return res

    # ? only used in offline eval
    def get_hand_pose_wrt_cam(self, idx):  # pose = root_rot + ...
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root, remains = handpose[:3], handpose[3:]
        root = SO3.exp(root).as_matrix()
        root = self.cam_extr[:3, :3] @ root
        root = SO3.log(SO3.from_matrix(root, normalize=True))
        handpose_transformed = np.concatenate((root, remains), axis=0)
        return handpose_transformed.astype(np.float32)

    # ? only used in offline eval
    def get_hand_tsl_wrt_cam(self, idx):
        hand_pose = torch.from_numpy(self.get_hand_pose_wrt_cam(idx)).unsqueeze(0)
        hand_shape = torch.from_numpy(self.get_hand_shape(idx)).unsqueeze(0)

        hand_verts, _ = self.layer(hand_pose, hand_shape)
        hand_verts = np.array(hand_verts.squeeze(0))
        tsl = self.get_hand_verts3d(idx) - hand_verts
        return tsl[0]

    # ? only used in offline eval
    def get_hand_axisang_wrt_cam(self, idx):
        root = self.get_hand_rot_wrt_cam(idx)
        root = SO3.log(SO3.from_matrix(root, normalize=True))
        return root.astype(np.float32)

    # ? only used in offline eval
    def get_hand_rot_wrt_cam(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root = handpose[:3]
        root = SO3.exp(root).as_matrix()
        root = self.cam_extr[:3, :3] @ root
        return root.astype(np.float32)

    # ? only used in offline eval
    def get_hand_shape(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        handshape = annot["handBeta"]
        return handshape.astype(np.float32)

    # ? for compatibility
    def get_hand_pose(self, idx):
        return self.get_hand_pose_wrt_cam(idx)

    # ? for compatibility
    def get_hand_tsl(self, idx):
        return self.get_hand_tsl_wrt_cam(idx)

    # ? only used in offline eval
    def get_obj_verts_transf_reduced(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        obj_id = annot["objName"]

        # This verts IS NOT EQUAL to the one in get_obj_verts_can,
        # since this verts is not translated to vertices center
        verts = self.obj_meshes_reduced[obj_id]["verts"]
        transf_verts = rot.dot(verts.transpose()).transpose() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(transf_verts.transpose()).transpose()
        return np.array(transf_verts).astype(np.float32)

    # ? only used in offline eval
    def get_obj_verts_can_reduced(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        verts = self.obj_meshes_reduced[obj_id]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()

        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # CENTERED HERE
        return verts_can, bbox_center, bbox_scale

    # ? only used in offline eval
    def get_obj_faces_reduced(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        objfaces = self.obj_meshes_reduced[obj_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int32)
        return objfaces

    # ? only used in post process
    def get_obj_voxel_points_can(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        verts = self.obj_voxels[obj_id]["points"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()

        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can

    # ? only used in post process
    def get_obj_voxel_points_transf(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        obj_id = annot["objName"]

        # This verts IS NOT EQUAL to the one in get_obj_verts_can,
        # since this verts is not translated to vertices center
        verts = self.obj_voxels[obj_id]["points"]
        transf_verts = rot.dot(verts.transpose()).transpose() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(transf_verts.transpose()).transpose()
        return np.array(transf_verts)

    # ? only used in post process
    def get_obj_voxel_element_volume(self, idx):
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        objvoxelvol = self.obj_voxels[obj_id]["element_volume"]
        return objvoxelvol


def view_data(ho_dataset):
    for i in tqdm(range(len(ho_dataset))):
        # i = len(ho_dataset) - 1 - i
        objverts2d = ho_dataset.get_obj_verts2d(i)
        joint2d = ho_dataset.get_joints2d(i)

        # TEST: obj_transf @ obj_verts_can == obj_verts_transf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        obj_transf = ho_dataset.get_obj_transf_wrt_cam(i)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)

        obj_verts_can, _, __ = ho_dataset.get_obj_verts_can(i)  # (N, 3)
        obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()
        obj_verts2d_pred = ho_dataset.project(obj_verts_pred, ho_dataset.get_cam_intr(i))

        # TEST: obj_transf @ obj_corners_can == obj_corners_3d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        obj_corners = ho_dataset.get_obj_corners3d(i)
        obj_corners2d = ho_dataset.project(obj_corners, ho_dataset.get_cam_intr(i))

        # TEST: MANO(get_hand_pose_wrt_cam) + get_hand_tsl_wrt_cam == get_hand_verts3d >>>>>>>>>>>>>>>>>>>>>>>>>>
        hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
        hand_shape = torch.from_numpy(ho_dataset.get_hand_shape(i)).unsqueeze(0)
        hand_tsl = ho_dataset.get_hand_tsl_wrt_cam(i)

        hand_verts, hand_joints = ho_dataset.layer(hand_pose, hand_shape)
        hand_verts = np.array(hand_verts.squeeze(0)) + hand_tsl
        hand_verts_2d = ho_dataset.project(hand_verts, ho_dataset.get_cam_intr(i))

        hand_verts_2dgt = ho_dataset.get_hand_verts2d(i)
        img = ho_dataset.get_image(i)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j in range(obj_verts2d_pred.shape[0]):
            v = obj_verts2d_pred[j]
            cv2.circle(img, (v[0], v[1]), radius=2, thickness=-1, color=(0, 255, 0))
        for j in range(objverts2d.shape[0]):
            v = objverts2d[j]
            cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(0, 255, 255))
        for j in range(hand_verts_2dgt.shape[0]):
            v = hand_verts_2dgt[j]
            cv2.circle(img, (v[0], v[1]), radius=3, thickness=-1, color=(255, 255, 0))
        for j in range(hand_verts_2d.shape[0]):
            v = hand_verts_2d[j]
            cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(255, 0, 0))
        for j in range(obj_corners2d.shape[0]):
            v = obj_corners2d[j]
            cv2.circle(img, (v[0], v[1]), radius=8, thickness=-1, color=(0, 0, 255))
        cv2.imshow("ho3d", img)
        cv2.waitKey(1)


def main(args):
    ho_dataset = HOdata.get_dataset(
        dataset="ho3d",
        data_root="data",
        data_split="train",
        split_mode="objects",
        use_cache=False,
        mini_factor=30.0,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=False,
        filter_thresh=10.0,
        block_rot=True,
        synt_factor=1,
    )

    import prettytable as pt
    from hocontact.utils.logger import logger

    print(len(ho_dataset))
    idx = np.random.randint(len(ho_dataset))

    sample = ho_dataset[idx]
    tb = pt.PrettyTable(padding_width=3, header=False)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            tb.add_row([key, type(value), value.shape])
        elif isinstance(value, torch.Tensor):
            tb.add_row([key, type(value), tuple(value.size())])
        else:
            tb.add_row([key, type(value), value])
    logger.warn(f"{'=' * 40} ALL HO3D SAMPLE KEYS {'>' * 40}", "blue")
    logger.info(str(tb))

    if args.vis:
        view_data(ho_dataset)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test ho3d dataset")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--data_split", choices=["train", "test", "all"], default="train", type=str)
    parser.add_argument("--split_mode", choices=["paper", "objects"], default="objects", type=str)
    main(parser.parse_args())
