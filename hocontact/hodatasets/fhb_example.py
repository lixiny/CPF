import os
import pickle
import re

import numpy as np
import torch
from PIL import Image
from liegroups import SO3
from termcolor import cprint

from hocontact.hodatasets import fhbutils
from hocontact.hodatasets.hodata import HOdata
from hocontact.hodatasets.hoquery import BaseQueries, get_trans_queries
from hocontact.utils import meshutils


def transform_obj_verts(verts, transf, cam_extr):
    verts = verts * 1000
    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    transf_verts = transf.dot(hom_verts.T).T
    transf_verts = cam_extr.dot(transf_verts.transpose()).transpose()[:, :3]
    return transf_verts


class FHBExample(HOdata):
    matcher = re.compile(r"^(Subject_[0-9]+)\/(.*?)\/([0-9]+)\/color\/color_([0-9]+)\.jpeg$")

    def __init__(
        self,
        data_root="data",
        data_split="example",
        split_mode="example",
        njoints=21,
        use_cache=True,
        filter_no_contact=True,
        filter_thresh=5.0,  # mm
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
        self.name = "fhbhands_example"
        self.split_mode = split_mode
        self.reduce_res = True
        self.full_image = True

        self.all_queries.update(
            {
                BaseQueries.IMAGE,
                BaseQueries.JOINTS_2D,
                BaseQueries.JOINTS_3D,
                BaseQueries.OBJ_VERTS_2D,
                BaseQueries.OBJ_VERTS_3D,
                BaseQueries.HAND_FACES,
                BaseQueries.HAND_VERTS_3D,
                BaseQueries.HAND_VERTS_2D,
                BaseQueries.OBJ_FACES,
                BaseQueries.OBJ_CAN_VERTS,
                BaseQueries.SIDE,
                BaseQueries.CAM_INTR,
                BaseQueries.OBJ_TRANSF,
                BaseQueries.IMAGE_PATH,
            }
        )
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.update(trans_queries)

        self.queries = query
        if query is None:
            self.queries = self.all_queries

        # Get camera info
        self.cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]])

        self.reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

        self.idxs = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19]

        # sample_list
        self.root = os.path.join(self.data_root, self.name)
        self.root_extra_info = os.path.normpath("assets")
        self.sample_list_path = os.path.join(self.root, "sample_list.txt")

    def load_dataset(self):
        # basic attr
        self.reduce_factor = float(1 / 4)
        self.image_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)]
        self.inp_res = self.image_size

        all_objects = ["juice", "liquid_soap", "milk", "salt"]

        with open(self.sample_list_path, "r") as fstream:
            contents = fstream.read()
        sample_list_lines = [x for x in contents.split("\n") if len(x) > 0]

        self.image_names = []
        self.joints3d = []
        self.joints2d = []
        self.objnames = []
        self.objtransforms = []
        self.mano_infos = []
        for sample_keyword in sample_list_lines:
            match_res = FHBExample.matcher.match(sample_keyword)
            subject_key, action_key, seq_id, frame_id = (
                match_res.group(1),
                match_res.group(2),
                match_res.group(3),
                match_res.group(4),
            )
            save_keyword = "__".join((subject_key, action_key, seq_id, frame_id))
            # image_names
            self.image_names.append(os.path.join(self.root, "images", f"{save_keyword}.jpeg"))
            # annotations
            with open(os.path.join(self.root, "annotations", f"{save_keyword}.pkl"), "rb") as fstream:
                anno_dict = pickle.load(fstream)
            self.joints3d.append(anno_dict["joints_3d"])
            self.joints2d.append(anno_dict["joints_2d"])  # reduced
            self.objnames.append(anno_dict["obj_name"])
            self.objtransforms.append(anno_dict["obj_transf"])
            self.mano_infos.append(anno_dict["mano_info"])

        self.split_objects = fhbutils.load_objects(
            obj_root=os.path.join(self.root, "object_models"), object_names=all_objects
        )
        self.split_objects_normal = fhbutils.load_objects_normal(
            obj_root=os.path.join(self.root, "object_models"), object_names=all_objects
        )
        self.fhb_objects_voxel = fhbutils.load_objects_voxel(
            obj_root=os.path.join(self.root, "object_models"), object_names=all_objects,
        )

        self.hand_palm_vertex_index = np.loadtxt(os.path.join(self.root_extra_info, "hand_palm_full.txt"), dtype=np.int)

        # reduce cam_intr
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor

        cprint(f"Got {len(self.image_names)} samples for data_split {self.data_split}")

    def __len__(self):
        return len(self.image_names)

    def get_image(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        return self.image_names[idx]

    def _fhb_get_hand_info(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"], mano_info["trans"], mano_info["shape"]

    def get_hand_verts3d(self, idx):
        pose, trans, shape = self._fhb_get_hand_info(idx)
        verts, _ = self.layer(torch.Tensor(pose).unsqueeze(0), torch.Tensor(shape).unsqueeze(0))
        verts = verts[0].numpy() + trans
        return np.array(verts).astype(np.float32)

    def get_hand_verts2d(self, idx):
        verts = self.get_hand_verts3d(idx)
        hom_2d = np.array(self.cam_intr).dot(verts.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return np.array(verts2d).astype(np.float32)

    def get_hand_faces(self, idx):
        faces = np.array(self.layer.th_faces).astype(np.long)
        return faces

    def get_obj_faces(self, idx):
        obj = self.objnames[idx]
        objfaces = self.split_objects[obj]["faces"]
        return np.array(objfaces).astype(np.int32)

    def get_obj_transf(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_pose(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_rot(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, :3]

    def get_obj_tsl(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, 3]

    def get_obj_normal(self, idx):
        obj = self.objnames[idx]
        normal = self.split_objects_normal[obj]
        return np.array(normal).astype(np.float32)

    def get_obj_verts_transf(self, idx):
        obj = self.objnames[idx]
        transf = self.objtransforms[idx]
        verts_raw = self.split_objects[obj]["verts"]
        transf_verts = fhbutils.transform_obj_verts(verts_raw, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)

    def get_obj_verts_can(self, idx):
        obj = self.objnames[idx]
        verts = self.split_objects[obj]["verts"]
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)
        return verts_can, bbox_center, bbox_scale

    def get_obj_verts2d(self, idx):
        objpoints3d = self.get_obj_verts_transf(idx)
        objpoints3d = objpoints3d * 1000
        hom_2d = np.array(self.cam_intr).dot(objpoints3d.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return verts2d.astype(np.float32)

    def get_obj_transf_wrt_cam(self, idx):
        verts_can, v_0, _ = self.get_obj_verts_can(idx)

        transf = self.objtransforms[idx]
        transf = self.cam_extr @ transf
        rot = transf[:3, :3]
        tsl = transf[:3, 3] / 1000.0
        tsl_wrt_cam = rot.dot(v_0) + tsl
        tsl_wrt_cam = tsl_wrt_cam[:, np.newaxis]  # (3, 1)

        obj_transf = np.concatenate([rot, tsl_wrt_cam], axis=1)  # (3, 4)
        obj_transf = np.concatenate([obj_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return obj_transf.astype(np.float32)

    def get_joints3d(self, idx):
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints2d(self, idx):
        joints = self.joints2d[idx]
        return joints

    def get_cam_intr(self, idx):
        camintr = self.cam_intr
        return camintr.astype(np.float32)

    def get_sides(self, idx):
        return "right"

    def get_center_scale(self, idx):
        if self.full_image:
            center = np.array((480 / 2, 270 / 2))
            scale = 480
        else:
            raise NotImplementedError()
        return center, scale

    def get_hand_tsl_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["trans"].astype(np.float32)

    def get_hand_shape(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["shape"].astype(np.float32)

    def get_hand_pose_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"].astype(np.float32)

    def get_hand_axisang_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"][0:3].astype(np.float32)

    def get_hand_rot_wrt_cam(self, idx):
        axisang = self.get_hand_axisang_wrt_cam(idx)
        rot = SO3.exp(axisang).as_matrix()
        return rot.astype(np.float32)

    def get_obj_voxel_points_can(self, idx):
        obj = self.objnames[idx]
        objvoxpts = self.fhb_objects_voxel[obj]["points"]
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(objvoxpts, scale=False)
        return verts_can

    def get_obj_voxel_points_transf(self, idx):
        obj = self.objnames[idx]
        transf = self.objtransforms[idx]
        objvoxpts = self.fhb_objects_voxel[obj]["points"]
        transf_verts = fhbutils.transform_obj_verts(objvoxpts, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)

    def get_obj_voxel_element_volume(self, idx):
        obj = self.objnames[idx]
        objvoxelvol = self.fhb_objects_voxel[obj]["element_volume"]
        return objvoxelvol


    def get_hand_palm_vert_idx(self, idx):
        return len(self.hand_palm_vertex_index)

    def get_hand_vis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts2d(idx))
        return handvis

    def get_joint_vis(self, idx):
        return np.ones(self.njoints)

    def get_n_hand_palm_vert(self, idx):
        return len(self.hand_palm_vertex_index)

    def get_obj_textures(self, idx):
        obj = self.objnames[idx]
        objtextures = self.split_objects[obj]["textures"]
        return np.array(objtextures)

    def get_obj_vis2d(self, idx):
        objvis = np.ones_like(self.get_obj_verts2d(idx))
        return objvis

    def get_sample_identifier(self, idx):
        identifier = (
            f"{self.data_split}_{self.split_mode}_mf{self.mini_factor}"
            f"_rf{self.reduce_factor}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec"
        )

        res = f"{self.name}/{identifier}/{idx}"
        return res
