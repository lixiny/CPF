import os
import pickle
import random

import cv2
import numpy as np
import torch
from PIL import Image
from liegroups import SO3
from manopth.anchorutils import anchor_load_driver
from scipy.spatial.distance import cdist

import hocontact.hodatasets.hodata as hodata
from hocontact.hodatasets import fhbutils
from hocontact.hodatasets.hodata import HOdata
from hocontact.hodatasets.hoquery import BaseQueries, get_trans_queries
from hocontact.utils import meshutils
from hocontact.utils.logger import logger


class FPHB(hodata.HOdata):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        split_mode="actions",
        njoints=21,
        use_cache=True,
        filter_no_contact=True,
        filter_thresh=10.0,  # mm
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
        # *======== FHB >>>>>>>>>>>>>>>>>>>
        full_image=True,
        reduce_res=True,
        enable_contact=False,
        contact_pad_vertex=True,
        contact_pad_anchor=True,
        contact_range_th=1000.0,
        contact_elasti_th=0.00,
        for_render=False,
        load_objects_reduced=False,
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
        self.name = "fhbhands"
        self.split_mode = split_mode
        self.reduce_res = reduce_res
        self.for_render = for_render
        self.load_objects_reduced = load_objects_reduced
        self.load_objects_voxel = load_objects_voxel
        self.enable_contact = enable_contact
        self.full_image = full_image
        self.contact_pad_vertex = contact_pad_vertex
        self.contact_pad_anchor = contact_pad_anchor
        self.contact_range_th = contact_range_th
        self.contact_elasti_th = contact_elasti_th

        self.rand_size = 0  # Always ZERO in the parent class

        self.mode_opts = ["actions", "objects", "subjects"]
        self.subjects = [
            "Subject_1",
            "Subject_2",
            "Subject_3",
            "Subject_4",
            "Subject_5",
            "Subject_6",
        ]
        if split_mode not in self.mode_opts:
            raise ValueError(f"Split for dataset {self.name} should be in {self.mode_opts}, got {split_mode}.")

        # get queries
        self.all_queries.update(
            {
                BaseQueries.IMAGE,
                BaseQueries.JOINTS_2D,
                BaseQueries.JOINTS_3D,
                BaseQueries.OBJ_VERTS_2D,
                BaseQueries.OBJ_VIS_2D,
                BaseQueries.OBJ_VERTS_3D,
                BaseQueries.HAND_FACES,
                BaseQueries.HAND_VERTS_3D,
                BaseQueries.HAND_VERTS_2D,
                BaseQueries.OBJ_FACES,
                BaseQueries.OBJ_CAN_VERTS,
                BaseQueries.SIDE,
                BaseQueries.CAM_INTR,
                BaseQueries.JOINT_VIS,
                BaseQueries.OBJ_TRANSF,
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

    def _preload(self):
        # ! ALL PATH SETTING SHOULD IN THIS PRE-LOAD
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_extra_info = os.path.normpath("assets")
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(self.root, "data_split_action_recognition.txt")
        small_rgb_root = os.path.join(self.root, "Video_files_480")
        if os.path.exists(small_rgb_root) and self.reduce_res:
            self.rgb_root = small_rgb_root
            self.reduce_factor = float(1 / 4)
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")
            self.reduce_factor = float(1)
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")

        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite

        self.cache_path = os.path.join("common", "cache", self.name)

        # NOTE: eci for "enable contact info"
        self.cache_path = os.path.join(
            self.cache_path,
            f"{self.data_split}_{self.split_mode}_mf{self.mini_factor}"
            f"_rf{self.reduce_factor}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
            f".pkl",
        )

    def load_dataset(self):

        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        all_objects = ["juice", "liquid_soap", "milk", "salt"]
        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache information for dataset {self.name} from {self.cache_path}")
        else:
            subjects_infos = {}
            for subject in self.subjects:
                subject_info_path = os.path.join(self.info_root, "{}_info.txt".format(subject))
                subjects_infos[subject] = {}
                with open(subject_info_path, "r") as subject_f:
                    raw_lines = subject_f.readlines()
                    for line in raw_lines[3:]:
                        line = " ".join(line.split())
                        action, action_idx, length = line.strip().split(" ")
                        subjects_infos[subject][(action, action_idx)] = length
                    subject_f.close()
            skel_info = fhbutils.get_skeletons(self.skeleton_root, subjects_infos)

            with open(self.info_split, "r") as annot_f:
                lines_raw = annot_f.readlines()
                annot_f.close()
            train_list, test_list, all_infos = fhbutils.get_action_train_test(lines_raw, subjects_infos)

            # use object is always ture
            self.fhb_objects = fhbutils.load_objects(
                obj_root=os.path.join(self.root_supp, "Object_models"), object_names=all_objects,
            )
            self.fhb_objects_normal = fhbutils.load_objects_normal(
                obj_root=os.path.join(self.root_supp, "Object_models"), object_names=all_objects,
            )

            obj_infos = fhbutils.load_object_infos(os.path.join(self.root, "Object_6D_pose_annotation_v1_1"))

            if self.split_mode == "actions":
                if self.data_split == "train":
                    sample_list = train_list
                elif self.data_split == "test":
                    sample_list = test_list
                elif self.data_split == "all":
                    sample_list = {**train_list, **test_list}
                else:
                    logger.error(
                        "Split {} not valid for {}, should be [train|test|all]".format(self.data_split, self.name)
                    )
                    raise KeyError(f"Split {self.data_split} not valid for {self.name}, should be [train|test|all]")
            elif self.split_mode == "subjects":
                if self.data_split == "train":
                    subjects = ["Subject_1", "Subject_3", "Subject_4"]
                elif self.data_split == "test":
                    subjects = ["Subject_2", "Subject_5", "Subject_6"]
                else:
                    logger.error(f"Split {self.data_split} not in [train|test] for split_type subjects")
                    raise KeyError(f"Split {self.data_split} not in [train|test] for split_type subjects")
                self.subjects = subjects
                sample_list = all_infos
            elif self.split_mode == "objects":
                sample_list = all_infos
            else:
                error = logger.error(f"split_type should be in [action|objects|subjects], got {self.split_mode}")
                raise KeyError(error)
            if self.split_mode != "subjects":
                self.subjects = [
                    "Subject_1",
                    "Subject_2",
                    "Subject_3",
                    "Subject_4",
                    "Subject_5",
                    "Subject_6",
                ]
            if self.split_mode != "objects":
                self.split_objects = self.fhb_objects
                self.split_objects_normal = self.fhb_objects_normal

            image_names = []
            joints2d = []
            joints3d = []
            hand_sides = []
            clips = []
            sample_infos = []
            objnames = []
            objtransforms = []
            for subject, action_name, seq_idx, frame_idx in sample_list:
                if subject not in self.subjects:
                    continue

                # * Skip samples without objects
                if subject not in obj_infos or (action_name, seq_idx, frame_idx) not in obj_infos[subject]:
                    continue

                img_path = os.path.join(
                    self.rgb_root, subject, action_name, seq_idx, "color", self.rgb_template.format(frame_idx),
                )
                skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
                skel = skel[self.reorder_idx]
                skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                skel_camcoords = self.cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

                obj, transf = obj_infos[subject][(action_name, seq_idx, frame_idx)]
                if obj not in self.split_objects:
                    continue

                if self.filter_no_contact:
                    verts = self.split_objects[obj]["verts"]
                    transf_verts = fhbutils.transform_obj_verts(verts, transf, self.cam_extr)
                    all_dists = cdist(transf_verts, skel_camcoords)
                    if all_dists.min() > self.filter_thresh:
                        continue

                # collect the results
                clips.append((subject, action_name, seq_idx))
                objtransforms.append(transf)
                objnames.append(obj)

                image_names.append(img_path)
                sample_infos.append(
                    {"subject": subject, "action_name": action_name, "seq_idx": seq_idx, "frame_idx": frame_idx,}
                )
                joints3d.append(skel_camcoords)
                hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
                joints2d.append(skel2d.astype(np.float32))
                hand_sides.append("right")

            # assemble annotation
            mano_objs, mano_infos = fhbutils.load_manofits(sample_infos)
            annotations = {
                "image_names": image_names,
                "joints2d": joints2d,
                "joints3d": joints3d,
                "hand_sides": hand_sides,
                "sample_infos": sample_infos,
                "mano_infos": mano_infos,
                "mano_objs": mano_objs,
                "objnames": objnames,
                "objtransforms": objtransforms,
                "split_objects": self.split_objects,
                "split_objects_normal": self.split_objects_normal,
            }

            if self.rand_size != 0:  # Become effective ONLY in fhb's subclass: eg. fhbsynt
                logger.info("FPHB subclass is duplicating annot")
                annotations = fhbutils.update_synt_anno(annotations, self.rand_size, self.super_name)

            # using mini_factor to expose only a small ratio of data
            # no effect if mini_factor=1.0
            if self.mini_factor and self.mini_factor != 1.0:
                idxs = list(range(len(image_names)))
                mini_nb = int(len(image_names) * self.mini_factor)
                random.Random(1).shuffle(idxs)
                idxs = idxs[:mini_nb]
                for key, vals in annotations.items():
                    if key == "split_objects" or key == "split_objects_normal":
                        continue
                    annotations[key] = [vals[idx] for idx in idxs]

            # dump cache
            with open(self.cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            logger.info("Wrote cache for dataset {} to {}".format(self.name, self.cache_path), "yellow")

        # register loaded information into object
        self.image_names = annotations["image_names"]
        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"]
        self.hand_sides = annotations["hand_sides"]
        self.sample_infos = annotations["sample_infos"]
        self.mano_objs = annotations["mano_objs"]
        self.mano_infos = annotations["mano_infos"]
        self.objnames = annotations["objnames"]
        self.objtransforms = annotations["objtransforms"]
        self.split_objects = annotations["split_objects"]
        self.split_objects_normal = annotations["split_objects_normal"]
        if self.rand_size != 0:
            self.rand_transf = annotations["rand_transf"]

        # ====== things will always do, regardless of cache
        if self.load_objects_reduced:
            self.fhb_objects_reduced = fhbutils.load_objects_reduced(
                obj_root=os.path.join(self.root_supp, "Object_models"), object_names=all_objects,
            )

        if self.load_objects_voxel:
            self.fhb_objects_voxel = fhbutils.load_objects_voxel(
                obj_root=os.path.join(self.root_supp, "Object_models_binvox"), object_names=all_objects,
            )

        # extra info: hand vertex & anchor stuff
        # this doesn't need to be cached, as it keeps the sampe for all samples
        self.hand_palm_vertex_index = np.loadtxt(os.path.join(self.root_extra_info, "hand_palm_full.txt"), dtype=np.int)
        self.n_palm_vert = self.get_n_hand_palm_vert(0)
        if self.enable_contact:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_extra_info)

        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.image_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)]
        self.inp_res = self.image_size
        logger.info(f"Got {len(self.image_names)} samples for data_split {self.data_split}")
        return

    def __len__(self):
        return len(self.image_names)

    def get_image(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        return self.image_names[idx]

    def get_hand_faces(self, idx):
        faces = np.array(self.layer.th_faces).astype(np.long)
        return faces

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

    def _fhb_get_hand_info(self, idx):
        """
        Get the hand annotation in the raw fhb datasets.
        !!! This Mehthods shoudln't be called outside.
        :param idx:
        :return:
        """
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"], mano_info["trans"], mano_info["shape"]

    def get_obj_textures(self, idx):
        obj = self.objnames[idx]
        objtextures = self.split_objects[obj]["textures"]
        return np.array(objtextures)

    def get_obj_faces(self, idx):
        obj = self.objnames[idx]
        objfaces = self.split_objects[obj]["faces"]
        return np.array(objfaces).astype(np.int32)

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

    def get_obj_pose(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_transf(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_rot(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, :3]

    def get_obj_tsl(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, 3]

    def get_obj_verts_transf(self, idx):
        obj = self.objnames[idx]
        transf = self.objtransforms[idx]
        verts_raw = self.split_objects[obj]["verts"]
        transf_verts = fhbutils.transform_obj_verts(verts_raw, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)

    def get_obj_normal(self, idx):
        obj = self.objnames[idx]
        normal = self.split_objects_normal[obj]
        return np.array(normal).astype(np.float32)

    def get_joint_vis(self, idx):
        return np.ones(self.njoints)

    def get_obj_verts_can(self, idx):
        obj = self.objnames[idx]
        verts = self.split_objects[obj]["verts"]
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)
        return verts_can, bbox_center, bbox_scale

    # ? only used for render
    def get_obj_full_color_can(self, idx):
        obj = self.objnames[idx]
        verts = self.obj_meshes_full_color[obj]["verts"]

        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can, self.obj_meshes_full_color[obj]["faces"], self.obj_meshes_full_color[obj]["vc"]

    def get_obj_verts2d(self, idx):
        objpoints3d = self.get_obj_verts_transf(idx)
        objpoints3d = objpoints3d * 1000
        hom_2d = np.array(self.cam_intr).dot(objpoints3d.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return verts2d.astype(np.float32)

    def get_joints3d(self, idx):
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints2d(self, idx):
        joints = self.joints2d[idx] * self.reduce_factor
        return joints

    def get_cam_intr(self, idx):
        camintr = self.cam_intr
        return camintr.astype(np.float32)

    def get_sides(self, idx):
        side = self.hand_sides[idx]
        return side

    def get_meta(self, idx):
        meta = {"objname": self.objnames[idx]}
        return meta

    def get_center_scale(self, idx):
        if self.full_image:
            center = np.array((480 / 2, 270 / 2))
            scale = 480
        else:
            logger.error("Non full_image mode is not implements")
            raise NotImplementedError()
            # joints2d = self.get_joints2d(idx)[0]
            # center = handutils.get_annot_center(joints2d)
            # scale = handutils.get_annot_scale(joints2d)
        return center, scale

    def get_obj_vis2d(self, idx):
        objvis = np.ones_like(self.get_obj_verts2d(idx))
        return objvis

    def get_hand_vis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts2d(idx))
        return handvis

    def get_hand_palm_vert_idx(self, _):
        return self.hand_palm_vertex_index

    def get_n_hand_palm_vert(self, _):
        return len(self.hand_palm_vertex_index)

    def get_obj_verts_can_raw(self, idx):
        obj = self.objnames[idx]
        verts = self.split_objects[obj]["verts"]
        return verts

    def get_sample_identifier(self, idx):
        identifier = (
            f"{self.data_split}_{self.split_mode}_mf{self.mini_factor}"
            f"_rf{self.reduce_factor}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
        )

        res = f"{self.name}/{identifier}/{idx}"
        return res

    # ? only used in offline eval
    def get_hand_tsl_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["trans"].astype(np.float32)

    # ? only used in offline eval
    def get_hand_shape(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["shape"].astype(np.float32)

    # ? only used in offline eval
    def get_hand_pose_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"].astype(np.float32)

    # ? only used in offline eval
    def get_hand_axisang_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"][0:3].astype(np.float32)

    # ? only used in offline eval
    def get_hand_rot_wrt_cam(self, idx):
        axisang = self.get_hand_axisang_wrt_cam(idx)
        rot = SO3.exp(axisang).as_matrix()
        return rot.astype(np.float32)

    # ? only used in offline eval
    def get_obj_verts_transf_reduced(self, idx):
        obj = self.objnames[idx]
        transf = self.objtransforms[idx]
        verts_raw = self.fhb_objects_reduced[obj]["verts"]
        transf_verts = fhbutils.transform_obj_verts(verts_raw, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)

    # ? only used in offline eval
    def get_obj_verts_can_reduced(self, idx):
        obj = self.objnames[idx]
        verts = self.fhb_objects_reduced[obj]["verts"]
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)
        return verts_can, bbox_center, bbox_scale

    # ? only used in offline eval
    def get_obj_faces_reduced(self, idx):
        obj = self.objnames[idx]
        objfaces = self.fhb_objects_reduced[obj]["faces"]
        return np.array(objfaces).astype(np.int32)

    # ? only used in post process
    def get_obj_voxel_points_can(self, idx):
        obj = self.objnames[idx]
        objvoxpts = self.fhb_objects_voxel[obj]["points"]
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(objvoxpts, scale=False)
        return verts_can

    # ? only used in post process
    def get_obj_voxel_points_transf(self, idx):
        obj = self.objnames[idx]
        transf = self.objtransforms[idx]
        objvoxpts = self.fhb_objects_voxel[obj]["points"]
        transf_verts = fhbutils.transform_obj_verts(objvoxpts, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)

    # ? only used in post process
    def get_obj_voxel_element_volume(self, idx):
        obj = self.objnames[idx]
        objvoxelvol = self.fhb_objects_voxel[obj]["element_volume"]
        return objvoxelvol


def view_image_data(ho_dataset):
    for i in range(len(ho_dataset)):
        img_path = ho_dataset.get_image_path(i)
        if "pour_milk" not in img_path:
            continue
        print(i, "   ", img_path)
        joint2d = ho_dataset.get_joints2d(i)

        # TEST: obj_transf @ obj_verts_can == obj_verts_transf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        obj_transf = ho_dataset.get_obj_transf_wrt_cam(i)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)

        obj_verts_can, _, __ = ho_dataset.get_obj_verts_can(i)  # (N, 3)
        obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()
        obj_verts2d_pred = ho_dataset.project(obj_verts_pred, ho_dataset.cam_intr)
        obj_verts2d_gt = ho_dataset.get_obj_verts2d(i)

        # TEST: MANO(get_hand_pose_wrt_cam) + get_hand_tsl_wrt_cam == get_hand_verts3d >>>>>>>>>>>>>>>>>>>>>>>>>>
        hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
        hand_shape = torch.from_numpy(ho_dataset.get_hand_shape(i)).unsqueeze(0)
        hand_tsl = ho_dataset.get_hand_tsl_wrt_cam(i)

        hand_verts, hand_joints = ho_dataset.layer(hand_pose, hand_shape)
        hand_verts = np.array(hand_verts.squeeze(0)) + hand_tsl
        hand_verts_gt = ho_dataset.get_hand_verts3d(i)

        hand_verts_2d = ho_dataset.project(hand_verts, ho_dataset.cam_intr)
        hand_verts_2dgt = ho_dataset.get_hand_verts2d(i)

        img = ho_dataset.get_image(i)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j in range(obj_verts2d_pred.shape[0]):
            if j % 2 == 0:
                continue
            v = obj_verts2d_pred[j]
            cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(0, 255, 0))

        for j in range(hand_verts_2d.shape[0]):
            if j % 2 == 0:
                continue
            v = hand_verts_2d[j]
            cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(255, 0, 0))

        cv2.imshow("fhbhands", img)
        cv2.waitKey(1)


def main(args):
    ho_dataset = HOdata.get_dataset(
        dataset="fhb",
        data_root="data",
        data_split=args.data_split,
        split_mode=args.split_mode,
        use_cache=False,
        mini_factor=1,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=True,
        filter_thresh=5.0,
        block_rot=True,
        synt_factor=1,
    )

    import prettytable as pt
    from hocontact.utils.logger import logger

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
    logger.warn(f"{'='*40} ALL FHB SAMPLE KEYS {'>'*40}", "blue")
    logger.info(str(tb))

    if args.vis:
        view_image_data(ho_dataset)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test fhbdataset")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--data_split", choices=["train", "test", "all"], default="train", type=str)
    parser.add_argument("--split_mode", choices=["actions", "subjects"], default="actions", type=str)
    main(parser.parse_args())
