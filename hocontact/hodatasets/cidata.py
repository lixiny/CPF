import os
import re
from abc import ABC, abstractmethod
from manopth.anchorutils import anchor_load
from hocontact.hodatasets.ciquery import CIAdaptQueries, CIDumpedQueries
from hocontact.hodatasets.hodata import HOdata
from hocontact.utils.logger import logger

fhb_tester = re.compile(r"\/fhbhands\/")
fhb_extractor = re.compile(
    r"^(.*)\/fhbhands\/(.*)_(.*)_mf([0-9]*\.?[0-9]*)_rf([0-9]*\.?[0-9]*)_fct([0-9]*\.?[0-9]*)_ec(.*)$"
)
ho3d_tester = re.compile(r"\/HO3D\/")
ho3d_extractor = re.compile(r"^(.*)\/HO3D\/(.*)_(.*)_mf([0-9]*\.?[0-9]*)_likev1(.*)_fct([0-9]*\.?[0-9]*|\(x\))_ec(.*)")
id_extractor = re.compile(r"^([0-9]+).*$")


class CIdata(ABC):
    # ! warning: this class and all its derived class
    # ! should ***NEVER*** be runned in batch mode
    # // (like torch.utils.data.DataLoader)

    @staticmethod
    def match_and_construct(in_string, hodata_path, use_cache, center_idx):
        fhb_flag = fhb_tester.search(in_string)
        if fhb_flag is not None:
            fhb_matches = fhb_extractor.fullmatch(in_string)
            if fhb_matches is None:
                raise NameError(f"{CIdata.__name__}: parse string error, trying to parse as fhb, in string{in_string}")

            ci_prefix = fhb_matches.group(1)
            data_split = fhb_matches.group(2)
            split_mode = fhb_matches.group(3)
            if fhb_matches.group(4) == "1":
                mini_factor = 1
            else:
                mini_factor = float(fhb_matches.group(4))
            reduce_factor = float(fhb_matches.group(5))
            if abs(reduce_factor - 1) < 1e-9:
                reduce_res = False
            else:
                reduce_res = True

            filter_contact_flag = fhb_matches.group(6)
            if filter_contact_flag == "(x)":
                filter_no_contact = False
                filter_thresh = -1.0
            else:
                filter_no_contact = True
                filter_thresh = float(filter_contact_flag)

            enable_contact_flag = fhb_matches.group(7)
            if enable_contact_flag == "":
                enable_contact = True
            else:
                enable_contact = False

            logger.info(
                (
                    f"\tCI prefix: {ci_prefix}\n"
                    f"\tHO prefix: {hodata_path}\n"
                    f"\tdata_split: {data_split}, split_mode: {split_mode}\n"
                    f"\tmini_factor: {mini_factor}\n"
                    f"\treduce_res: {reduce_res}, reduce_factor: {reduce_factor}\n"
                    f"\tfilter_no_contact: {filter_no_contact}, filter_thresh: {filter_thresh}\n"
                    f"\tenable_contact: {enable_contact}\n"
                ),
                "cyan",
            )

            target_dataset = HOdata.get_dataset(
                "fhb",
                data_root=hodata_path,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                reduce_res=reduce_res,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                load_objects_reduced=True,
                load_objects_voxel=True,
                synt_factor=0,
            )
            return target_dataset

        ho3d_flag = ho3d_tester.search(in_string)
        if ho3d_flag is not None:
            ho3d_matches = ho3d_extractor.fullmatch(in_string)
            if ho3d_matches is None:
                raise NameError(f"{CIdata.__name__}: parse string error, trying to parse as ho3d, in string{in_string}")

            ci_prefix = ho3d_matches.group(1)
            data_split = ho3d_matches.group(2)
            split_mode = ho3d_matches.group(3)
            if ho3d_matches.group(4) == "1":
                mini_factor = 1
            else:
                mini_factor = float(ho3d_matches.group(4))

            like_v1 = ho3d_matches.group(5) == ""

            filter_contact_flag = ho3d_matches.group(6)
            if filter_contact_flag == "(x)":
                filter_no_contact = False
                filter_thresh = -1.0
            else:
                filter_no_contact = True
                filter_thresh = float(filter_contact_flag)

            enable_contact = ho3d_matches.group(7) == ""

            logger.info(
                (
                    f"\tCI prefix: {ci_prefix}\n"
                    f"\tHO prefix: {hodata_path}\n"
                    f"\tdata_split: {data_split}, split_mode: {split_mode}\n"
                    f"\tmini_factor: {mini_factor}\n"
                    f"\tlike_v1: {like_v1}\n"
                    f"\tfilter_no_contact: {filter_no_contact}, filter_thresh: {filter_thresh}\n"
                    f"\tenable_contact: {enable_contact}\n"
                ),
                "cyan",
            )

            target_dataset = HOdata.get_dataset(
                "ho3d",
                data_root=hodata_path,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                like_v1=like_v1,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                synt_factor=0,  # do not need synthetic data when testing
                load_objects_reduced=True,
                load_objects_voxel=True,
            )
            return target_dataset

        raise NameError(f"{CIdata.__name__}: parse string error, in string {in_string}")

    def __init__(
        self, data_path, hodata_path, anchor_path, hodata_use_cache=True, hodata_center_idx=9,
    ):
        super().__init__()
        logger.info(f"{self.__class__.__name__}:", "cyan")
        self.data_path = data_path
        self.hodata_path = hodata_path
        self.anchor_path = anchor_path
        (
            self.anchor_face_vertex_index,
            self.anchor_weights,
            self.hand_vertex_merged_assignment,
            self.anchor_mapping,
        ) = anchor_load(self.anchor_path)
        # these fields are originally designed as flags
        # though now they are constant within the project
        self.contact_pad_vertex = True
        self.contact_pad_anchor = True
        self.contact_elasti_th = 0.00

        # ==================== match and extract information from data_path >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # ==================== construct dataset accordingly >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.hodataset = self.match_and_construct(self.data_path, self.hodata_path, hodata_use_cache, hodata_center_idx)
        self.hodataset_type = self.hodataset.name
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ==================== cache pkl paths >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self._file_list = list(os.listdir(self.data_path))
        self.file_list = set()
        for x in self._file_list:
            res = id_extractor.match(x)
            if res is None:
                raise RuntimeError(f"{res} matches failed! illegal name for intermediate file")
            self.file_list.add(int(res.group(1)))
        assert len(self.file_list) == len(
            self.hodataset
        ), f"dataset length unmatched: {len(self.file_list)}, {len(self.hodataset)}"
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ==================== provide queries >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # for interface (with data so not pure), we only provide CIDumpedQueries
        self.queries = {
            CIDumpedQueries.VERTEX_CONTACT,
            CIDumpedQueries.CONTACT_REGION_ID,
            CIDumpedQueries.CONTACT_ANCHOR_ID,
            CIDumpedQueries.CONTACT_ANCHOR_ELASTI,
            CIDumpedQueries.CONTACT_ANCHOR_PADDING_MASK,
        }
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def __len__(self):
        return len(self.file_list)

    # ? must be implemented
    @abstractmethod
    def get_dumped_processed_contact_info(self, index):
        pass

    # ? raise exception when not implemented
    def get_dumped_processed_pose(self, index):
        raise RuntimeError("unimplemented")

    def __getitem__(self, index):
        queries = self.queries
        sample = {}

        # ==================== process adapted queries >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if CIAdaptQueries.HAND_VERTS_3D in queries:
            sample[CIAdaptQueries.HAND_VERTS_3D] = self.hodataset.get_hand_verts3d(index)

        if CIAdaptQueries.HAND_JOINTS_3D in queries:
            sample[CIAdaptQueries.HAND_JOINTS_3D] = self.hodataset.get_joints3d(index)

        if CIAdaptQueries.HAND_FACES in queries:
            sample[CIAdaptQueries.HAND_FACES] = self.hodataset.get_hand_faces(index)

        if CIAdaptQueries.HAND_ROT in queries:
            sample[CIAdaptQueries.HAND_ROT] = self.hodataset.get_hand_rot_wrt_cam(index)

        if CIAdaptQueries.HAND_TSL in queries:
            sample[CIAdaptQueries.HAND_TSL] = self.hodataset.get_hand_tsl_wrt_cam(index)

        if CIAdaptQueries.HAND_SHAPE in queries:
            sample[CIAdaptQueries.HAND_SHAPE] = self.hodataset.get_hand_shape(index)

        if CIAdaptQueries.HAND_POSE in queries:
            sample[CIAdaptQueries.HAND_POSE] = self.hodataset.get_hand_pose_wrt_cam(index)

        if CIAdaptQueries.OBJ_VERTS_3D in queries:
            sample[CIAdaptQueries.OBJ_VERTS_3D] = self.hodataset.get_obj_verts_transf(index)

        if CIAdaptQueries.OBJ_VERTS_3D_REDUCED in queries:
            sample[CIAdaptQueries.OBJ_VERTS_3D_REDUCED] = self.hodataset.get_obj_verts_transf_reduced(index)

        if CIAdaptQueries.OBJ_CAN_VERTS in queries:
            sample[CIAdaptQueries.OBJ_CAN_VERTS], _, _ = self.hodataset.get_obj_verts_can(index)

        if CIAdaptQueries.OBJ_CAN_VERTS_REDUCED in queries:
            sample[CIAdaptQueries.OBJ_CAN_VERTS_REDUCED], _, _ = self.hodataset.get_obj_verts_can_reduced(index)

        if CIAdaptQueries.OBJ_FACES in queries:
            sample[CIAdaptQueries.OBJ_FACES] = self.hodataset.get_obj_faces(index)

        if CIAdaptQueries.OBJ_FACES_REDUCED in queries:
            sample[CIAdaptQueries.OBJ_FACES_REDUCED] = self.hodataset.get_obj_faces_reduced(index)

        if CIAdaptQueries.OBJ_NORMAL in queries:
            sample[CIAdaptQueries.OBJ_NORMAL] = self.hodataset.get_obj_normal(index)

        if CIAdaptQueries.OBJ_TRANSF in queries:
            sample[CIAdaptQueries.OBJ_TRANSF] = self.hodataset.get_obj_transf_wrt_cam(index)

        if CIAdaptQueries.OBJ_TSL in queries:
            sample[CIAdaptQueries.OBJ_TSL] = self.hodataset.get_obj_tsl(index)

        if CIAdaptQueries.OBJ_ROT in queries:
            sample[CIAdaptQueries.OBJ_ROT] = self.hodataset.get_obj_rot(index)

        if CIAdaptQueries.IMAGE_PATH in queries:
            sample[CIAdaptQueries.IMAGE_PATH] = self.hodataset.get_image_path(index)

        if CIAdaptQueries.OBJ_VOXEL_POINTS_CAN in queries:
            sample[CIAdaptQueries.OBJ_VOXEL_POINTS_CAN] = self.hodataset.get_obj_voxel_points_can(index)

        if CIAdaptQueries.OBJ_VOXEL_POINTS in queries:
            sample[CIAdaptQueries.OBJ_VOXEL_POINTS] = self.hodataset.get_obj_voxel_points_transf(index)

        if CIAdaptQueries.OBJ_VOXEL_EL_VOL in queries:
            sample[CIAdaptQueries.OBJ_VOXEL_EL_VOL] = self.hodataset.get_obj_voxel_element_volume(index)

        if CIAdaptQueries.HAND_PALM_VERT_IDX in queries:
            sample[CIAdaptQueries.HAND_PALM_VERT_IDX] = self.hodataset.get_hand_palm_vert_idx(index)

        if (
            CIAdaptQueries.VERTEX_CONTACT in queries
            or CIAdaptQueries.CONTACT_REGION_ID in queries
            or CIAdaptQueries.CONTACT_ANCHOR_ID in queries
            or CIAdaptQueries.CONTACT_ANCHOR_ELASTI in queries
        ):
            processed_dict = self.hodataset.get_processed_contact_info(index)
            if CIAdaptQueries.VERTEX_CONTACT in queries:
                sample[CIAdaptQueries.VERTEX_CONTACT] = processed_dict["vertex_contact"]
            if CIAdaptQueries.CONTACT_REGION_ID in queries:
                sample[CIAdaptQueries.CONTACT_REGION_ID] = processed_dict["hand_region"]
            if CIAdaptQueries.CONTACT_ANCHOR_ID in queries:
                sample[CIAdaptQueries.CONTACT_ANCHOR_ID] = processed_dict["anchor_id"]
            if CIAdaptQueries.CONTACT_ANCHOR_ELASTI in queries:
                sample[CIAdaptQueries.CONTACT_ANCHOR_ELASTI] = processed_dict["anchor_elasti"]
            contact_anchor_padding_mask = processed_dict["anchor_padding_mask"]
            if contact_anchor_padding_mask is not None:
                sample[CIAdaptQueries.CONTACT_ANCHOR_PADDING_MASK] = contact_anchor_padding_mask
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ==================== process dumped queries >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (
            (CIDumpedQueries.HAND_VERTS_3D in queries)
            or (CIDumpedQueries.HAND_JOINTS_3D in queries)
            or (CIDumpedQueries.HAND_TSL in queries)
            or (CIDumpedQueries.HAND_ROT in queries)
            or (CIDumpedQueries.HAND_POSE in queries)
            or (CIDumpedQueries.HAND_SHAPE in queries)
            or (CIDumpedQueries.OBJ_VERTS_3D in queries)
            or (CIDumpedQueries.OBJ_TRANSF in queries)
            or (CIDumpedQueries.OBJ_TSL in queries)
            or (CIDumpedQueries.OBJ_ROT in queries)
        ):
            honet_dict = self.get_dumped_processed_pose(index)
            if CIDumpedQueries.HAND_VERTS_3D in queries:
                sample[CIDumpedQueries.HAND_VERTS_3D] = honet_dict["hand_verts_3d"]

            if CIDumpedQueries.HAND_JOINTS_3D in queries:
                sample[CIDumpedQueries.HAND_JOINTS_3D] = honet_dict["hand_joints_3d"]

            if CIDumpedQueries.HAND_TSL in queries:
                sample[CIDumpedQueries.HAND_TSL] = honet_dict["hand_tsl"].reshape((3,))

            if CIDumpedQueries.HAND_ROT in queries:
                sample[CIDumpedQueries.HAND_ROT] = honet_dict["hand_full_pose"][0:3]

            if CIDumpedQueries.HAND_POSE in queries:
                sample[CIDumpedQueries.HAND_POSE] = honet_dict["hand_full_pose"].reshape((16, 3))

            if CIDumpedQueries.HAND_SHAPE in queries:
                sample[CIDumpedQueries.HAND_SHAPE] = honet_dict["hand_shape"]
            if CIDumpedQueries.OBJ_VERTS_3D in queries:
                sample[CIDumpedQueries.OBJ_VERTS_3D] = honet_dict["obj_verts_3d"]

            if CIDumpedQueries.OBJ_TSL in queries:
                sample[CIDumpedQueries.OBJ_TSL] = honet_dict["obj_tsl"].reshape((3,))

            if CIDumpedQueries.OBJ_ROT in queries:
                sample[CIDumpedQueries.OBJ_ROT] = honet_dict["obj_rot"].reshape((3,))

        if (
            CIDumpedQueries.VERTEX_CONTACT in queries
            or CIDumpedQueries.CONTACT_REGION_ID in queries
            or CIDumpedQueries.CONTACT_ANCHOR_ID in queries
            or CIDumpedQueries.CONTACT_ANCHOR_ELASTI in queries
        ):
            processed_dict = self.get_dumped_processed_contact_info(index)
            if CIDumpedQueries.VERTEX_CONTACT in queries:
                sample[CIDumpedQueries.VERTEX_CONTACT] = processed_dict["vertex_contact"]
            if CIDumpedQueries.CONTACT_REGION_ID in queries:
                sample[CIDumpedQueries.CONTACT_REGION_ID] = processed_dict["hand_region"]
            if CIDumpedQueries.CONTACT_ANCHOR_ID in queries:
                sample[CIDumpedQueries.CONTACT_ANCHOR_ID] = processed_dict["anchor_id"]
            if CIDumpedQueries.CONTACT_ANCHOR_ELASTI in queries:
                sample[CIDumpedQueries.CONTACT_ANCHOR_ELASTI] = processed_dict["anchor_elasti"]
            contact_anchor_padding_mask = processed_dict["anchor_padding_mask"]
            if contact_anchor_padding_mask is not None:
                sample[CIDumpedQueries.CONTACT_ANCHOR_PADDING_MASK] = contact_anchor_padding_mask
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return sample
