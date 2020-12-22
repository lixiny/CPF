import os
import pickle

from hocontact.hodatasets.cidata import CIdata
from hocontact.hodatasets.ciquery import CIAdaptQueries, CIDumpedQueries
from hocontact.utils.contactutils import dumped_process_contact_info


class CIOnline(CIdata):
    def __init__(
        self, data_path, hodata_path, anchor_path, hodata_use_cache=True, hodata_center_idx=9,
    ):
        super().__init__(
            data_path, hodata_path, anchor_path, hodata_use_cache=hodata_use_cache, hodata_center_idx=hodata_center_idx
        )

        # along side with basic CIDumpedQueries, we need some adapt queries
        # for offline eval
        self.queries.update(
            {
                CIAdaptQueries.OBJ_VERTS_3D,
                CIAdaptQueries.OBJ_CAN_VERTS,
                CIAdaptQueries.OBJ_FACES,
                CIAdaptQueries.OBJ_TSL,
                CIAdaptQueries.OBJ_ROT,
                CIAdaptQueries.OBJ_NORMAL,
                CIAdaptQueries.HAND_VERTS_3D,
                CIAdaptQueries.HAND_JOINTS_3D,
                CIAdaptQueries.HAND_FACES,
                CIAdaptQueries.HAND_SHAPE,
                CIAdaptQueries.HAND_POSE,
                CIAdaptQueries.HAND_TSL,
                CIAdaptQueries.HAND_ROT,
                CIAdaptQueries.OBJ_VOXEL_POINTS_CAN,
                CIAdaptQueries.OBJ_VOXEL_EL_VOL,
                CIAdaptQueries.IMAGE_PATH,
                CIDumpedQueries.OBJ_VERTS_3D,
                CIDumpedQueries.OBJ_TSL,
                CIDumpedQueries.OBJ_ROT,
                CIDumpedQueries.HAND_VERTS_3D,
                CIDumpedQueries.HAND_JOINTS_3D,
                CIDumpedQueries.HAND_SHAPE,
                CIDumpedQueries.HAND_TSL,
                CIDumpedQueries.HAND_ROT,
                CIDumpedQueries.HAND_POSE,
            }
        )

    def get_dumped_processed_contact_info(self, index):
        dumped_file_path = os.path.join(self.data_path, f"{index}_contact.pkl")
        with open(dumped_file_path, "rb") as bytestream:
            dumped_contact_info_list = pickle.load(bytestream)
        (vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask) = dumped_process_contact_info(
            dumped_contact_info_list,
            self.anchor_mapping,
            pad_vertex=self.contact_pad_vertex,
            pad_anchor=self.contact_pad_anchor,
            elasti_th=self.contact_elasti_th,
        )
        res = {
            "vertex_contact": vertex_contact,
            "hand_region": hand_region,
            "anchor_id": anchor_id,
            "anchor_elasti": anchor_elasti,
            "anchor_padding_mask": anchor_padding_mask,
        }
        return res

    def get_dumped_processed_pose(self, index):
        dumped_file_path = os.path.join(self.data_path, f"{index}_honet.pkl")
        with open(dumped_file_path, "rb") as bytestream:
            dumped_pose_dict = pickle.load(bytestream)
        return dumped_pose_dict
