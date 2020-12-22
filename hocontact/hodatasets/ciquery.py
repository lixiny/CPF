from enum import Enum, auto


class CIAdaptQueries(Enum):
    # GT Information directly adapt from hodataset
    HAND_VERTS_3D = auto()
    HAND_JOINTS_3D = auto()
    HAND_FACES = auto()
    HAND_TSL = auto()
    HAND_ROT = auto()
    HAND_SHAPE = auto()
    HAND_POSE = auto()
    OBJ_VERTS_3D = auto()
    OBJ_VERTS_3D_REDUCED = auto()
    OBJ_CAN_VERTS = auto()
    OBJ_CAN_VERTS_REDUCED = auto()
    OBJ_FACES = auto()
    OBJ_FACES_REDUCED = auto()
    OBJ_NORMAL = auto()
    OBJ_TRANSF = auto()
    OBJ_TSL = auto()
    OBJ_ROT = auto()
    IMAGE_PATH = auto()
    OBJ_VOXEL_POINTS_CAN = auto()
    OBJ_VOXEL_POINTS = auto()
    OBJ_VOXEL_EL_VOL = auto()
    HAND_PALM_VERT_IDX = auto()
    VERTEX_CONTACT = auto()  # NEW: for each vertex, whether it is in contact with hand
    CONTACT_REGION_ID = auto()  # NEW: returns region id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ID = auto()  # NEW: returns anchor id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ELASTI = auto()  # NEW: returns anchor elasti [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_PADDING_MASK = auto()  # NEW: if padding enabled, this field will be append to the query


class CIDumpedQueries(Enum):
    HAND_VERTS_3D = auto()
    HAND_JOINTS_3D = auto()
    HAND_TSL = auto()
    HAND_ROT = auto()
    HAND_POSE = auto()
    HAND_SHAPE = auto()
    OBJ_VERTS_3D = auto()
    OBJ_TRANSF = auto()
    OBJ_TSL = auto()
    OBJ_ROT = auto()
    VERTEX_CONTACT = auto()  # NEW: for each vertex, whether it is in contact with hand
    CONTACT_REGION_ID = auto()  # NEW: returns region id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ID = auto()  # NEW: returns anchor id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ELASTI = auto()  # NEW: returns anchor elasti [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_PADDING_MASK = auto()  # NEW: if padding enabled, this field will be append to the query
