from enum import Enum, auto


class BaseQueries(Enum):
    CAM_INTR = auto()  # camera intrinsic for raw image
    OBJ_FACES = auto()
    OBJ_CORNERS_2D = auto()
    OBJ_CORNERS_3D = auto()
    OBJ_VERTS_3D = auto()  # object verts after extr transform (raw)
    OBJ_VERTS_2D = auto()  # object verts on raw image (raw)
    OBJ_VIS_2D = auto()
    HAND_FACES = auto()  # NEW: hand faces
    HAND_VERTS_3D = auto()  # hand verts after extr transform (raw)
    HAND_VERTS_2D = auto()  # hand verts on raw image (raw)
    HAND_VIS_2D = auto()
    CENTER_3D = auto()  # hand center
    JOINTS_3D = auto()
    JOINTS_2D = auto()
    IMAGE = auto()  # raw image
    SIDE = auto()  # flipped side
    OBJ_CAN_VERTS = auto()  # object verts under canonical system (the object is centered)
    OBJ_CAN_SCALE = auto()
    OBJ_CAN_TRANS = auto()
    OBJ_CAN_CORNERS = auto()
    IMAGE_PATH = auto()
    JOINT_VIS = auto()
    OBJ_TRANSF = auto()  # NEW: object transform, camera system
    HAND_POSE_WRT_CAM = auto()
    HAND_QUAT = auto()  # NEW: 16x4 quaternions to recov mano full mesh
    HAND_BETA = auto()  # NEW: 10 shape coefficient to recov mano full mesh


# NOTE: This queries are meant to supervise the PICR netwrok during training.
# NOTE: We currently remove all the training codes from the released CPF.
# NOTE: Thus ContactQueries and all the methods that are related to it, are not avaliable.
class ContactQueries(Enum):
    CONTACT_INFO = auto()  # NEW: contact info
    HAND_PALM_VERT_IDX = auto()  # NEW: static index mask for selecting palm index
    VERTEX_CONTACT = auto()  # NEW: for each vertex, whether it is in contact with hand
    ANCHOR_MAPPING = auto()  # NEW: return anchor_mapping [[ NOT IN DEFAULT QUERY ]]
    REV_ANCHOR_MAPPING = auto()  # NEW: return rev_anchor_mapping [[ NOT IN DEFAULT QUERY ]]
    CONTACT_REGION_ID = auto()  # NEW: returns region id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ID = auto()  # NEW: returns anchor id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_DIST = auto()  # NEW: returns anchor dist [[ NOTE ITS INTERACTION 2WITH PADDING ]]
    CONTACT_ANCHOR_ELASTI = auto()  # NEW: returns anchor elasti [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_PADDING_MASK = auto()  # NEW: if padding enabled, this field will be append to the query


class TransQueries(Enum):
    CAM_INTR = auto()  # camera intrinsic for augmented image
    OBJ_TRANSF = auto()  # NEW: object transform, camera system
    OBJ_VERTS_3D = auto()  # object verts after extr transform, under hand centered system
    OBJ_VERTS_2D = auto()  # object verts on augmented image (augmented)
    OBJ_CORNERS_2D = auto()
    OBJ_CORNERS_3D = auto()
    HAND_VERTS_3D = auto()  # hand verts after extr transform, under hand centered system
    HAND_VERTS_2D = auto()  # hand verts on augmented image (augmented)
    JOINTS_3D = auto()
    JOINTS_2D = auto()
    CENTER_3D = auto()  # hand center (augmented)
    IMAGE = auto()  # augmented image
    SIDE = auto()  # unused
    SCALE = auto()  # unused
    AFFINETRANS = auto()  # image affine trans in augmentation process
    ROTMAT = auto()  # unused


class MetaQueries(Enum):
    SAMPLE_IDENTIFIER = auto()


class CollateQueries(Enum):
    # dataset fields generated in collate fn
    # these data queries will only appear in dataloaders
    # if use plain datasets, there will never be such query
    PADDING_MASK = auto()
    FACE_PADDING_MASK = auto()


def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False


def get_trans_queries(base_queries):
    trans_queries = set()
    if BaseQueries.OBJ_VERTS_3D in base_queries:
        trans_queries.add(TransQueries.OBJ_VERTS_3D)
    if BaseQueries.IMAGE in base_queries:
        trans_queries.add(TransQueries.IMAGE)
        trans_queries.add(TransQueries.AFFINETRANS)
        trans_queries.add(TransQueries.ROTMAT)
    if BaseQueries.JOINTS_2D in base_queries:
        trans_queries.add(TransQueries.JOINTS_2D)
    if BaseQueries.JOINTS_3D in base_queries:
        trans_queries.add(TransQueries.JOINTS_3D)
    if BaseQueries.HAND_VERTS_3D in base_queries:
        trans_queries.add(TransQueries.HAND_VERTS_3D)
        trans_queries.add(TransQueries.CENTER_3D)
    if BaseQueries.HAND_VERTS_2D in base_queries:
        trans_queries.add(TransQueries.HAND_VERTS_2D)
    if BaseQueries.OBJ_VERTS_3D in base_queries:
        trans_queries.add(TransQueries.OBJ_VERTS_3D)
    if BaseQueries.OBJ_VERTS_2D in base_queries:
        trans_queries.add(TransQueries.OBJ_VERTS_2D)
    if BaseQueries.OBJ_CORNERS_3D in base_queries:
        trans_queries.add(TransQueries.OBJ_CORNERS_3D)
    if BaseQueries.OBJ_CORNERS_2D in base_queries:
        trans_queries.add(TransQueries.OBJ_CORNERS_2D)
    if BaseQueries.CAM_INTR in base_queries:
        trans_queries.add(TransQueries.CAM_INTR)
    if BaseQueries.OBJ_TRANSF in base_queries:
        trans_queries.add(TransQueries.OBJ_TRANSF)
    if BaseQueries.OBJ_CAN_VERTS in base_queries or BaseQueries.OBJ_CAN_CORNERS:
        trans_queries.add(BaseQueries.OBJ_CAN_SCALE)
        trans_queries.add(BaseQueries.OBJ_CAN_TRANS)
    return trans_queries


def match_collate_queries(query_spin):
    object_vertex_queries = [
        TransQueries.OBJ_VERTS_3D,
        BaseQueries.OBJ_VERTS_3D,
        BaseQueries.OBJ_CAN_VERTS,
        BaseQueries.OBJ_VERTS_2D,
        BaseQueries.OBJ_VIS_2D,
        TransQueries.OBJ_VERTS_2D,
    ]
    object_face_quries = [
        BaseQueries.OBJ_FACES,
    ]

    if query_spin in object_vertex_queries:
        return CollateQueries.PADDING_MASK
    elif query_spin in object_face_quries:
        return CollateQueries.FACE_PADDING_MASK
