import trimesh
import pyrender
import torch
from manopth.manolayer import ManoLayer
import numpy as np
from matplotlib import cm
from manopth.anchorutils import anchor_load_driver, recover_anchor


def get_colors(num_color, alpha=1.0, shuffle=False):
    cmap = cm.get_cmap("rainbow")
    x = np.linspace(0.0, 1.0, num_color)
    res = cmap(x)
    res[:, 3] = alpha
    if shuffle:
        np.random.shuffle(res)
    return res


def main(render=False, face_vertex_index=None, anchor_weight=None):
    # init mano
    mano_layer = ManoLayer(
        mano_root="assets/mano",
        use_pca=False,
        ncomps=12,
        flat_hand_mean=True,
        center_idx=9,
        return_transf=True,
    )
    faces = np.array(mano_layer.th_faces).astype(np.long)

    # gen random shape
    vec_shape = torch.rand(1, 10)
    # gen zero pose
    vec_pose = torch.zeros(1, 48)

    # gen hand
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    joints = np.array(joints[0])
    vertices = np.array(vertices[0])
    transf = np.array(transf[0])
    n_vertex = vertices.shape[0]

    # ========= MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    anchors = recover_anchor(vertices, face_vertex_index, anchor_weight)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # set up viewer
    if render:
        scene = pyrender.Scene()
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(node_cam)
        scene.set_pose(node_cam, pose=np.eye(4))
        vertex_colors = np.array([0.8, 0.8, 0.8, 0.8])
        vertex_colors = np.expand_dims(vertex_colors, 0).repeat(n_vertex, axis=0)
        joint_colors = np.array([10, 73, 233, 255])
        transl = np.array([-0, -0, -500.0])
        transl = transl[np.newaxis, :]

        joints = joints * 1000.0 + transl
        vertices = vertices * 1000.0 + transl
        transf[:, :3, 3] = transf[:, :3, 3] * 1000.0 + transl

        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene.add(mesh)

        anchor_colors = get_colors(anchors.shape[0], alpha=1.0)
        for k in range(len(anchors)):
            anchor_sphere = trimesh.creation.uv_sphere(radius=2)
            anchor_sphere.visual.vertex_colors = anchor_colors[k]
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = anchors[k] * 1000 + transl
            anchor_mesh = pyrender.Mesh.from_trimesh(anchor_sphere, poses=tfs)
            scene.add(anchor_mesh)

        # pyrender.Viewer(scene, use_raymond_lighting=True, render_flags={"all_wireframe": True})
        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, default="./assets")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    fvi, aw, _, _ = anchor_load_driver(args.inpath)
    main(render=args.render, face_vertex_index=fvi, anchor_weight=aw)
