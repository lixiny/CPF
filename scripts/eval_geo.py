import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import trimesh
import pickle
from liegroups import SO3

from hocontact.hodatasets.cionline import CIOnline
from hocontact.hodatasets.ciquery import CIAdaptQueries, CIDumpedQueries
from hocontact.postprocess.geo_optimizer import GeOptimizer
from hocontact.utils.collisionutils import (
    batch_index_select,
    batch_pairwise_dist,
    batch_mesh_contains_points,
    masked_mean_loss,
    penetration_loss_hand_in_obj,
    intersection_volume,
    solid_intersection_volume,
)
from hocontact.utils.disjointnessutils import region_disjointness_metric
from hocontact.utils.anatomyutils import AnatomyMetric
from termcolor import colored, cprint
import open3d as o3d
from manopth.anchorutils import masking_load_driver, get_region_palm_mask
from manopth.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis
from joblib import Parallel, delayed
import traceback
import time
import shutil


def collapse_res_list(res_list_list):
    res = []
    for item in res_list_list:
        res.extend(item[0])
    return res


def merge_res_list(res_list):
    if len(res_list) < 1:
        return dict()

    keys_list = list(res_list[0].keys())

    # create init dict
    res = {k: 0.0 for k in keys_list}
    count = 0

    # iterate
    for item_id, item in enumerate(res_list):
        for k in keys_list:
            if np.isnan(item[k]):
                cprint(f"encountered nan in {item_id} key {k}", "red")
                continue
            res[k] += item[k]
        count += 1

    # avg
    for k in keys_list:
        res[k] /= count

    return res


def summarize(res_dict):
    for k, v in res_dict.items():
        print("mean " + str(k), v)


def run_sample_by_idx(
    device,
    hoptim,
    mode,
    cidata,
    kmetric,
    index,
    hand_region_assignment,
    hand_palm_vertex_mask,
    save_path,
    contact_ratio_thresh=0.01,
    hand_closed_path="assets/closed_hand/hand_mesh_close.obj",
):
    save_file = os.path.join(save_path, f"{index}_save.pkl")
    if os.path.exists(save_file):
        with open(save_file, "rb") as fstream:
            res = pickle.load(fstream)
            print_msg = {
                "hand_dist_before": res["hand_dist_before"],
                "hand_dist_after": res["hand_dist_after"],
                "hand_joints_dist_before": res["hand_joints_dist_before"],
                "hand_joints_dist_after": res["hand_joints_dist_after"],
                "object_dist_before": res["object_dist_before"],
                "object_dist_after": res["object_dist_after"],
                "penetration_depth_gt": res["penetration_depth_gt"],
                "penetration_depth_before": res["penetration_depth_before"],
                "penetration_depth_after": res["penetration_depth_after"],
                "solid_intersection_volume_gt": res["solid_intersection_volume_gt"],
                "solid_intersection_volume_before": res["solid_intersection_volume_before"],
                "solid_intersection_volume_after": res["solid_intersection_volume_after"],
                "disjointness_tip_only_gt": res["disjointness_tip_only_gt"],
                "disjointness_tip_biased_gt": res["disjointness_tip_biased_gt"],
                "disjointness_tip_only_before": res["disjointness_tip_only_before"],
                "disjointness_tip_biased_before": res["disjointness_tip_biased_before"],
                "disjointness_tip_only_after": res["disjointness_tip_only_after"],
                "disjointness_tip_biased_after": res["disjointness_tip_biased_after"],
                "hand_kinetic_score_gt": res["hand_kinetic_score_gt"],
                "hand_kinetic_score_before": res["hand_kinetic_score_before"],
                "hand_kinetic_score_after": res["hand_kinetic_score_after"],
            }
        return False, print_msg, True

    test_sample = cidata[index]

    # ==================== preparation stage >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # region
    # obj: patch dumper point number dismatch
    obj_verts_3d_adapt_np = test_sample[CIAdaptQueries.OBJ_VERTS_3D]
    n_adapt = len(obj_verts_3d_adapt_np)

    obj_verts_3d_np = np.asarray(test_sample[CIDumpedQueries.OBJ_VERTS_3D])
    n_pts = len(obj_verts_3d_np)
    if n_adapt != n_pts:
        obj_verts_3d_np = obj_verts_3d_np[:n_adapt, :]

    # obj: dumped trimesh & normals & tsl & rot
    obj_faces_np = np.asarray(test_sample[CIAdaptQueries.OBJ_FACES])

    rot_matrix = SO3.exp(test_sample[CIDumpedQueries.OBJ_ROT]).as_matrix()
    obj_normals_np = (rot_matrix @ test_sample[CIAdaptQueries.OBJ_NORMAL].T).T

    obj_rot_np = np.asarray(test_sample[CIDumpedQueries.OBJ_ROT])
    obj_tsl_np = np.asarray(test_sample[CIDumpedQueries.OBJ_TSL])

    # obj: adapt tsl & rot
    obj_rot_adapt_np = np.asarray(test_sample[CIAdaptQueries.OBJ_ROT])
    obj_tsl_adapt_np = np.asarray(test_sample[CIAdaptQueries.OBJ_TSL])

    # obj: canonical trimesh & normals
    obj_verts_3d_can_np = np.asarray(test_sample[CIAdaptQueries.OBJ_CAN_VERTS])
    obj_normals_can_np = np.asarray(test_sample[CIAdaptQueries.OBJ_NORMAL])

    # obj: binvox
    obj_vox_can_np = np.asarray(test_sample[CIAdaptQueries.OBJ_VOXEL_POINTS_CAN])
    obj_vox_el_vol = np.asarray(test_sample[CIAdaptQueries.OBJ_VOXEL_EL_VOL])

    # hand: verts & joints adapt, compensation
    hand_verts_adapt_np = np.asarray(test_sample[CIAdaptQueries.HAND_VERTS_3D])
    hand_joints_adapt_np = np.asarray(test_sample[CIAdaptQueries.HAND_JOINTS_3D])

    # hand: faces (adapt) & pose (dumped)
    hand_faces_np = np.asarray(test_sample[CIAdaptQueries.HAND_FACES])
    hand_pose_axisang_adapt_np = np.asarray(test_sample[CIAdaptQueries.HAND_POSE])
    hand_pose_axisang_np = np.asarray(test_sample[CIDumpedQueries.HAND_POSE])
    hand_pose_axisang = torch.from_numpy(hand_pose_axisang_np).float().to(device)
    hand_pose = angle_axis_to_quaternion(hand_pose_axisang)

    # hand: verts & joints dumped
    hand_verts_np = np.asarray(test_sample[CIDumpedQueries.HAND_VERTS_3D])
    hand_joints_np = np.asarray(test_sample[CIDumpedQueries.HAND_JOINTS_3D])
    hand_joints_0 = torch.from_numpy(hand_joints_np[0, ...])

    # hand: close faces => "data/info/closed_hand/hand_mesh_close.obj"
    hand_closed_trimesh = trimesh.load(hand_closed_path, process=False)
    hand_close_faces_np = np.array(hand_closed_trimesh.faces)

    # no viz required
    runtime_viz = None

    # get contact_ratio
    contact_ratio = np.sum(test_sample[CIDumpedQueries.VERTEX_CONTACT]) / len(
        test_sample[CIDumpedQueries.VERTEX_CONTACT]
    )
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== evaluate gt & dumped info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # region
    hand_dist_before = torch.mean(
        torch.norm(
            torch.from_numpy(hand_verts_np).float() - torch.from_numpy(hand_verts_adapt_np).float(),
            p=2,
            dim=1,
        )
    ).item()
    hand_joints_dist_before = torch.mean(
        torch.norm(
            torch.from_numpy(hand_joints_np).float() - torch.from_numpy(hand_joints_adapt_np).float(),
            p=2,
            dim=1,
        )
    ).item()
    # ! center only in ho3d official version
    object_dist_before = (
        torch.mean(
            torch.norm(
                torch.from_numpy(obj_verts_3d_np - test_sample[CIDumpedQueries.HAND_JOINTS_3D][0]).float()
                - torch.from_numpy(obj_verts_3d_adapt_np - test_sample[CIAdaptQueries.HAND_JOINTS_3D][0]).float(),
                p=2,
                dim=1,
            )
        ).item()
        if cidata.hodataset.split_mode == "official"
        else torch.mean(
            torch.norm(
                torch.from_numpy(obj_verts_3d_np).float() - torch.from_numpy(obj_verts_3d_adapt_np).float(),
                p=2,
                dim=1,
            )
        ).item()
    )

    penetration_depth_gt = torch.sqrt(
        penetration_loss_hand_in_obj(
            torch.from_numpy(hand_verts_adapt_np).float(),
            torch.from_numpy(obj_verts_3d_adapt_np).float(),
            torch.from_numpy(obj_faces_np).long(),
        )
    ).item()
    penetration_depth_before = torch.sqrt(
        penetration_loss_hand_in_obj(
            torch.from_numpy(hand_verts_np).float(),
            torch.from_numpy(obj_verts_3d_np).float(),
            torch.from_numpy(obj_faces_np).long(),
        )
    ).item()
    solid_intersection_volume_gt, _, _ = solid_intersection_volume(
        hand_verts_adapt_np,
        hand_close_faces_np,
        obj_vox_can_np,
        obj_tsl_adapt_np,
        obj_rot_adapt_np,
        obj_vox_el_vol,
    )
    solid_intersection_volume_before, _, _ = solid_intersection_volume(
        hand_verts_np,
        hand_close_faces_np,
        obj_vox_can_np,
        obj_tsl_np,
        obj_rot_np,
        obj_vox_el_vol,
    )
    dj_vec_gt, dj_tip_only_gt, dj_tip_biased_gt = region_disjointness_metric(
        hand_verts_adapt_np,
        obj_verts_3d_adapt_np,
        hand_region_assignment,
    )
    dj_vec_before, dj_tip_only_before, dj_tip_biased_before = region_disjointness_metric(
        hand_verts_np,
        obj_verts_3d_np,
        hand_region_assignment,
    )
    hand_ks_gt = kmetric.compute_loss(
        torch.from_numpy(hand_pose_axisang_adapt_np).unsqueeze(0).float(),
        torch.from_numpy(hand_joints_adapt_np).unsqueeze(0).float(),
    ).item()
    hand_ks_before = kmetric.compute_loss(
        torch.from_numpy(hand_pose_axisang_np).unsqueeze(0).float(),
        torch.from_numpy(hand_joints_np).unsqueeze(0).float(),
    ).item()
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== optimize engine >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # region
    use_honet = False
    if contact_ratio < contact_ratio_thresh:
        # return honet result
        hand_verts_pred = torch.from_numpy(hand_verts_np).float()
        hand_joints_pred = torch.from_numpy(hand_joints_np).float()
        hand_pose_pred = hand_pose.detach().cpu().clone()
        obj_verts_pred = torch.from_numpy(obj_verts_3d_np).float()
        use_honet = True
    else:
        # prepare kwargs according to mode
        opt_val_kwargs = dict(
            # static
            vertex_contact=torch.from_numpy(test_sample[CIDumpedQueries.VERTEX_CONTACT]).long().to(device),
            contact_region=torch.from_numpy(test_sample[CIDumpedQueries.CONTACT_REGION_ID]).long().to(device),
            anchor_id=torch.from_numpy(test_sample[CIDumpedQueries.CONTACT_ANCHOR_ID]).long().to(device),
            anchor_elasti=torch.from_numpy(test_sample[CIDumpedQueries.CONTACT_ANCHOR_ELASTI]).float().to(device),
            anchor_padding_mask=torch.from_numpy(test_sample[CIDumpedQueries.CONTACT_ANCHOR_PADDING_MASK])
            .long()
            .to(device),
            hand_region_assignment=torch.from_numpy(hand_region_assignment).long().to(device),
            hand_palm_vertex_mask=torch.from_numpy(hand_palm_vertex_mask).long().to(device),
        )
        if mode == "hand":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_init=torch.from_numpy(test_sample[CIDumpedQueries.HAND_SHAPE]).float().to(device),
                    hand_tsl_init=torch.from_numpy(test_sample[CIDumpedQueries.HAND_TSL]).float().to(device),
                    hand_pose_gt=([0], hand_pose[0:1, :]),
                    hand_pose_init=(list(range(1, 16)), hand_pose[1:, :]),
                    # obj
                    obj_verts_3d_gt=torch.from_numpy(obj_verts_3d_np).float().to(device),
                    obj_normals_gt=torch.from_numpy(obj_normals_np).float().to(device),
                )
            )
        elif mode == "obj":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_gt=torch.from_numpy(test_sample[CIDumpedQueries.HAND_SHAPE]).float().to(device),
                    hand_tsl_gt=torch.from_numpy(test_sample[CIDumpedQueries.HAND_TSL]).float().to(device),
                    hand_pose_gt=(list(range(0, 16)), hand_pose[0:, :]),
                    # obj
                    obj_verts_3d_can=torch.from_numpy(obj_verts_3d_can_np).float().to(device),
                    obj_normals_can=torch.from_numpy(obj_normals_can_np).float().to(device),
                    obj_tsl_init=torch.from_numpy(test_sample[CIDumpedQueries.OBJ_TSL]).float().to(device),
                    obj_rot_init=torch.from_numpy(test_sample[CIDumpedQueries.OBJ_ROT]).float().to(device),
                )
            )
        elif mode == "hand_obj":
            opt_val_kwargs.update(
                dict(
                    # hand
                    hand_shape_init=torch.from_numpy(test_sample[CIDumpedQueries.HAND_SHAPE]).float().to(device),
                    hand_tsl_init=torch.from_numpy(test_sample[CIDumpedQueries.HAND_TSL]).float().to(device),
                    hand_pose_gt=([0], hand_pose[0:1, :]),
                    hand_pose_init=(list(range(1, 16)), hand_pose[1:, :]),
                    # obj
                    obj_verts_3d_can=torch.from_numpy(obj_verts_3d_can_np).float().to(device),
                    obj_normals_can=torch.from_numpy(obj_normals_can_np).float().to(device),
                    obj_tsl_init=torch.from_numpy(test_sample[CIDumpedQueries.OBJ_TSL]).float().to(device),
                    obj_rot_init=torch.from_numpy(test_sample[CIDumpedQueries.OBJ_ROT]).float().to(device),
                )
            )
        else:
            raise KeyError(f"unknown optimization mode {mode}")
        opt_val_kwargs.update(
            dict(
                # hand compensate
                hand_compensate_root=hand_joints_0.float().to(device),
                # viz
                runtime_vis=runtime_viz,
            )
        )

        hoptim.set_opt_val(**opt_val_kwargs)

        hoptim.optimize(progress=False)

        hand_verts_pred, hand_joints_pred, hand_transf_pred = hoptim.recover_hand()
        hand_verts_pred = hand_verts_pred.cpu()
        hand_joints_pred = hand_joints_pred.cpu()
        hand_pose_pred = hoptim.recover_hand_pose().cpu()
        obj_verts_pred = hoptim.recover_obj()
        obj_verts_pred = obj_verts_pred.cpu()
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== eval >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # region
    hand_dist_after = torch.mean(
        torch.norm(torch.from_numpy(hand_verts_adapt_np).float() - hand_verts_pred, p=2, dim=1)
    ).item()
    hand_joints_dist_after = torch.mean(
        torch.norm(torch.from_numpy(hand_joints_adapt_np).float() - hand_joints_pred, p=2, dim=1)
    ).item()
    # ! center only in ho3d official version
    object_dist_after = (
        torch.mean(
            torch.norm(
                torch.from_numpy(obj_verts_3d_adapt_np - test_sample[CIAdaptQueries.HAND_JOINTS_3D][0]).float()
                - (obj_verts_pred - hand_joints_pred[0]),
                p=2,
                dim=1,
            )
        ).item()
        if cidata.hodataset.split_mode == "official"
        else torch.mean(
            torch.norm(
                torch.from_numpy(obj_verts_3d_adapt_np).float() - obj_verts_pred,
                p=2,
                dim=1,
            )
        ).item()
    )
    penetration_depth_after = torch.sqrt(
        penetration_loss_hand_in_obj(
            hand_verts_pred,
            obj_verts_pred,
            torch.from_numpy(obj_faces_np).long(),
        )
    ).item()
    # ! dispatch given mode
    if contact_ratio < contact_ratio_thresh:
        obj_tsl_final_np = obj_tsl_np
        obj_rot_final_np = obj_rot_np
    else:
        if mode == "obj" or mode == "hand_obj":
            # obj optimiziing option on
            obj_tsl_final_np = hoptim.obj_tsl_np()
            obj_rot_final_np = hoptim.obj_rot_np()
        else:
            obj_tsl_final_np = obj_tsl_np
            obj_rot_final_np = obj_rot_np
    solid_intersection_volume_after, _, _ = solid_intersection_volume(
        np.asarray(hand_verts_pred).astype(np.float64),
        hand_close_faces_np,
        obj_vox_can_np,
        obj_tsl_final_np,
        obj_rot_final_np,
        obj_vox_el_vol,
    )
    dj_vec_after, dj_tip_only_after, dj_tip_biased_after = region_disjointness_metric(
        np.asarray(hand_verts_pred), np.asarray(obj_verts_pred), hand_region_assignment
    )
    hand_ks_after = kmetric.compute_loss(
        quaternion_to_angle_axis(hand_pose_pred.unsqueeze(0)), hand_joints_pred.unsqueeze(0)
    ).item()

    # res dict
    res = {
        "hand_verts_pred": hand_verts_pred.numpy(),
        "hand_joints_pred": hand_joints_pred.numpy(),
        "obj_verts_pred": obj_verts_pred.numpy(),
        "hand_dist_before": hand_dist_before,
        "hand_dist_after": hand_dist_after,
        "hand_joints_dist_before": hand_joints_dist_before,
        "hand_joints_dist_after": hand_joints_dist_after,
        "object_dist_before": object_dist_before,
        "object_dist_after": object_dist_after,
        "penetration_depth_gt": penetration_depth_gt,
        "penetration_depth_before": penetration_depth_before,
        "penetration_depth_after": penetration_depth_after,
        "solid_intersection_volume_gt": solid_intersection_volume_gt * 1e6,
        "solid_intersection_volume_before": solid_intersection_volume_before * 1e6,
        "solid_intersection_volume_after": solid_intersection_volume_after * 1e6,
        "disjointness_vector_gt": dj_vec_gt,
        "disjointness_tip_only_gt": dj_tip_only_gt,
        "disjointness_tip_biased_gt": dj_tip_biased_gt,
        "disjointness_vector_before": dj_vec_before,
        "disjointness_tip_only_before": dj_tip_only_before,
        "disjointness_tip_biased_before": dj_tip_biased_before,
        "disjointness_vector_after": dj_vec_after,
        "disjointness_tip_only_after": dj_tip_only_after,
        "disjointness_tip_biased_after": dj_tip_biased_after,
        "hand_kinetic_score_gt": hand_ks_gt,
        "hand_kinetic_score_before": hand_ks_before,
        "hand_kinetic_score_after": hand_ks_after,
        "image_path": test_sample[CIAdaptQueries.IMAGE_PATH],
    }

    # save result
    with open(save_file, "wb") as fstream:
        pickle.dump(res, fstream)

    # print msg
    print_msg = {
        "hand_dist_before": res["hand_dist_before"],
        "hand_dist_after": res["hand_dist_after"],
        "hand_joints_dist_before": res["hand_joints_dist_before"],
        "hand_joints_dist_after": res["hand_joints_dist_after"],
        "object_dist_before": res["object_dist_before"],
        "object_dist_after": res["object_dist_after"],
        "penetration_depth_gt": res["penetration_depth_gt"],
        "penetration_depth_before": res["penetration_depth_before"],
        "penetration_depth_after": res["penetration_depth_after"],
        "solid_intersection_volume_gt": res["solid_intersection_volume_gt"],
        "solid_intersection_volume_before": res["solid_intersection_volume_before"],
        "solid_intersection_volume_after": res["solid_intersection_volume_after"],
        "disjointness_tip_only_gt": res["disjointness_tip_only_gt"],
        "disjointness_tip_biased_gt": res["disjointness_tip_biased_gt"],
        "disjointness_tip_only_before": res["disjointness_tip_only_before"],
        "disjointness_tip_biased_before": res["disjointness_tip_biased_before"],
        "disjointness_tip_only_after": res["disjointness_tip_only_after"],
        "disjointness_tip_biased_after": res["disjointness_tip_biased_after"],
        "hand_kinetic_score_gt": res["hand_kinetic_score_gt"],
        "hand_kinetic_score_before": res["hand_kinetic_score_before"],
        "hand_kinetic_score_after": res["hand_kinetic_score_after"],
    }

    return True, print_msg, use_honet
    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def worker(
    device,
    worker_id,
    n_workers,
    data_path,
    mano_root,
    hodata_path,
    anchor_path,
    palm_path,
    hodata_use_cache,
    hodata_center_idx,
    fhb,
    load_fhb_path,
    compensate_tsl,
    lr,
    n_iter,
    mode,
    save_prefix,
    contact_ratio_thresh,
    hand_closed_path,
    lambda_contact_loss,
    lambda_repulsion_loss,
    repulsion_query,
    repulsion_threshold,
):
    ci_online = CIOnline(
        data_path, hodata_path, anchor_path, hodata_use_cache=hodata_use_cache, hodata_center_idx=hodata_center_idx
    )
    hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)

    param_str = f"lcl{lambda_contact_loss}_lrl{lambda_repulsion_loss}_rq{repulsion_query}_rt{repulsion_threshold}"
    param_str += f"_ctsl{'' if compensate_tsl else '(x)'}"

    if save_prefix is None:
        # need to figure out save prefix
        save_prefix = os.path.join(
            "common/optimize", ci_online.hodataset.name, ci_online.hodataset.split_mode, param_str, mode
        )
    # check make dir
    os.makedirs(save_prefix, exist_ok=True)

    begin_index = worker_id * len(ci_online) // n_workers
    end_index = (worker_id + 1) * len(ci_online) // n_workers
    cprint(
        f"====== {worker_id:>3} begin: {begin_index:0>4} end: {end_index:0>4} len: {len(ci_online)} >>>>>>",
        "cyan",
    )
    cprint(
        f"====== {worker_id:>3} using device: {device} >>>>>>",
        "cyan",
    )
    hoptim = GeOptimizer(
        device,
        lr=lr,
        n_iter=n_iter,
        verbose=False,
        mano_root=mano_root,
        anchor_path=anchor_path,
        fhb=fhb,
        load_fhb_path=load_fhb_path,
        compensate_tsl=compensate_tsl,
        lambda_contact_loss=lambda_contact_loss,
        lambda_repulsion_loss=lambda_repulsion_loss,
        repulsion_query=repulsion_query,
        repulsion_threshold=repulsion_threshold,
    )
    kmetric = AnatomyMetric()
    cprint(
        f"====== optimizer created on device: {device} >>>>>>",
        "cyan",
    )

    res_list = []
    use_honet_cnt = 0
    for i in range(begin_index, end_index):
        try:
            cprint(f"       {worker_id:>3} processing: {i:0>4}, mode: {mode}, param: {param_str}", "yellow")
            time_start = time.time()
            flag, res, use_honet = run_sample_by_idx(
                device=device,
                hoptim=hoptim,
                mode=mode,
                cidata=ci_online,
                kmetric=kmetric,
                index=i,
                hand_region_assignment=hand_region_assignment,
                hand_palm_vertex_mask=hand_palm_vertex_mask,
                save_path=save_prefix,
                contact_ratio_thresh=contact_ratio_thresh,
                hand_closed_path=hand_closed_path,
            )
            if not flag:
                cprint(f" x     {worker_id:>3} skip: {i:0>4}", "yellow")
            res_list.append(res)
            time_end = time.time()

            use_honet_cnt += 1 if use_honet else 0
            better_color = "blue" if not use_honet else "white"
            worse_color = "red" if not use_honet else "white"

            print_line = f"   x   {worker_id:>3} processed: {i:0>4} elapsed {round(time_end - time_start):>4} result: "
            print_line += colored(f"HDB={res['hand_dist_before']:.4f}, ", "white")
            print_line += colored(f"ODB={res['object_dist_before']:.4f}, ", "white")
            print_line += colored(f"PD:GT={res['penetration_depth_gt']:.4f}, ", "white")
            print_line += colored(f"PD:B={res['penetration_depth_before']:.4f}, ", "white")
            print_line += colored(f"SI:GT={res['solid_intersection_volume_gt']:.4f}, ", "white")
            print_line += colored(f"SI:B={res['solid_intersection_volume_before']:.4f}, ", "white")
            print_line += "\n"

            print_line += f"   |   {worker_id:>3} processed: {i:0>4}           continue:  "
            print_line += colored(f"DJ_TO:GT={res['disjointness_tip_only_gt']:.4f}, ", "white")
            print_line += colored(f"DJ_TO:B={res['disjointness_tip_only_before']:.4f}, ", "white")
            print_line += colored(f"DJ_TB:GT={res['disjointness_tip_biased_gt']:.4f}, ", "white")
            print_line += colored(f"DJ_TB:B={res['disjointness_tip_biased_before']:.4f}, ", "white")
            print_line += colored(f"HKS:GT={res['hand_kinetic_score_gt']:.4f}, ", "white")
            print_line += colored(f"HKS:B={res['hand_kinetic_score_before']:.4f}, ", "white")
            print_line += "\n"

            print_line += f"   |   {worker_id:>3} processed: {i:0>4}           continue:  "
            color_str = better_color if res["hand_dist_after"] < res["hand_dist_before"] else worse_color
            print_line += colored(f"HDA={res['hand_dist_after']:.4f}, ", color_str)
            color_str = better_color if res["object_dist_after"] < res["object_dist_before"] else worse_color
            print_line += colored(f"ODA={res['object_dist_after']:.4f}, ", color_str)
            color_str = (
                better_color if res["penetration_depth_after"] < res["penetration_depth_before"] else worse_color
            )
            print_line += colored(f"PD:A={res['penetration_depth_after']:.4f}, ", color_str)
            color_str = (
                better_color
                if res["solid_intersection_volume_after"] < res["solid_intersection_volume_before"]
                else worse_color
            )
            print_line += colored(f"SI:A={res['solid_intersection_volume_after']:.4f}, ", color_str)
            color_str = (
                better_color
                if res["disjointness_tip_only_after"] < res["disjointness_tip_only_before"]
                else worse_color
            )
            print_line += colored(f"DJ_TO:A={res['disjointness_tip_only_after']:.4f}, ", color_str)
            color_str = (
                better_color
                if res["disjointness_tip_biased_after"] < res["disjointness_tip_biased_before"]
                else worse_color
            )
            print_line += colored(f"DJ_TB:A={res['disjointness_tip_biased_after']:.4f}, ", color_str)
            color_str = (
                better_color if res["hand_kinetic_score_after"] < res["hand_kinetic_score_before"] else worse_color
            )
            print_line += colored(f"HKS:A={res['hand_kinetic_score_after']:.4f}, ", color_str)

            print(print_line)
        except Exception as e:
            exc_trace = traceback.format_exc()
            err_msg = f"  x    {worker_id:>3}: sample {i:0>4}, \n{exc_trace}"
            cprint(err_msg, "red")

    cprint(
        f"====== {worker_id:>3} conclude <<<<<<",
        "cyan",
    )
    return res_list, use_honet_cnt


def main(
    n_workers,
    data_path,
    mano_root,
    hodata_path,
    anchor_path,
    palm_path,
    hodata_use_cache,
    hodata_center_idx,
    fhb,
    load_fhb_path,
    compensate_tsl,
    lr,
    n_iter,
    mode,
    save_prefix,
    contact_ratio_thresh,
    hand_closed_path,
    lambda_contact_loss,
    lambda_repulsion_loss,
    repulsion_query,
    repulsion_threshold,
):
    # get all cuda device ids
    device_count = torch.cuda.device_count()
    # create device for each worker
    device_list = []
    for worker_id in range(n_workers):
        device_list.append(torch.device(f"cuda:{worker_id % device_count}"))

    # initial jobs
    collected = Parallel(n_jobs=n_workers)(
        delayed(worker)(
            device=device_list[worker_id],
            worker_id=worker_id,
            n_workers=n_workers,
            data_path=data_path,
            mano_root=mano_root,
            hodata_path=hodata_path,
            anchor_path=anchor_path,
            palm_path=palm_path,
            hodata_use_cache=hodata_use_cache,
            hodata_center_idx=hodata_center_idx,
            fhb=fhb,
            load_fhb_path=load_fhb_path,
            lr=lr,
            compensate_tsl=compensate_tsl,
            n_iter=n_iter,
            mode=mode,
            save_prefix=save_prefix,
            contact_ratio_thresh=contact_ratio_thresh,
            hand_closed_path=hand_closed_path,
            lambda_contact_loss=lambda_contact_loss,
            lambda_repulsion_loss=lambda_repulsion_loss,
            repulsion_query=repulsion_query,
            repulsion_threshold=repulsion_threshold,
        )
        for worker_id in list(range(n_workers))
    )

    # post process
    use_honet = 0
    for c in collected:
        use_honet += c[1]
    collapsed = collapse_res_list(collected)
    merged = merge_res_list(collapsed)
    summarize(merged)
    cprint(f"use honet samples {use_honet}", "blue")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--hodata_path", type=str, default="data")
    parser.add_argument("--mano_root", type=str, default="assets/mano")
    parser.add_argument("--anchor_path", type=str, default="assets/anchor")
    parser.add_argument("--palm_path", type=str, default="assets/hand_palm_full.txt")
    parser.add_argument("--hodata_use_cache", action="store_true")
    parser.add_argument("--hodata_no_use_cache", action="store_true")
    parser.add_argument("--hodata_center_idx", type=int, default=9)
    parser.add_argument("--fhb", action="store_true")
    parser.add_argument("--load_fhb_path", type=str, default="assets/mano/fhb_skel_centeridx9.pkl")
    parser.add_argument("--compensate_tsl", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_iter", type=int, default=400)
    parser.add_argument("--mode", type=str, choices=["hand", "obj", "hand_obj"], default="hand")
    parser.add_argument("--save_prefix", type=str, default=None)
    parser.add_argument("--contact_ratio_thresh", type=float, default=0.01)
    parser.add_argument("--hand_closed_path", type=str, default="assets/closed_hand/hand_mesh_close.obj")
    parser.add_argument("--lambda_contact_loss", type=float, default=10.0)
    parser.add_argument("--lambda_repulsion_loss", type=float, default=0.5)
    parser.add_argument("--repulsion_query", type=float, default=0.030)
    parser.add_argument("--repulsion_threshold", type=float, default=0.080)
    parser.add_argument("--fresh_start", action="store_true")  # TODO
    parser.add_argument("--refresh_score", action="store_true")  # TODO

    args = parser.parse_args()

    # deal with dual flags
    if not args.hodata_use_cache and not args.hodata_no_use_cache:
        g_hodata_use_cache = True
    elif not args.hodata_use_cache and args.hodata_no_use_cache:
        g_hodata_use_cache = False
    elif args.hodata_use_cache and not args.hodata_no_use_cache:
        g_hodata_use_cache = True
    else:
        g_hodata_use_cache = True

    # main
    main(
        n_workers=args.n_workers,
        data_path=args.data_path,
        mano_root=args.mano_root,
        hodata_path=args.hodata_path,
        anchor_path=args.anchor_path,
        palm_path=args.palm_path,
        hodata_use_cache=g_hodata_use_cache,
        hodata_center_idx=args.hodata_center_idx,
        fhb=args.fhb,
        load_fhb_path=args.load_fhb_path,
        compensate_tsl=args.compensate_tsl,
        lr=args.lr,
        n_iter=args.n_iter,
        mode=args.mode,
        save_prefix=args.save_prefix,
        contact_ratio_thresh=args.contact_ratio_thresh,
        hand_closed_path=args.hand_closed_path,
        lambda_contact_loss=args.lambda_contact_loss,
        lambda_repulsion_loss=args.lambda_repulsion_loss,
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
    )
