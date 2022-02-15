import argparse
import random
from copy import deepcopy
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import trimesh
from hocontact.hodatasets.hodata import HOdata, ho_collate
from hocontact.models.picr import PicrHourglassPointNet
from hocontact.postprocess.geo_optimizer import GeOptimizer, init_runtime_viz, update_runtime_viz
from hocontact.utils import ioutils
from hocontact.utils.anatomyutils import AnatomyMetric
from hocontact.utils.collisionutils import penetration_loss_hand_in_obj, solid_intersection_volume
from hocontact.utils.contactutils import dumped_process_contact_info
from hocontact.utils.disjointnessutils import region_disjointness_metric
from liegroups import SO3
from manopth.anchorutils import anchor_load, get_rev_anchor_mapping, masking_load_driver
from manopth.manolayer import ManoLayer
from manopth.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis
from matplotlib import pyplot as plt
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

np.seterr(all="raise")
plt.switch_backend("agg")


def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def process_result(results, vc_thresh, mano_root, anchor_root):
    _, _, _, anchor_mapping = anchor_load(anchor_root)
    rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping)
    honet_fields = [
        "hand_tsl",
        "hand_joints_3d",
        "hand_verts_3d",
        "hand_full_pose",
        # "hand_pca_pose", # NOTE: this fields is not maintained in current version
        "hand_shape",
        "obj_tsl",
        "obj_rot",
        "obj_verts_3d",
    ]
    # ====== assert fields in results: contact related
    assert "recov_vertex_contact" in results, f"vertex_contact not found"
    assert "recov_contact_region" in results, f"contact_region not found"
    assert "recov_anchor_elasti" in results, f"anchor_elasti not found"
    recov_vertex_contact = results["recov_vertex_contact"].detach()  # TENSOR[B, N]
    recov_contact_in_image_mask = results["recov_contact_in_image_mask"].detach()  # TENSOR[B, N]
    recov_contact_region = results["recov_contact_region"].detach()  # TENSOR[B, N, 17]
    recov_anchor_elasti_pred = results["recov_anchor_elasti"].detach()  # TENSOR[B, N, 4]
    collate_mask = torch.ones_like(recov_vertex_contact)

    recov_vertex_contact_pred = (torch.sigmoid(recov_vertex_contact) > vc_thresh).bool()  # TENSOR[B, N]
    recov_contact_region_pred = torch.argmax(recov_contact_region, dim=2)  # TENSOR[B, N]

    # ====== assert fields in results: honet related
    for field in honet_fields:
        assert field in results, f"{field} not found"

    # ==================== dump contact related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    idx = 0
    sample_collate_mask = collate_mask[idx, :].bool()  # TENSOR[N, ]
    sample_vertex_contact = recov_vertex_contact_pred[idx, :]  # TENSOR[N, ]
    sample_contact_in_image_mask = recov_contact_in_image_mask[idx, :]  # TENSOR[N, ]
    combined_vertex_contact = sample_vertex_contact.bool() & sample_contact_in_image_mask.bool()  # TENSOR[N,]
    filtered_vertex_contact = combined_vertex_contact[sample_collate_mask]  # TENSOR[X, ]

    sample_contact_region = recov_contact_region_pred[idx, :]  # TENSOR[N, ]
    filtered_contact_region = sample_contact_region[sample_collate_mask]  # TENSOR[X, ]
    sample_anchor_elasti = recov_anchor_elasti_pred[idx, :, :]  # TENSOR[N, 4]
    filtered_anchor_elasti = sample_anchor_elasti[sample_collate_mask, :]  # TENSOR[X, 4]

    # transport from cuda to cpu
    filtered_vertex_contact = filtered_vertex_contact.cpu()
    filtered_contact_region = filtered_contact_region.cpu()
    filtered_anchor_elasti = filtered_anchor_elasti.cpu()

    # iterate over all points
    sample_res = []
    n_points = filtered_vertex_contact.shape[0]  # X
    for p_idx in range(n_points):
        p_contact = int(filtered_vertex_contact[p_idx])
        if p_contact == 0:
            p_res = {
                "contact": 0,
            }
        else:  # p_contact == 1
            p_region = int(filtered_contact_region[p_idx])
            p_anchor_id = rev_anchor_mapping[p_region]
            p_n_anchor = len(p_anchor_id)
            p_anchor_elasti = filtered_anchor_elasti[p_idx, :p_n_anchor].tolist()
            p_res = {
                "contact": 1,
                "region": p_region,
                "anchor_id": p_anchor_id,
                "anchor_elasti": p_anchor_elasti,
            }
        sample_res.append(p_res)

    (vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask) = dumped_process_contact_info(
        deepcopy(sample_res), anchor_mapping, pad_vertex=True, pad_anchor=True, elasti_th=0.00
    )
    new_sample_res = {
        "vertex_contact": vertex_contact,
        "hand_region": hand_region,
        "anchor_id": anchor_id,
        "anchor_elasti": anchor_elasti,
        "anchor_padding_mask": anchor_padding_mask,
    }
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== dump honet related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    honet_res = {}
    for field in honet_fields:
        if field in ["obj_verts_3d"]:
            honet_res[field] = results[field][idx, ...][sample_collate_mask, :].detach().cpu()
        honet_res[field] = results[field][idx, ...].detach().cpu()
    verts = honet_res["hand_verts_3d"]
    joints = honet_res["hand_joints_3d"]
    obj_verts_3d = honet_res["obj_verts_3d"]
    obj_tsl = honet_res["obj_tsl"].reshape((3,))
    obj_rot = honet_res["obj_rot"].reshape((3,))

    tmp_layer = ManoLayer(
        joint_rot_mode="quat",
        root_rot_mode="quat",
        use_pca=False,
        mano_root=mano_root,
        center_idx=9,
        flat_hand_mean=True,
        return_transf=True,
        return_full_pose=True,
    )
    # adjust honet_res hand_tsl, as we have different manolayer
    pose = angle_axis_to_quaternion(honet_res["hand_full_pose"].reshape((16, 3)))
    shape = honet_res["hand_shape"]
    tsl = honet_res["hand_tsl"].reshape((3,))
    _, rebuild_joints, _, _ = tmp_layer(pose.reshape((1, 64)), shape.reshape((1, 10)))
    rebuild_joints = rebuild_joints + tsl
    rebuild_joints = rebuild_joints.squeeze(0)
    new_tsl = tsl + joints[0] - rebuild_joints[0]  # align to honet_res

    new_honet_res = {
        "hand_verts_3d": verts.detach().clone().cpu().numpy(),
        "hand_joints_3d": joints.detach().clone().cpu().numpy(),
        "hand_pose": pose.detach().clone().cpu().numpy(),
        "hand_shape": shape.detach().clone().cpu().numpy(),
        "hand_tsl": new_tsl.detach().clone().cpu().numpy(),
        "obj_verts_3d": obj_verts_3d.detach().clone().cpu().numpy(),
        "obj_tsl": obj_tsl.detach().clone().cpu().numpy(),
        "obj_rot": obj_rot.detach().clone().cpu().numpy(),
    }

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return (new_sample_res, new_honet_res)


def epoch_pass(rank, prefix, loader, model, epoch=0):
    cprint(f"{prefix.capitalize()} Epoch {epoch}", "blue")

    # model will always be in eval mode
    model.eval()

    res = []

    # ==================== Forward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # loop over dataset
    loader = tqdm(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # model
            ls_results = model(batch, rank=rank)
            res.append(ls_results[-1])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return res


def picr_stage(args):
    rank = args.gpu
    target_device = f"cuda:{rank}"
    set_all_seeds(args.manual_seed)

    if args.exp_keyword is None:
        now = datetime.now()
        exp_keyword = f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}"
    else:
        exp_keyword = args.exp_keyword

    dat_str = "example"
    split_str = "example"
    exp_id = f"checkpoints/picr_geo_example/{dat_str}_{split_str}"
    exp_id = f"{exp_id}/{exp_keyword}"

    ioutils.print_args(args)

    cprint(
        f"Saving experiment logs, models, and training curves and images to {exp_id}",
        "white",
    )

    # ==================== Creating Datasets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    example_dataset = HOdata.get_dataset(
        dataset="fhb_example",
        data_root=args.data_root,
        data_split="example",
        split_mode="example",
        use_cache=True,
        mini_factor=1.0,
        center_idx=9,
        enable_contact=True,
        filter_no_contact=True,
        filter_thresh=10.0,
        synt_factor=1,
    )
    ioutils.print_query(example_dataset.queries, desp="example_dataset_queries")
    example_loader = DataLoader(
        example_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=ho_collate,
        pin_memory=True,
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== initialize model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    _model = PicrHourglassPointNet(
        hg_stacks=args.hg_stacks,
        hg_blocks=args.hg_blocks,
        hg_classes=args.hg_classes,
        obj_scale_factor=args.obj_scale_factor,
        honet_resnet_version=args.honet_resnet_version,
        honet_center_idx=args.center_idx,
        honet_mano_lambda_recov_joints3d=args.honet_mano_lambda_recov_joints3d,
        honet_mano_lambda_recov_verts3d=args.honet_mano_lambda_recov_verts3d,
        honet_mano_lambda_shape=args.honet_mano_lambda_shape,
        honet_mano_lambda_pose_reg=args.honet_mano_lambda_pose_reg,
        honet_obj_lambda_recov_verts3d=args.honet_obj_lambda_recov_verts3d,
        honet_obj_trans_factor=args.honet_obj_trans_factor,
        honet_mano_fhb_hand=args.honet_mano_fhb_hand,
    )
    model = _model.to(target_device)

    # check init_ckpt option
    if args.init_ckpt is None:
        cprint("no initializing checkpoint provided. abort!", "red")
        exit()
    map_location = f"cuda:{rank}"
    _ = ioutils.reload_checkpoint(model, resume_path=args.init_ckpt, as_parallel=False, map_location=map_location)
    # only weights is reloaded, others are dropped

    # ====== print model size information
    cprint(f"Model total size == {ioutils.param_size(model)} MB")
    cprint(f"  |  HONet total size == {ioutils.param_size(model.ho_net)} MB")
    cprint(f"  |  BaseNet total size == {ioutils.param_size(model.base_net)} MB")
    cprint(f"  \\  ContactHead total size == {ioutils.param_size(model.contact_head)} MB")
    cprint(f"    |  EncodeModule total size == {ioutils.param_size(model.contact_head.encoder)} MB")
    decode_vertex_contact_size = ioutils.param_size(model.contact_head.vertex_contact_decoder)
    decode_contact_region_size = ioutils.param_size(model.contact_head.contact_region_decoder)
    decode_anchor_elasti_size = ioutils.param_size(model.contact_head.anchor_elasti_decoder)
    cprint(f"    |  DecodeModule_VertexContact total size == {decode_vertex_contact_size} MB")
    cprint(f"    |  DecodeModule_ContactRegion total size == {decode_contact_region_size} MB")
    cprint(f"    |  DecodeModule_AnchorElasti total size == {decode_anchor_elasti_size} MB")

    # ==================== dumping train >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # epoch pass
    epoch_res = epoch_pass(rank, "example", example_loader, model, epoch=0)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # =================== process results >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    stage_res = []
    for res_item in epoch_res:
        total_res = {}
        picr_res, honet_res = process_result(res_item, args.vertex_contact_thresh, args.mano_root, args.anchor_root)
        total_res.update(picr_res)
        total_res.update(honet_res)
        stage_res.append(total_res)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    cprint("\nPICR DONE!", "cyan")
    return stage_res


viz_initialized = False
runtime_viz = None


def run_sample(
    device,
    hoptim,
    info,
    hodataset,
    index,
    kmetric,
    hand_region_assignment,
    hand_palm_vertex_mask,
    hand_closed_path="assets/closed_hand/hand_mesh_close.obj",
):
    global viz_initialized, runtime_viz
    # ==================== preparation stage >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # region
    # obj: patch dumper point number dismatch
    obj_verts_3d_adapt_np = hodataset.get_obj_verts_transf(index)
    obj_verts_3d_np = np.asarray(info["obj_verts_3d"])

    # obj: dumped trimesh & normals & tsl & rot
    obj_faces_np = np.asarray(hodataset.get_obj_faces(index))

    rot_matrix = SO3.exp(info["obj_rot"]).as_matrix()
    obj_normals_np = (rot_matrix @ (hodataset.get_obj_normal(index)).T).T

    obj_rot_np = np.asarray(info["obj_rot"])
    obj_tsl_np = np.asarray(info["obj_tsl"])

    # obj: adapt tsl & rot
    obj_rot_adapt_np = np.asarray(hodataset.get_obj_rot(index))
    obj_tsl_adapt_np = np.asarray(hodataset.get_obj_tsl(index))

    # obj: canonical trimesh & normals
    obj_verts_3d_can_np = np.asarray(hodataset.get_obj_verts_can(index)[0])
    obj_normals_can_np = np.asarray(hodataset.get_obj_normal(index))

    # obj: binvox
    obj_vox_can_np = np.asarray(hodataset.get_obj_voxel_points_can(index))
    obj_vox_el_vol = np.asarray(hodataset.get_obj_voxel_element_volume(index))

    # hand: verts & joints adapt, compensation
    hand_verts_adapt_np = np.asarray(hodataset.get_hand_verts3d(index))
    hand_joints_adapt_np = np.asarray(hodataset.get_joints3d(index))

    # hand: faces (adapt) & pose (dumped)
    hand_faces_np = np.asarray(hodataset.get_hand_faces(index))
    hand_pose_axisang_adapt_np = np.asarray(hodataset.get_hand_pose_wrt_cam(index))
    hand_pose_np = np.asarray(info["hand_pose"])
    hand_pose = torch.from_numpy(hand_pose_np).float().to(device)
    hand_pose_axisang_np = quaternion_to_angle_axis(hand_pose).detach().cpu().numpy()

    # hand: verts & joints dumped
    hand_verts_np = np.asarray(info["hand_verts_3d"])
    hand_joints_np = np.asarray(info["hand_joints_3d"])

    # hand: close faces => "data/info/closed_hand/hand_mesh_close.obj"
    hand_closed_trimesh = trimesh.load(hand_closed_path, process=False)
    hand_close_faces_np = np.array(hand_closed_trimesh.faces)

    # no viz required
    if not viz_initialized:
        runtime_viz = init_runtime_viz(
            hand_verts_adapt_np,
            hand_verts_np,
            obj_verts_3d_adapt_np,
            hand_faces_np,
            obj_verts_3d_np,
            obj_faces_np,
            contact_info=info,
        )
        viz_initialized = True
    else:
        update_runtime_viz(
            runtime_viz,
            hand_verts_adapt_np,
            hand_verts_np,
            obj_verts_3d_adapt_np,
            obj_verts_3d_np,
            hand_faces_np,
            obj_faces_np,
        )
    # runtime_viz = None

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
    # no joint score, as hand model different
    object_dist_before = torch.mean(
        torch.norm(
            torch.from_numpy(obj_verts_3d_np).float() - torch.from_numpy(obj_verts_3d_adapt_np).float(),
            p=2,
            dim=1,
        )
    ).item()
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
        hand_verts_adapt_np, hand_close_faces_np, obj_vox_can_np, obj_tsl_adapt_np, obj_rot_adapt_np, obj_vox_el_vol
    )
    solid_intersection_volume_before, _, _ = solid_intersection_volume(
        hand_verts_np, hand_close_faces_np, obj_vox_can_np, obj_tsl_np, obj_rot_np, obj_vox_el_vol
    )
    dj_vec_gt, dj_tip_only_gt, dj_tip_biased_gt = region_disjointness_metric(
        hand_verts_adapt_np, obj_verts_3d_adapt_np, hand_region_assignment
    )
    dj_vec_before, dj_tip_only_before, dj_tip_biased_before = region_disjointness_metric(
        hand_verts_np, obj_verts_3d_np, hand_region_assignment
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
    # prepare kwargs according to mode
    opt_val_kwargs = dict(
        # static
        vertex_contact=torch.from_numpy(info["vertex_contact"]).long().to(device),
        contact_region=torch.from_numpy(info["hand_region"]).long().to(device),
        anchor_id=torch.from_numpy(info["anchor_id"]).long().to(device),
        anchor_elasti=torch.from_numpy(info["anchor_elasti"]).float().to(device),
        anchor_padding_mask=torch.from_numpy(info["anchor_padding_mask"]).long().to(device),
        # hand
        hand_shape_init=torch.from_numpy(info["hand_shape"]).float().to(device),
        hand_tsl_init=torch.from_numpy(info["hand_tsl"]).float().to(device),
        hand_pose_gt=([0], hand_pose[0:1, :]),
        hand_pose_init=(list(range(1, 16)), hand_pose[1:, :]),
        # obj
        obj_verts_3d_can=torch.from_numpy(obj_verts_3d_can_np).float().to(device),
        obj_normals_can=torch.from_numpy(obj_normals_can_np).float().to(device),
        obj_tsl_init=torch.from_numpy(info["obj_tsl"]).float().to(device),
        obj_rot_init=torch.from_numpy(info["obj_rot"]).float().to(device),
        # viz
        runtime_vis=runtime_viz,
    )

    hoptim.set_opt_val(**opt_val_kwargs)

    hoptim.optimize(progress=True)

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
    # no joint score, as hand models differs
    object_dist_after = torch.mean(
        torch.norm(
            torch.from_numpy(obj_verts_3d_adapt_np).float() - obj_verts_pred,
            p=2,
            dim=1,
        )
    ).item()
    penetration_depth_after = torch.sqrt(
        penetration_loss_hand_in_obj(
            hand_verts_pred,
            obj_verts_pred,
            torch.from_numpy(obj_faces_np).long(),
        )
    ).item()
    obj_tsl_final_np = hoptim.obj_tsl_np()
    obj_rot_final_np = hoptim.obj_rot_np()
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
    }

    # print msg
    print_msg = {
        "hand_dist_before": res["hand_dist_before"],
        "hand_dist_after": res["hand_dist_after"],
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
    return print_msg


def geo_stage(intermediate, args):
    rank = args.gpu
    target_device = f"cuda:{rank}"
    set_all_seeds(args.manual_seed)

    example_dataset = HOdata.get_dataset(
        dataset="fhb_example",
        data_root=args.data_root,
        data_split="example",
        split_mode="example",
        use_cache=True,
        mini_factor=1.0,
        center_idx=9,
        enable_contact=True,
        filter_no_contact=True,
        filter_thresh=10.0,
        synt_factor=1,
    )
    hoptim = GeOptimizer(
        target_device,
        lr=1e-2,
        n_iter=200,
        verbose=False,
        mano_root=args.mano_root,
        anchor_path=args.anchor_root,
        # values to initialize coef_val
        lambda_contact_loss=args.lambda_contact_loss,
        lambda_repulsion_loss=args.lambda_repulsion_loss,
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
    )
    kmetric = AnatomyMetric()
    hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(args.anchor_root, args.palm_path)
    for index in range(len(example_dataset)):
        print_msg = run_sample(
            target_device,
            hoptim,
            intermediate[index],
            example_dataset,
            index,
            kmetric,
            hand_region_assignment,
            hand_palm_vertex_mask,
            args.hand_closed_path,
        )
        pprint(print_msg)

    cprint("\nGEO DONE!", "cyan")


def main(args):
    intermediate = picr_stage(args)
    geo_stage(intermediate, args)


if __name__ == "__main__":
    # ==================== argument parser >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )

    # exp arguments
    parser.add_argument("--exp_keyword", type=str, default=None)

    # Dataset params
    parser.add_argument("--data_root", type=str, default="data", help="hodata root")
    parser.add_argument("--center_idx", default=9, type=int)
    parser.add_argument("--mano_root", default="assets/mano")
    parser.add_argument("--anchor_root", default="assets/anchor")

    # Model parameters
    parser.add_argument("--init_ckpt", type=str, required=True, default=None)
    parser.add_argument("--hg_stacks", type=int, default=2)
    parser.add_argument("--hg_blocks", type=int, default=1)
    parser.add_argument("--hg_classes", type=int, default=64)
    parser.add_argument(
        "--obj_scale_factor",
        type=float,
        default=0.0001,
        help="Multiplier for scale prediction",
    )
    parser.add_argument("--honet_resnet_version", choices=[18, 50], default=18)
    parser.add_argument(
        "--honet_mano_lambda_recov_joints3d",
        type=float,
        default=0.5,
        help="Weight for 3D vertices supervision in camera space",
    )
    parser.add_argument(
        "--honet_mano_lambda_recov_verts3d",
        type=float,
        default=0,
        help="Weight for 3D joints supervision, in camera space",
    )
    parser.add_argument(
        "--honet_mano_lambda_shape",
        type=float,
        default=5e-07,
        help="Weight for hand shape regularization",
    )
    parser.add_argument(
        "--honet_mano_lambda_pose_reg",
        type=float,
        default=5e-06,
        help="Weight for hand pose regularization",
    )
    parser.add_argument(
        "--honet_obj_lambda_recov_verts3d",
        type=float,
        default=0.5,
        help="Weight for object vertices supervision, in camera space",
    )
    parser.add_argument(
        "--honet_obj_lambda_recov_verts2d",
        type=float,
        default=0.0,
        help="Weight for object vertices supervision, in 2d UV space",
    )
    parser.add_argument(
        "--honet_obj_trans_factor",
        type=float,
        default=100,
        help="Multiplier for translation prediction",
    )
    parser.add_argument("--honet_mano_fhb_hand", action="store_true")

    # Training parameters
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers for multiprocessing")

    # Dump parameters
    parser.add_argument("--vertex_contact_thresh", type=float, default=0.7)

    # GEO
    parser.add_argument("--palm_path", type=str, default="assets/hand_palm_full.txt")
    parser.add_argument("--hand_closed_path", type=str, default="assets/closed_hand/hand_mesh_close.obj")
    parser.add_argument("--lambda_contact_loss", type=float, default=10.0)
    parser.add_argument("--lambda_repulsion_loss", type=float, default=1.6)
    parser.add_argument("--repulsion_query", type=float, default=0.020)
    parser.add_argument("--repulsion_threshold", type=float, default=0.050)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== setup environment & run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    args = parser.parse_args()

    main(args)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
