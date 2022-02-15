from hocontact.models.picr import PicrHourglassPointNet
from hocontact.hodatasets.hodata import HOdata, ho_collate
from hocontact.utils.logger import logger
from hocontact.utils.lossutils import update_loss
from hocontact.utils import eval as evalutils
from hocontact.utils import dump as dumputils
from hocontact.utils import ioutils
from hocontact.utils.eval.disteval import merge_evaluator
from hocontact.utils.eval.summarize import summarize_evaluator_picr
from hocontact.utils.dump.distdump import summarize_dumper_list
from torch.utils.data import ConcatDataset, DataLoader
import random
import argparse
from datetime import datetime
import os
import pickle
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import shutil


def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


plt.switch_backend("agg")


def dump_main(args):
    # ==================== setup things before distributed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device_count = torch.cuda.device_count()
    logger.warn("\nUSING {} GPUs".format(torch.cuda.device_count()))

    set_all_seeds(args.manual_seed)

    if args.exp_keyword is None:
        now = datetime.now()
        exp_keyword = f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}"
    else:
        exp_keyword = args.exp_keyword

    dat_str = "_".join(args.train_datasets)
    split_str = "_".join(args.train_splits)
    exp_id = f"checkpoints/picr_dump/{dat_str}_{split_str}_mini{args.mini_factor}/bs{args.batch_size}"
    exp_id = f"{exp_id}_brot"
    exp_id = f"{exp_id}_dump"
    exp_id = f"{exp_id}/{exp_keyword}"

    logger.initialize(exp_id, "dump")
    ioutils.print_args(args)
    ioutils.save_args(args, exp_id, "opt")

    logger.info(f"Saving experiment logs, models, and training curves and images to {exp_id}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== distributed run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    torch.multiprocessing.spawn(dump_main_worker, args=(device_count, exp_id, args), nprocs=device_count, join=True)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def dump_main_worker_setup(rank, world_size):
    # initialize the progress group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def dump_main_worker_cleanup():
    torch.distributed.destroy_process_group()


def dump_main_worker(rank, world_size, exp_id, args):
    # ====================  setup distributed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info(f"====== worker {rank} of {world_size} initiate >>>>>>", "cyan")
    dump_main_worker_setup(rank, world_size)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== Creating Datasets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    assert len(args.train_datasets) == len(
        args.train_splits
    ), f"train dataset and split not match, got {args.train_datasets} and {args.train_splits}"

    datasets = []
    for data_name, data_split in zip(args.train_datasets, args.train_splits):
        train_dataset = HOdata.get_dataset(
            dataset=data_name,
            data_root=args.data_root,
            data_split=data_split,
            split_mode=args.split_mode,
            use_cache=args.use_cache,
            mini_factor=args.mini_factor,
            center_idx=args.center_idx,
            enable_contact=args.enable_contact,
            filter_no_contact=True,
            filter_thresh=args.filter_thresh,
            like_v1=(args.version == 1),
            block_rot=True,
            synt_factor=0,
        )
        datasets.append(train_dataset)
        if rank == 0:
            ioutils.print_query(train_dataset.queries, desp=f"training_set_{data_name}_queries")

    train_dataset = ConcatDataset(datasets)
    train_dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=ho_collate,
        pin_memory=True,
        sampler=train_dist_sampler,
    )

    val_dataset = HOdata.get_dataset(
        dataset=args.val_dataset,
        data_root=args.data_root,
        data_split=args.val_split,
        split_mode=args.split_mode,
        use_cache=args.use_cache,
        mini_factor=args.mini_factor,
        center_idx=args.center_idx,
        enable_contact=args.enable_contact,
        filter_no_contact=args.test_dump,
        filter_thresh=args.filter_thresh,
        like_v1=(args.version == 1),
        block_rot=True,
        synt_factor=0,
    )
    val_dist_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn=ho_collate,
        pin_memory=True,
        sampler=val_dist_sampler,
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
        honet_mano_fhb_hand="fhb" in args.train_datasets,
    )
    _model = _model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(_model, device_ids=[rank])

    # check init_ckpt option
    if args.init_ckpt is None:
        logger.error("no initializing checkpoint provided. abort!", "red")
        exit()
    map_location = f"cuda:{rank}"
    _ = ioutils.reload_checkpoint(
        model,
        resume_path=args.init_ckpt,
        as_parallel=True,
        map_location=map_location,
        reload_honet_checkpoints=args.reload_honet_checkpoints,
    )
    # only weights is reloaded, others are dropped

    # ====== print model size information
    if rank == 0:
        logger.info(f"Model total size == {ioutils.param_size(model.module)} MB")
        logger.info(f"  |  HONet total size == {ioutils.param_size(model.module.ho_net)} MB")
        logger.info(f"  |  BaseNet total size == {ioutils.param_size(model.module.base_net)} MB")
        logger.info(f"  \\  ContactHead total size == {ioutils.param_size(model.module.contact_head)} MB")
        logger.info(f"    |  EncodeModule total size == {ioutils.param_size(model.module.contact_head.encoder)} MB")
        decode_vertex_contact_size = ioutils.param_size(model.module.contact_head.vertex_contact_decoder)
        decode_contact_region_size = ioutils.param_size(model.module.contact_head.contact_region_decoder)
        decode_anchor_elasti_size = ioutils.param_size(model.module.contact_head.anchor_elasti_decoder)
        logger.info(f"    |  DecodeModule_VertexContact total size == {decode_vertex_contact_size} MB")
        logger.info(f"    |  DecodeModule_ContactRegion total size == {decode_contact_region_size} MB")
        logger.info(f"    |  DecodeModule_AnchorElasti total size == {decode_anchor_elasti_size} MB")

    train_dist_sampler.set_epoch(0)
    val_dist_sampler.set_epoch(0)

    # ==================== dumping train >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # job before epoch
    tmp_dir = dumping_epoch_before(rank, exp_id, 0, train=True)

    # epoch pass
    dumping_epoch_pass(
        rank,
        "train",
        train_loader,
        model,
        epoch=0,
        use_eval=args.dump_eval,
        use_dump=args.dump,
        dump_prefix=args.dump_prefix,
        tmp_dir=tmp_dir,
        vertex_contact_thresh=args.vertex_contact_thresh,
    )

    # job after epoch
    dumping_epoch_after(
        rank,
        exp_id,
        0,
        train=True,
        use_eval=args.dump_eval,
        use_dump=args.dump,
        tmp_dir=tmp_dir,
        target_count=len(train_dataset),
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== dumping val >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # job before epoch
    val_tmp_dir = dumping_epoch_before(rank, exp_id, 0, train=False)

    # epoch pass
    dumping_epoch_pass(
        rank,
        "val",
        val_loader,
        model,
        epoch=0,
        use_eval=args.dump_eval,
        use_dump=args.dump,
        dump_prefix=args.dump_prefix,
        tmp_dir=val_tmp_dir,
        vertex_contact_thresh=args.vertex_contact_thresh,
    )

    # job after epoch
    dumping_epoch_after(
        rank,
        exp_id,
        0,
        train=False,
        use_eval=args.dump_eval,
        use_dump=args.dump,
        tmp_dir=val_tmp_dir,
        target_count=len(val_dataset),
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if rank == 0:
        logger.warn("\nDONE!")

    dump_main_worker_cleanup()


def dumping_epoch_before(rank, exp_id, epoch_idx, train):
    train_word = "train" if train else "val"
    exp_id_word = str(exp_id).replace("/", "__")
    tmp_dir = os.path.join("/tmp", f"{exp_id_word}__{train_word}__epoch_{epoch_idx}")
    # create a tempdirectory, only rank 0 does it
    if rank == 0:
        if os.path.exists(tmp_dir):
            logger.warn(f"tmp_dir exists, removed: {tmp_dir}", "red")
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir, 0o700)
    else:
        pass
    # only after rank 0 make the path, other ranks can proceed
    torch.distributed.barrier()
    return tmp_dir


def dumping_epoch_pass(
    rank,
    prefix,
    loader,
    model,
    epoch=0,
    use_eval=False,
    use_dump=False,
    dump_prefix=None,
    tmp_dir=None,
    vertex_contact_thresh=0.5,
):
    if rank == 0:
        logger.warn(f"{prefix.capitalize()} Epoch {epoch}", "blue")
        logger.warn(f"Showing Information about Node {rank}", "blue")

    # model will always be in eval mode
    model.eval()
    if use_eval:
        # create evaluator
        evaluator = evalutils.Evaluator()
    if use_dump:
        # create dumper
        dumper = dumputils.PicrDumper(dump_prefix, "assets/anchor")

    # ==================== Forward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # loop over dataset
    if rank == 0:
        loader = tqdm(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # model
            ls_results = model(batch, rank=rank)

            if use_eval:
                # feed evaluator with the output of the last stack
                evaluator.feed_loss_meters(batch, ls_results[-1])
                evaluator.feed_eval_meters(batch, ls_results[-1])

                # dump evaluator, if tmp_dir is not None
                if tmp_dir is not None:
                    tmp_evaluator_file = os.path.join(tmp_dir, f"evaluator_{rank}.pkl")
                    with open(tmp_evaluator_file, "wb") as fstream:
                        pickle.dump(evaluator, fstream)

            if use_dump:
                # feed dumper with the output of the last stack
                dumper.feed_and_dump(batch, ls_results[-1], vertex_contact_thresh)

                # dump dumper, if tmp_dir is not None
                if tmp_dir is not None:
                    tmp_dumper_file = os.path.join(tmp_dir, f"dumper_{rank}.pkl")
                    with open(tmp_dumper_file, "wb") as fstream:
                        pickle.dump(dumper, fstream)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== Postprocess >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # if use_eval:
    #     eval_msg = summarize_evaluator_picr(evaluator)
    #     logger.warn(eval_msg, color="yellow")

    if use_dump:
        dump_msg = dumper.info()
        dump_msg = f"Rank {rank}: {dump_msg}"
        logger.warn(dump_msg, color="cyan")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def dumping_epoch_after(
    rank, exp_id, epoch_idx, train, use_eval=False, use_dump=False, tmp_dir=None, target_count=None
):
    # wait for all ranks done there epoch
    torch.distributed.barrier()
    if rank == 0:
        assert tmp_dir is not None
        tmp_file_list = list(os.listdir(tmp_dir))

        if use_eval:
            evaluator_list = []
        if use_dump:
            dumper_list = []
        for pkl_name in tmp_file_list:
            if use_eval and pkl_name.startswith("evaluator"):
                pkl_path = os.path.join(tmp_dir, pkl_name)
                with open(pkl_path, "rb") as fstream:
                    tmp_evaluator = pickle.load(fstream)
                evaluator_list.append(tmp_evaluator)
            if use_dump and pkl_name.startswith("dumper"):
                pkl_path = os.path.join(tmp_dir, pkl_name)
                with open(pkl_path, "rb") as fstream:
                    tmp_dumper = pickle.load(fstream)
                dumper_list.append(tmp_dumper)

        if use_eval:
            evaluator = merge_evaluator(evaluator_list)
            save_dict = summarize_evaluator_picr(evaluator, exp_id, epoch_idx, train=train)
            logger.warn(f"    {save_dict}", color="yellow")
        if use_dump:
            dump_combined_msg, dump_combined_msg_color = summarize_dumper_list(dumper_list, target_count)
            logger.warn(dump_combined_msg, color=dump_combined_msg_color)

        shutil.rmtree(tmp_dir)
    else:
        pass

    # wait for rank 0 to complete
    torch.distributed.barrier()
    return


if __name__ == "__main__":
    # ==================== argument parser >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument("--gpu", type=str, default=None, help="override enviroment var CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dist_master_addr", type=str, default="localhost")
    parser.add_argument("--dist_master_port", type=str, default="12355")

    # exp arguments
    parser.add_argument("--exp_keyword", type=str, default=None)

    # Dataset params
    parser.add_argument("--data_root", type=str, default="data", help="hodata root")
    parser.add_argument(
        "--train_datasets",
        choices=["ho3d", "ho3dsynt", "fhb", "fhbsynt"],
        default=["fhb"],
        nargs="+",
    )
    parser.add_argument(
        "--val_dataset",
        choices=["ho3d", "fhb"],
        default="fhb",
    )
    parser.add_argument("--train_splits", default=["train"], nargs="+")
    parser.add_argument("--val_split", default="test", choices=["test", "train", "val", "trainval"])
    parser.add_argument(
        "--split_mode",
        default="actions",
        choices=["objects", "actions", "official"],
    )
    parser.add_argument(
        "--mini_factor", type=float, default=1.0, help="Work on fraction of the datase for debugging purposes"
    )
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--enable_contact", action="store_true", help="Enable contact info", default=True)
    parser.add_argument("--version", default=1, type=int, help="Version of HO3D dataset to use")
    parser.add_argument("--center_idx", default=9, type=int)

    # Model parameters
    parser.add_argument("--init_ckpt", type=str, required=True, default=None)
    parser.add_argument("--reload_honet_checkpoints", type=str, default=None)
    parser.add_argument("--hg_stacks", type=int, default=2)
    parser.add_argument("--hg_blocks", type=int, default=1)
    parser.add_argument("--hg_classes", type=int, default=64)
    parser.add_argument("--obj_scale_factor", type=float, default=0.0001, help="Multiplier for scale prediction")
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
        "--honet_mano_lambda_shape", type=float, default=5e-07, help="Weight for hand shape regularization"
    )
    parser.add_argument(
        "--honet_mano_lambda_pose_reg", type=float, default=5e-06, help="Weight for hand pose regularization"
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
        "--honet_obj_trans_factor", type=float, default=100, help="Multiplier for translation prediction"
    )

    # Training parameters
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers for multiprocessing")

    # Loss parameters
    parser.add_argument(
        "--contact_lambda_vertex_contact",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--contact_lambda_contact_region",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--contact_lambda_anchor_elasti",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--region_focal_loss_alpha",
        type=str,
        choices=["fhb", "ho3d"],
        default=None,
    )
    parser.add_argument("--test_dump", action="store_false")

    # Dump parameters
    parser.add_argument("--dump_prefix", type=str, default="common/picr")
    parser.add_argument("--dump_eval", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--vertex_contact_thresh", type=float, default=0.9)
    parser.add_argument("--filter_thresh", type=float, default=1.0)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== setup environment & run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["MASTER_ADDR"] = args.dist_master_addr
    os.environ["MASTER_PORT"] = args.dist_master_port

    dump_main(args)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
