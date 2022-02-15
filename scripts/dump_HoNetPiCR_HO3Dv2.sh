echo "Using GPU_ID: ${GPU_ID}"

python scripts/dump_picr_res.py  \
    --gpu ${GPU_ID} \
    --dist_master_addr localhost \
    --dist_master_port 12356 \
    --exp_keyword ho3dofficial \
    --train_datasets ho3d \
    --train_splits val \
    --val_dataset ho3d \
    --val_split test \
    --split_mode official \
    --batch_size 4 \
    --dump_eval \
    --dump \
    --test_dump \
    --vertex_contact_thresh 0.8 \
    --filter_thresh 5.0 \
    --dump_prefix common/picr_ho3dofficial \
    --init_ckpt CPF_checkpoints/picr/ho3dofficial/checkpoint_300.pth.tar