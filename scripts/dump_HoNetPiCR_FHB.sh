echo "Using GPU_ID: ${GPU_ID}"

python ./scripts/dump_picr_res.py \
    --gpu ${GPU_ID} \
    --dist_master_addr localhost \
    --dist_master_port 12355 \
    --exp_keyword fhb \
    --train_datasets fhb \
    --train_splits train \
    --val_dataset fhb \
    --val_split test \
    --split_mode actions \
    --batch_size 8 \
    --dump_eval \
    --dump \
    --vertex_contact_thresh 0.8 \
    --filter_thresh 5.0 \
    --dump_prefix common/picr \
    --init_ckpt CPF_checkpoints/picr/fhb/checkpoint_200.pth.tar