echo "Using GPU_ID: ${GPU_ID}"

python scripts/fit_geo.py \
    --gpu ${GPU_ID} \
    --n_workers 24 \
    --data_path common/picr_ho3dofficial/HO3D/test_official_mf1_likev1_fct\(x\)_ec/  \
    --lr 1e-2 \
    --n_iter 500 \
    --hodata_no_use_cache \
    --lambda_contact_loss 10.0 \
    --lambda_repulsion_loss 2.0 \
    --repulsion_query 0.030 \
    --repulsion_threshold 0.080 \
    --mode hand_obj