echo "Using GPU_ID: ${GPU_ID}"

python scripts/fit_geo.py \
    --gpu ${GPU_ID} \
    --n_workers 16 \
    --data_path common/picr/fhbhands/test_actions_mf1.0_rf0.25_fct5.0_ec \
    --mode hand