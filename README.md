# Contact Potential Field
[[`Project Page`](https://lixiny.github.io/CPF)]


This repo contains model, demo, and test codes of our paper:

> [CPF: Learning a **C**ontact **P**otential **F**ield to Model the Hand-Object Interaction](https://arxiv.org/abs/2012.00924)    
> Lixin Yang, Xinyu Zhan, Kailin Li, Wenqiang Xu, Jiefeng Li, Cewu Lu    
> ICCV 2021    

<p align="center">
    <img src="teaser.png", width="100%">
</p>


# Guide to the Demo
## 1. Get our code:
```shell script
$ git clone --recursive https://github.com/lixiny/CPF.git
$ cd CPF
```
## 2. Set up your new environment:

```shell script
$ conda env create -f environment.yaml
$ conda activate cpf
```

## 3. Download asset files
Down load our [[assets.zip](https://drive.google.com/file/d/1IP7dJimk0G-rixfDprgYE0F8kquB6PWf/view?usp=sharing)]  and unzip it as an `assets/` folder.


Download the MANO model files from [official MANO website](https://mano.is.tue.mpg.de/), and put it into `assets/mano/`.
We currently only use the `MANO_RIGHT.pkl`

Now your `assets/` folder should look like this:
```
assets/
├── anchor/
│   ├── anchor_mapping_path.pkl
│   ├── anchor_weight.txt
│   ├── face_vertex_idx.txt
│   └── merged_vertex_assignment.txt
├── closed_hand/
│   └── hand_mesh_close.obj
├── fhbhands_fits/
│   ├── Subject_1/
│   │   ├── ...
│   ├── Subject_2/
|   ├── ...
├── hand_palm_full.txt
└── mano/
    ├── fhb_skel_centeridx9.pkl
    ├── info.txt
    ├── LICENSE.txt
    └── MANO_RIGHT.pkl
```
## 4. Download dataset
### First-Person Hand Action Benchmark (fhb)

Download and unzip the First-Person Hand Action Benchmark dataset following the [official instructions](https://github.com/guiggh/hand_pose_action) to the `data/fhbhands/` folder

If everything is correct, your `data/fhbhands/` should look like this:
```
.
├── action_object_info.txt
├── action_sequences_normalized/
├── change_log.txt
├── data_split_action_recognition.txt
├── file_system.jpg
├── Hand_pose_annotation_v1/
├── Object_6D_pose_annotation_v1_1/
├── Object_models/
├── Subjects_info/
├── Video_files/
├── Video_files_480/ # Optionally
```
Optionally, resize the images (speeds up training !) based on the [handobjectconsist/reduce_fphab.py](https://github.com/hassony2/handobjectconsist/blob/master/reduce_fphab.py).
```shell
$ python reduce_fphab.py
```
Download our [[fhbhands_supp.zip](https://drive.google.com/file/d/1hY_gyrZD_RU3nxI90oJZ6tNkwxKYhUGs/view?usp=sharing)] and unzip it as `data/fhbhands_supp`:

Download our [[fhbhands_example.zip](https://drive.google.com/file/d/14wxN23RmVCSphHIV00qk-ht00yfdu_Hu/view?usp=sharing)] and unzip it as `data/fhbhands_example`.
This `fhbhands_example/` contains 10 samples that are designed to demonstrate our pipeline.

Currently, your `data/` folder should look like this:

```
data/
├── fhbhands/
├── fhbhands_supp/
│   ├── Object_models/
│   └── Object_models_binvox/
├── fhbhands_example/
│   ├── annotations/
│   ├── images/
│   ├── object_models/
│   └── sample_list.txt
```

### HO3D
Download and unzip the [HO3D](https://www.tugraz.at/index.php?id=40231) dataset following the [official instructions](https://github.com/shreyashampali/ho3d?) to the `data/HO3D` folder.
if everything is correct, the HO3D & YCB folder in your `data/` folder should look like this:
```
data/
├── HO3D/
│   ├── evaluation/
│   ├── evaluation.txt
│   ├── train/
│   └── train.txt
├── YCB_models/
│   ├── 002_master_chef_can/
│   ├── ...
...
```

Download our [[YCB_models_supp.zip](https://drive.google.com/file/d/1daSKseF-PrVLwd4wIcF2PLtAjYBF2XH1/view?usp=sharing)] and place it at `data/YCB_models_supp`

Finally, the `data/` folder should have a structure like:
```
data/
├── fhbhands/
├── fhbhands_supp/
├── fhbhands_example/
├── HO3D/
├── YCB_models/
├── YCB_models_supp/
```

## 5. Download pre-trained checkpoints
download our pre-trained [[CPF_checkpoints.zip](https://drive.google.com/file/d/1JWu5TSTTIWvNrTZmmjhTEm4xGqv_cMhd/view?usp=sharing)], unzip it as `CPF_checkpoints/` folder:
```
CPF_checkpoints/
├── honet/
│   ├── fhb/
│   ├── ho3dofficial/
│   └── ho3dv1/
├── picr/
│   ├── fhb/
│   ├── ho3dofficial/
│   └── ho3dv1/
```

## 6. Launch visualization

> Replace the `${GPU_ID}` with a list of integers that indicates the GPU id.   
> eg: `--gpu 0,1`; `--gpu 0`; `--gpu 0,1,2,3`

We create a `FHBExample` dataset in `hocontact/hodatasets/fhb_example.py` that only contains 10 samples to demonstrate our pipeline.
Notice: this demo requires active screen for visualizing. Press `q` in the "runtime hand" window to start fitting.

```shell
# recommend 1 GPU
$ python scripts/run_demo.py \
    --gpu ${GPU_ID} \
    --init_ckpt CPF_checkpoints/picr/fhb/checkpoint_200.pth.tar \
    --honet_mano_fhb_hand
```

<p align="center">
    <img src="teaser_fitting.gif", width="60%">
</p>

## 7. Test on full dataset (FHB, HO3D v1/v2)

We provide shell srcipts to test on the full dataset to approximately reproduce our results.

### FHB
dump the results of HoNet and PiCR:
```shell
# recommend 2 GPUs
$ python scripts/dump_picr_res.py \
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
```
and reload the GeO optimizer:
```shell
# setting 1: hand-only
# CUDA_VISIBLE_DEVICES=0,1,2,3
# recommend 4 GPUs
$ python scripts/eval_geo.py \
    --gpu ${GPU_ID} \
    --n_workers 16 \
    --data_path common/picr/fhbhands/test_actions_mf1.0_rf0.25_fct5.0_ec \
    --mode hand

# setting 2: hand-obj
$ python scripts/eval_geo.py \
    --gpu ${GPU_ID} \
    --n_workers 16 \
    --data_path common/picr/fhbhands/test_actions_mf1.0_rf0.25_fct5.0_ec \
    --mode hand_obj \
    --compensate_tsl
```
### HO3Dv1
dump:
```shell
# recomment 2 GPUs
$ python scripts/dump_picr_res.py  \
    --gpu ${GPU_ID} \
    --dist_master_addr localhost \
    --dist_master_port 12356 \
    --exp_keyword ho3dv1 \
    --train_datasets ho3d \
    --train_splits train \
    --val_dataset ho3d \
    --val_split test \
    --split_mode objects \
    --batch_size 4 \
    --dump_eval \
    --dump \
    --vertex_contact_thresh 0.8 \
    --filter_thresh 5.0 \
    --dump_prefix common/picr_ho3dv1 \
    --init_ckpt CPF_checkpoints/picr/ho3dv1/checkpoint_300.pth.tar
```
and reload optimizer:
```shell
# hand-only
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# recommend 8 GPUs
$ python scripts/eval_geo.py \
    --gpu ${GPU_ID}
    --n_workers 24 \
    --data_path common/picr_ho3dv1/HO3D/test_objects_mf1_likev1_fct5.0_ec/ \
    --lr 1e-2 \
    --n_iter 500 \
    --hodata_no_use_cache \
    --lambda_contact_loss 10.0 \
    --lambda_repulsion_loss 4.0 \
    --repulsion_query 0.030 \
    --repulsion_threshold 0.080 \
    --mode hand

# hand-obj
# recommend 8 GPUs
$ python scripts/eval_geo.py \
    --gpu ${GPU_ID} \
    --n_workers 24 \
    --data_path common/picr_ho3dv1/HO3D/test_objects_mf1_likev1_fct5.0_ec/ \
    --lr 1e-2 \
    --n_iter 500  \
    --hodata_no_use_cache \
    --lambda_contact_loss 10.0 \
    --lambda_repulsion_loss 6.0 \
    --repulsion_query 0.030 \
    --repulsion_threshold 0.080 \
    --mode hand_obj

```
### HO3Dofficial
dump:
```shell
# recommend 2 GPUs
$ python scripts/dump_picr_res.py  \
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
```
and reload optimizer:
```shell
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# recommend 8 GPUs
$ python scripts/eval_geo.py \
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
```



## Results

Testing on the full dataset may take a while ( 0.5 ~ 1.5 day ), thus we also provide our test results at [fitting_res.txt](https://github.com/lixiny/CPF/blob/main/fitting_res.txt).

## Anatomically Constrained A-MANO

We provide pytorch implementation of our A-MANO in [lixiny/manopth](https://github.com/lixiny/manopth), which is modified from the original [hassony2/manopth](https://github.com/hassony2/manopth). Thank [Yana Hasson](https://hassony2.github.io/) for providing the code.



## Citation
If you find this work helpful, please consider citing us:
```
@inproceedings{yang2021cpf,
    title={{CPF}: Learning a Contact Potential Field to Model the Hand-Object Interaction},
    author={Yang, Lixin and Zhan, Xinyu and Li, Kailin and Xu, Wenqiang and Li, Jiefeng and Lu, Cewu},
    booktitle={ICCV},
    year={2021}
}
```











