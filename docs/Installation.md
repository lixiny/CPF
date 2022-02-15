# Getting started
```shell script
$ git clone --recursive https://github.com/lixiny/CPF.git
$ cd CPF
```
# Set up new environment:
```shell script
$ conda env create -f environment.yaml
$ conda activate cpf
```

# Download assets

* Download our [[assets.zip](https://drive.google.com/file/d/1IP7dJimk0G-rixfDprgYE0F8kquB6PWf/view?usp=sharing)]  and unzip it as an `assets` folder.

* Download the MANO model files from [official MANO website](https://mano.is.tue.mpg.de/) (you need to register first), and put `MANO_RIGHT.pkl` file into `assets/mano`

Now your `assets/` folder should look like this:
```
assets
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
# Download datasets
## First-Person Hand Action Benchmark (fhb)  

* Download and unzip the First-Person Hand Action Benchmark dataset following the [official instructions](https://github.com/guiggh/hand_pose_action) to the `data/fhbhands/` folder.  
* Resize the original FHB images based on the [handobjectconsist/reduce_fphab.py](https://github.com/hassony2/handobjectconsist/blob/master/reduce_fphab.py).  ```python reduce_fphab.py```

* Download our [[fhbhands_supp.zip](https://drive.google.com/file/d/1hY_gyrZD_RU3nxI90oJZ6tNkwxKYhUGs/view?usp=sharing)] and unzip it as `data/fhbhands_supp`:

* Download our [[fhbhands_example.zip](https://drive.google.com/file/d/14wxN23RmVCSphHIV00qk-ht00yfdu_Hu/view?usp=sharing)] and unzip it as `data/fhbhands_example`.
This `fhbhands_example/` contains 10 samples that are designed to demonstrate our pipeline.

Currently, your `data` folder should look like this:

```
data
├── fhbhands/
│   ├── action_object_info.txt
│   ├── action_sequences_normalized/
│   ├── change_log.txt
│   ├── data_split_action_recognition.txt
│   ├── file_system.jpg
│   ├── Hand_pose_annotation_v1/
│   ├── Object_6D_pose_annotation_v1_1/
│   ├── Object_models/
│   ├── Subjects_info/
│   ├── Video_files/
│   └── Video_files_480/
├── fhbhands_supp/
│   ├── Object_models/
│   └── Object_models_binvox/
├── fhbhands_example/
│   ├── annotations/
│   ├── images/
│   ├── object_models/
│   └── sample_list.txt
```

##  HO3D
* Download the [HO3D](https://www.tugraz.at/index.php?id=40231) dataset (version 2) following the [official instructions](https://github.com/shreyashampali/ho3d?)  and unzip it in the `data/HO3D` folder.
* Download the YCB object models from [here](https://rse-lab.cs.washington.edu/projects/posecnn/), choose (The YCB-Video 3D Models ~ 367M) and unzip it in `data/YCB_models`.
* Download our [[YCB_models_supp.zip](https://drive.google.com/file/d/1daSKseF-PrVLwd4wIcF2PLtAjYBF2XH1/view?usp=sharing)] and unzip it as `data/YCB_models_supp`

Finally, the `data` folder should have a structure like:

```
data
├── fhbhands/
├── fhbhands_supp/
├── fhbhands_example/
├── HO3D/
│   ├── evaluation/
│   ├── evaluation.txt
│   ├── train/
│   └── train.txt
├── YCB_models/
│   ├── 002_master_chef_can/
│   └── ...
├── YCB_models_supp/
│   ├── 002_master_chef_can/
│   └── ...
```

# Download pretrained models
download our pre-trained [[CPF_checkpoints.zip](https://drive.google.com/file/d/1JWu5TSTTIWvNrTZmmjhTEm4xGqv_cMhd/view?usp=sharing)], unzip it as `CPF_checkpoints`
```
CPF_checkpoints
├── honet/
│   ├── fhb/
│   ├── ho3dofficial/
│   └── ho3dv1/
├── picr/
│   ├── fhb/
│   ├── ho3dofficial/
│   └── ho3dv1/
```

---
Notice: for researchers from China, you can alternatively download our preprocessed files at this mirror: [百度盘](https://pan.baidu.com/s/1x6SvGpNZqWlLA-cBZvv8Og) (`2tqv`)