## High-Order Structural Relation Distillation Networks From LiDAR to Monocular Image 3D Detectors

## 1.Framework Overview
![image](/docs/framework.png)
checkpoints(deeplabV3 and second):https://drive.google.com/drive/folders/13Bd9yxWiBzFqi_9Lekj4WPSVQFW5SyYQ?usp=drive_link
## 2.Pretrained Models
If you would like to use some pretrained models, download them and put them into ../checkpoints
```
OpenPCDet
├── checkpoints
|   ├── second_teacher.pth
|   ├── ···
├── data
├── pcdet
├── tools
```
## 3.Requirements

All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv) or [`spconv v2.x`](https://github.com/traveller59/spconv)

a. Clone this repository.
```shell
git clone https://github.com/xiaoxupossible/HSRDN.git
```

b. Install the dependent libraries as follows:

[comment]: <> (* Install the dependent python libraries: )

[comment]: <> (```)

[comment]: <> (pip install -r requirements.txt )

[comment]: <> (```)

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 
    * You could also install latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
## 4.KITTI Dataset

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to use the depth maps for trainval set, download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI trainval set
* Download the [KITTI Raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and put in into data/kitti/raw/KITTI_Raw
* (optional) If you want to use the [sparse depth maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) for KITTI Raw, download it and put it into data/kitti/raw/depth_sparse

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
│   │   │── raw
|   |   |   |——calib & KITTI_Raw & (optional: depth_sparse)
├── pcdet
├── tools
```
* Generate the data infos by running the following command (kitti train, kitti val, kitti test): 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
* Generate the data infos by running the following command (kitti train + eigen clean, unlabeled):
```python 
python -m pcdet.datasets.kitti.kitti_dataset_cmkd create_kitti_infos_unlabel tools/cfgs/dataset_configs/kitti_dataset.yaml
```
## 5.Evaluate the pretrained models
* Test with a pretrained model: 
```python
python test_cmkd.py --cfg ${CONFIG_FILE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```python
python test_cmkd.py --cfg ${CONFIG_FILE} --ckpt_dir ${CKPT_DIR}  --eval_all
```

* To test with multiple GPUs:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 test_cmkd.py --launcher pytorch --cfg ${CONFIG_FILE} --tcp_port 16677 --ckpt ${CKPT}
```

### 6.Train a model

* Train with a single GPU:
```python
python train_cmkd.py --cfg_file ${CONFIG_FILE} --pretrained_lidar_model ${TEACHER_MODEL_PATH}
```

* Train with multiple GPUs or multiple machines
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ${CONFIG_FILE} --tcp_port 16677 --pretrained_lidar_model ${TEACHER_MODEL_PATH}
```
## 7.Citation
If you find our work useful in your research, please consider citing:

```latex
@article{yan2023high,
  title={High-order Structural Relation Distillation Networks from LiDAR to Monocular Image 3D Detectors},
  author={Yan, Weiqing and Xu, Long and Liu, Hao and Tang, Chang and Zhou, Wujie},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  publisher={IEEE}
}
```

If you have any problems, contact me via ydxulong@163.com.
