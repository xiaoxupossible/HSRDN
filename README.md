## High-Order Structural Relation Distillation Networks From LiDAR to Monocular Image 3D Detectors

## Framework Overview
![image](/docs/framework.png)
checkpoints(deeplabV3 and second):https://drive.google.com/drive/folders/13Bd9yxWiBzFqi_9Lekj4WPSVQFW5SyYQ?usp=drive_link
## Pretrained Models
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
## Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## Datasets

The Synthetic3d, Prokaryotic, and MNIST-USPS datasets are placed in "data" folder. The others dataset could be downloaded from [cloud](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ). key: data

## To reproduce our results with SECOND teacher, use
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ../tools/cfgs/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_bev.yaml --tcp_port 16677 --pretrained_lidar_model ../checkpoints/second_teacher.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_cmkd.py --launcher pytorch --cfg ../tools/cfgs/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_V2.yaml  --tcp_port 16677 --pretrained_lidar_model ../checkpoints/second_teacher.pth
--pretrained_img_model ../output/kitti_models/CMKD/cmkd_kitti_eigen_R50_scd_bev/default/ckpt/checkpoint_epoch_30.pth
```
## Citation
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
