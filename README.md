# Multi-Target 3D Object Detection Using LiDAR – KITTI Dataset

This repository contains the implementation of a point-cloud-based multi-target 3D object detection system developed using LiDAR sensor data. The proposed method is designed for autonomous-driving scenarios and focuses on achieving accurate, stable, and efficient detection performance across multiple object classes.

---

##  Overview

Traditional 3D object detection pipelines either project point clouds into 2D views or discretize them using coarse voxel grids, which often leads to information loss. Our method addresses these challenges by introducing:

- A **custom point encoding technique** that preserves geometric structure.
- A **unified classification head** that jointly handles positive and negative samples, improving training stability.
- **Single-network multi-class detection**, reducing computational complexity.
- **Focal Loss** to mitigate severe class imbalance—especially important for pedestrians and cyclists.
- An end-to-end pipeline evaluated on the **KITTI 3D Object Detection Benchmark**.

---
Sincere thanks for the great open-source architectures mmcv, mmdet and mmdet3d, which helps me to learn 3D detetion and implement this repo.

## 📂 Repository Structure

/configs/ # training configuration files
/data/ # scripts or instructions for dataset preparation
/models/ # model definitions and network modules
/utils/ # helper functions, encoding, evaluation scripts
/train.py # training entry point
/test.py # testing / inference script
/README.md # project documentation

**Install**

pip install -r requirements.txt

python setup.py install

pip install .


**[Dataset]**
Please use the preprocessed dataset from KITTI official portal
KITTI/
├── training/
│ ├── image_2/
│ ├── velodyne/
│ ├── calib/
│ └── label_2/
└── testing/
├── image_2/
├── velodyne/
└── calib/

**Evaluation**

python test_pc.py --checkpoint path_to_model.pth

python read_pkl.py pred_instances_3d.pkl


##  Repository Contents

Your uploaded repository consists of the following key components:

### ** Main Python Scripts**
| File | Description |
|------|-------------|
| `voxelnet_kitti-3d-3class.py` | Main model implementation for KITTI (new method). |
| `point_clouds.py` | Utility functions for point cloud processing. |
| `read_pkl.py` | Reads `.pkl` prediction files. |
| `pred_instances_3d.pkl` | Sample prediction output. |
| `test_pc.py` | Testing script for model inference. |
| `config_loss_modif_fp_test.py` | Modified configuration for loss (forward pass). |
| `config_loss_modif_test.py` | Modified configuration for loss (test mode). |
| `config_test.py` | General testing configuration. |
| `setup.py`, `setup.cfg` | Build and package configuration. |

---

### ** ZIP Archives Provided**
| ZIP File | Contents |
|---------|----------|
| `codes.zip` | The complete codebase (legacy version). |
| `mmdet3d.zip` | MMDetection3D framework components. |
| `modifiedLibs.zip` | Custom library modifications. |
| `tools.zip` | Helper scripts and utilities. |
| `docker.zip` | Docker environment setup. |
| `requirements.zip` | Required packages (text form inside ZIP). |

>  **Note:** Users should unzip these folders after cloning for full functionality.



   ** Acknowledements**
Thanks for the open source code mmcv, mmdet and mmdet3d.
