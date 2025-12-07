import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmdet3d.datasets import KittiDataset
from mmengine.dataset import default_collate
from mmcv.ops import points_in_boxes_cpu
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets import KittiDataset
from mmcv.transforms import Compose

# 1) Load config & model
cfg = Config.fromfile('/raid/cs21resch15003/mmdetection3d/configs/my_voxelnet/voxelnet_kitti-3d-3class_original.py')
model = init_model(cfg, '/raid/cs21resch15003/mmdetection3d/checkpoin_originalt/dual_attention_expt/epoch_150.pth', device='cuda:0')

# 2) Build the test dataset by hand
#    We use Compose to turn your cfg.test_pipeline (list of dicts) into a callable
test_pipeline = Compose(cfg.test_pipeline)

test_ds_cfg = cfg.test_dataloader.dataset
dataset = KittiDataset(
    data_root     = test_ds_cfg.data_root,
    ann_file      = test_ds_cfg.ann_file,
    pipeline      = test_pipeline,
    box_type_3d   = test_ds_cfg.box_type_3d,
    metainfo      = cfg.metainfo,
    modality      = test_ds_cfg.modality,
    test_mode     = True,
    data_prefix = test_ds_cfg.data_prefix, 
)

# 3) Wrap it in a DataLoader
loader = DataLoader(
    dataset,
    batch_size    = cfg.test_dataloader.batch_size,
    shuffle       = False,
    num_workers   = cfg.test_dataloader.num_workers,
    collate_fn    = default_collate,
    persistent_workers = cfg.test_dataloader.get('persistent_workers', False),
)

# 4) Iterate, run inference, and crop points
out_dir = 'cropped_pts'
os.makedirs(out_dir, exist_ok=True)

for idx, data in enumerate(loader):
    # a) Run the model
    result = inference_detector(model, data)[0]
    # b) Get raw points as numpy
    pts = data['inputs'][0]._data['points'].numpy()  # (N,4)
    # c) Get predicted boxes (M,7)
    box_tensor = result.pred_instances_3d.tensor.numpy()
    boxes = LiDARInstance3DBoxes(box_tensor).tensor  # still a tensor

    # d) Compute masks [M×N]
    #    points_in_boxes_cpu expects torch tensors: [1,N,3] and [1,M,7]
    masks = points_in_boxes_cpu(
      torch.from_numpy(pts[:,:3]).unsqueeze(0),
      boxes.unsqueeze(0)
    )[0].permute(1,0).numpy()

    # e) Save each box’s points
    for bi, mask in enumerate(masks):
        pts_in = pts[mask]
        if pts_in.size == 0:
            continue
        np.save(f'{out_dir}/sample{idx:04d}_box{bi:02d}.npy', pts_in)
        print(f'Saved {pts_in.shape[0]} points → sample{idx:04d}_box{bi:02d}.npy')
