import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmdet3d.apis import init_model, inference_detector

# ──────────────────────────────── CONFIG ────────────────────────────────
cfg     = '/raid/cs21resch15003/mmdetection3d/configs/my_voxelnet/voxelnet_kitti-3d-3class_original.py'
ckpt    = '/raid/cs21resch15003/mmdetection3d/checkpoin_originalt/dual_attention_expt/epoch_150.pth'
binfile = '/raid/cs21resch15003/mmdetection3d/data/kitti/testing/velodyne/000000.bin'
out_png = '/raid/cs21resch15003/mmdetection3d/vis_results/000000_pc.png'
# ─────────────────────────────────────────────────────────────────────────

# 1) Load model and run inference on a single LiDAR scan
model = init_model(cfg, ckpt, device='cuda:0')
result, _ = inference_detector(model, binfile)

# 2) Manually load raw LiDAR points (KITTI .bin format: float32 x,y,z,intensity)
pc   = np.fromfile(binfile, dtype=np.float32).reshape(-1, 4)
pts  = pc[:, :3]  # (N,3)

# 3) Grab the predicted 3D boxes
pred_instances = result.pred_instances_3d
# .bboxes_3d is a DepthInstance3DBoxes; .tensor → (M,7) [x,y,z,dx,dy,dz,yaw]
bboxes3d = pred_instances.bboxes_3d.tensor.cpu().numpy()

# 4) Plot
fig = plt.figure(figsize=(8, 8), facecolor='black')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# scatter points in white
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           s=0.5, c='white', alpha=0.8, linewidths=0)

# helper to compute the 8 corners of a box
def corners_of_bbox(x, y, z, dx, dy, dz, yaw):
    # corners in the box’s local frame
    xc = np.array([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2])
    yc = np.array([   0 ,    0 ,    0 ,    0 ,  -dz ,  -dz ,  -dz ,  -dz ])
    zc = np.array([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2])
    # rotation around Y axis
    R = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                  [           0, 1,           0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    corners = (R @ np.vstack([xc, yc, zc])).T
    corners += np.array([x, y, z])
    return corners  # (8,3)

# edges between the 8 corners
edges = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

# draw each box in cyan
for x,y,z,dx,dy,dz,yaw in bboxes3d:
    c = corners_of_bbox(x, y, z, dx, dy, dz, yaw)
    for i,j in edges:
        ax.plot(*zip(c[i], c[j]), color='cyan', linewidth=1)

# clean up axes
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlim(0, 70);  ax.set_ylim(-40, 40);  ax.set_zlim(-3,  1)
ax.set_xlabel('X');  ax.set_ylabel('Y');    ax.set_zlabel('Z')
ax.set_box_aspect([70, 80, 4])  # keep aspect ratio

# save with black background
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.tight_layout(pad=0)
plt.savefig(out_png, dpi=200, facecolor='black')
print(f"Saved point‐cloud view to {out_png}")
