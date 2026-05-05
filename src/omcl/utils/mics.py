import torch
import os
import numpy as np
import kaolin as kal
from scipy.spatial.transform import Rotation as R
import clip
import cv2


def mp3d_load_poses(scene_name, config):
    data_path = os.path.join(os.path.expanduser('~'), config.paths.all_data,
                         config.dataset.name,
                        scene_name)
    poses = torch.from_numpy(np.loadtxt( os.path.join(data_path,  "poses.txt"))).float() # rotations are in (x, y, z, w) format
    pose_init = poses[0][:3]
    rot_init = torch.from_numpy(R.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32))

    poses44 = kal.math.quat.rot44_from_quat(poses[:, 3:])
    poses44[:, :3, -1] = poses[:, :3]
    T_init = torch.eye(4)
    T_init[:3, :3] = rot_init
    T_init[:3, -1] = -rot_init @ pose_init
    return data_path, poses44, T_init