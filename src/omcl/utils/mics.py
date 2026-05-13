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


def kitti_load_poses_kitti(scene_name, config):
    data_path = os.path.join(os.path.expanduser('~'), config.paths.all_data,
                         config.dataset.name,
                        scene_name)
    # poses = torch.from_numpy(read_poses( os.path.join(data_path,  "poses.txt"))).float()
    poses, velodyne_files, label_files, sequence_folder, velodyne_folder, label_folder = get_semantic_kitti_sequence_data(
            os.path.join(os.path.expanduser('~'), config.paths.all_data, 'dataset', 'sequences'), scene_name)
    poses44 = torch.tensor(poses).float()
    pose_init = poses44[0][:3, -1].float()
    # rot_init = torch.from_numpy(R.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32))
    rot_init = torch.from_numpy(np.eye(3).astype(np.float32))  
    return data_path, poses44, pose_init, rot_init


def load_data(scene_name, config):
    # expected format of input data for omcl
    if config.dataset.name == 'semantic_kitti':
        data_path, poses44, T_init = kitti_load_poses_kitti(scene_name, config)
    else:
        data_path, poses44, T_init = mp3d_load_poses(scene_name, config)
    map_path = os.path.join(data_path, f"{config.visual_model.name}_octree_map.pt")
    print(f'LOADED MAP: {map_path}')
    octree_map = torch.load(map_path, weights_only=True)
    return (data_path, octree_map['points'].float(), octree_map['points_labels'], 
            poses44, T_init, octree_map['features'], octree_map['labels'])


def read_pf_data(scene_dir, i, features_db, config):
    rgb_image = cv2.imread(os.path.join(scene_dir, f"00000{i}"[-6:]+'.png'))
    obs_data = torch.load(os.path.join(scene_dir, f"00000{i}"[-6:]+'.pt'))
    input_data_mask = obs_data > -1
    obs_data[~input_data_mask] = 0  # random class for wrong observations
    return rgb_image, features_db[obs_data], input_data_mask, obs_data


def load_mesh_pts(data_path):
    map_path = os.path.join(data_path, "gt_octree_map_mesh.pt")
    features_path = os.path.join(data_path, "gt_features_db.pt")

    data = torch.load(map_path, weights_only=True, map_location=torch.device('cpu'))
    features_data = torch.load(features_path, weights_only=True, map_location=torch.device('cpu'))  
    return (data['points'].float(), data['points_labels'], 
            features_data['features'], features_data['labels'])
