import torch
import os
import numpy as np
import kaolin as kal
from scipy.spatial.transform import Rotation as R
import clip
import cv2
from omcl.utils.rays import create_rays


def mp3d_load_poses(scene_name, config):
    data_path = os.path.join(os.path.expanduser('~'), config.paths.all_data,
                         config.dataset.name,
                        scene_name)
    opengl_mat = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0,  -1]], dtype=torch.float32)
    
    poses = torch.from_numpy(np.loadtxt( os.path.join(data_path,  "poses.txt"))).float() # rotations are in (x, y, z, w) format
    poses44 = kal.math.quat.rot44_from_quat(poses[:, 3:])
    poses44[:, :3, -1] = poses[:, :3]
    poses44[:, :3, :3] = poses44[:, :3, :3] @ opengl_mat 

    # rot_init = opengl_mat @ torch.from_numpy(R.from_euler('x', -90, degrees=True).as_matrix().astype(np.float32))
    rot_init = torch.from_numpy(R.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32))
    T_init = torch.eye(4)
    T_init[:3, :3] = rot_init
    T_init[:3, -1] = -rot_init @ poses[0][:3]
    

    poses44 = T_init @ poses44
    intr = get_mp3d_intrinsics(config.dataset.camera.h, config.dataset.camera.h, config.dataset.camera.hfov)
    print(intr)
    T2 = kal.math.quat.pad_mat33_to_mat44(torch.from_numpy(R.from_euler('z', -90, degrees=True).as_matrix().astype(np.float32)))
    print(T2)
    T_rays = torch.tensor([
        [ 0, -1, 0, 0],
        [ 0,  0, 1, 0],
        [-1,  0, 0, 0],
        [ 0,  0, 0, 1],
    ], dtype=poses44[0].dtype, device=poses44[0].device)
    rays_o, rays_d, cam = create_rays(pose= T_rays @ T2 @  poses44[0], height=config.dataset.camera.h, 
                                      width=config.dataset.camera.w, fx=intr[0][0], fy=intr[0][0])
    return data_path, poses44, rays_o, rays_d


def kitti_load_poses_kitti(scene_name, config):
    data_path = os.path.join(os.path.expanduser('~'), config.paths.all_data,
                         config.dataset.name,'dataset', 'sequences',
                        scene_name)
    calib = parse_calibration_kitti(os.path.join(data_path, "calib.txt"))
    poses44 = parse_poses_kitti(os.path.join(data_path, "poses.txt"), calib)
    poses44 = poses44 @ np.linalg.inv(calib['Tr']) # approximately
    poses44 = torch.tensor(poses44).float()
    # pose_init = poses44[0][:3, -1].float()
    # rot_init = torch.from_numpy(R.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32))
    # rot_init = torch.eye(3)
    # T_init = torch.eye(4)
    # T_init[:3, :3] = rot_init
    # T_init[:3, -1] = -rot_init @ pose_init
    # T_init =  poses44[0] @ torch.from_numpy(np.linalg.inv(calib['Tr'])).float() @ torch.linalg.inv(poses44[0])
    print(calib['P2'])
    T_rays = torch.tensor([
        [ 0, -1, 0, 0],
        [ 0,  0, 1, 0],
        [-1,  0, 0, 0],
        [ 0,  0, 0, 1],
    ], dtype=poses44[0].dtype, device=poses44[0].device)
    rays_o, rays_d, cam = create_rays(pose=T_rays @ poses44[0], height=config.dataset.camera.h, 
                                      width=config.dataset.camera.w, fx=calib['P2'][0][0], fy=calib['P2'][0][0])
    return data_path, poses44, rays_o, rays_d


def parse_poses_kitti(filename, calibration):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses

# from semantic kitti api
def parse_calibration_kitti(filename):
    """ read calibration file with given filename
        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose
    calib_file.close()
    return calib


def load_data(scene_name, config, device):
    # expected format of input data for omcl
    if config.dataset.name == 'semantic_kitti':
        data_path, poses44, rays_o, rays_d = kitti_load_poses_kitti(scene_name, config)
    else:
        data_path, poses44, rays_o, rays_d = mp3d_load_poses(scene_name, config)
    map_path = os.path.join(data_path, f"{config.visual_model.name}_octree_map.pt")
    print(f'LOADED MAP: {map_path}')
    octree_map = torch.load(map_path, weights_only=True)
    return (data_path, octree_map['points'].float(), octree_map['points_features_idx'], 
            poses44, rays_o, rays_d, octree_map['map_features'], octree_map['rgb_features'], octree_map['vis_scene_features'].to(device))
    


def read_pf_data(scene_dir, i, features_db, config):
    semantic_rgb = os.path.join(scene_dir, f"00000{i}"[-6:]+'.png') # for visualization
    if os.path.exists(semantic_rgb):
        rgb_image = cv2.imread(semantic_rgb)
    else:
        rgb_image = None
    obs_data = torch.load(os.path.join(scene_dir, f"00000{i}"[-6:]+'.pt')).int()
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



def get_mp3d_intrinsics(h, w, fov):
    # https://github.com/vlmaps/vlmaps/blob/master/vlmaps/utils/mapping_utils.py
    # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html#gsc.tab=0
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

