import numpy as np
import os
from tqdm.auto import tqdm, trange
import viser
import yaml
import torch
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig
from omcl.utils.spatial import crop_height, voxel_down_sample
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
from omcl.models.odom import combine_odoms, estimate_odoms
from omcl.utils.mics import mp3d_load_poses
from omcl.models.map import increment_map, merge_submaps, init_map
from omcl.utils.plot import plot_floor, pose2viser_wxyz
from omcl.models.image_encoders.minkowski.model import OpenSceneModel
from omcl.models.processing import get_unique_features
from tools import parse_calibration, parse_poses, COLOR_MAP, LABELS, remap_array
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), "omcl/third_party/semantic-kitti-api"))
from auxiliary import laserscan # semantic-kitti-api



def process(viser_server: viser.ViserServer, scene_name: str, config: DictConfig, submap_size=500):
    scene_dir = os.path.join(os.path.expanduser('~'), 'data', 'semantic_kitti', 'dataset', 'sequences', scene_name)
    scene_config = config.dataset.scenes_config[scene_name]
    # From KITTI dataset
    velodyne_folder = os.path.join(scene_dir, "velodyne")
    calib_file = os.path.join(scene_dir, "calib.txt")
    velodyne_files = sorted(os.listdir(velodyne_folder))
    calib = parse_calibration(calib_file)
    # From SemanticKITTI dataset
    poses_file = os.path.join(scene_dir, "poses.txt")
    label_folder = os.path.join(scene_dir, "labels")
    label_files = sorted(os.listdir(label_folder))
    poses44 = parse_poses(poses_file, calib)
    
    max_label = max(COLOR_MAP.keys())
    color_array = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for label_id, color in COLOR_MAP.items():
        color_array[label_id] = color
    
    # Create a mapping from original keys to new sequential keys
    key_mapping = {original_key: new_key for new_key, original_key in enumerate(sorted(LABELS.keys()))}

    
    all_points = []
    all_sem_labels = []  
    # _vis_all_sem_images = [] # for debug visualization
    _vis_all_sem_colors = [] # for debug visualoization
    pts_map = np.empty((0,3))
    labels_map = np.empty((0,))
    _vis_colors_map = np.empty((0,3))
    
    for i in trange(len(poses44)):
        # read data
        ls = laserscan.SemLaserScan(COLOR_MAP, project=True)
        ls.open_scan(filename=os.path.join(velodyne_folder, velodyne_files[i]))
        ls.open_label(filename=os.path.join(label_folder, label_files[i]))    
        pose = poses44[i]
        ls.do_label_projection()
        ls.colorize()
        point_cloud_hom = np.hstack((ls.points, np.ones((ls.points.shape[0], 1))))
        world_points_hom = pose  @ point_cloud_hom.T
        cam_pose = pose @ np.linalg.inv(calib['Tr']) # approximately
        world_points = world_points_hom.T[:, :3]
        all_points.append(world_points)
        # all_sem_labels.append(ls.sem_label)
        all_sem_labels.append(remap_array(ls.sem_label, key_mapping))
        _vis_all_sem_colors.append(ls.sem_label_color)
        # filter invalid points
        valid_mask = np.logical_and(ls.sem_label > 1, ls.sem_label < 200) 
        all_sem_labels[-1] = all_sem_labels[-1][valid_mask]
        all_points[-1] = all_points[-1][valid_mask]
        _vis_all_sem_colors[-1] = _vis_all_sem_colors[-1][valid_mask]
        # _vis_all_sem_images.append(remap_array(ls.proj_sem_label, key_mapping))
        aspect = ls.proj_sem_color.shape[1] / ls.proj_sem_color.shape[0]
        viser_server.scene.add_point_cloud(name="measurements", points=world_points, colors=ls.sem_label_color, point_size=0.02)
        _ = viser_server.scene.add_camera_frustum(name='semantic_image', fov=90, aspect=aspect, position=cam_pose[:3, -1], scale=0.5, wxyz=pose2viser_wxyz(torch.tensor(cam_pose).float()), image=ls.proj_sem_color)
        if i % submap_size == 0:
            points_submap = np.concatenate(all_points)
            sem_labels_submap = np.concatenate(all_sem_labels)
            _vis_sem_colors_submap = np.concatenate(_vis_all_sem_colors)
            points_map_ds, downs_idx = voxel_down_sample(points_submap, 0.05)
            pts_map = np.concatenate((pts_map, points_map_ds))
            _vis_colors_map = np.concatenate((_vis_colors_map, _vis_sem_colors_submap[downs_idx]))
            labels_map = np.concatenate((labels_map, sem_labels_submap[downs_idx]))
            all_points = []
            all_sem_labels = []
            _vis_all_sem_colors = []
            assert len(pts_map) == len(labels_map)
            assert len(pts_map) == len(_vis_colors_map)
            viser_server.scene.add_point_cloud(name="semantic_map", points=pts_map[::10], colors=_vis_colors_map[::10], point_size=0.05)
    
    if len(all_points) > 0:
        points_submap = np.concatenate(all_points)
        sem_labels_submap = np.concatenate(all_sem_labels)
        print("Downsampling...")
        points_map_ds, downs_idx = voxel_down_sample(points_submap, 0.05)
        print('Finished')
        pts_map = np.concatenate((pts_map, points_map_ds))
        labels_map = np.concatenate((labels_map, sem_labels_submap[downs_idx])).astype(np.long)
    
    sem_colors = np.array([v for v in COLOR_MAP.values()])
    # hide previous map
    viser_server.scene.add_point_cloud(name="semantic_map", points=pts_map[::10], colors=_vis_colors_map[::10], point_size=0.05, visible=False)
    # show actual map colored by labels variable
    viser_server.scene.add_point_cloud(name="final_map", points=pts_map[::10], colors=sem_colors[labels_map[::10].astype(np.long)], point_size=0.05)
    pts_map = torch.from_numpy(pts_map)
    labels_map = torch.from_numpy(labels_map).to(torch.long)
    
    # use features database extracted on previous step
    features_data = torch.load(os.path.join(scene_dir, f"{config.visual_model.name}_features_db.pt"), weights_only=True)
    rgb_features_db = features_data['features']
    scene_features = features_data['scene_features']
    
    
    torch.save({'points': pts_map, 
                'points_features_idx': labels_map,
                'map_features': scene_features,
                'rgb_features': rgb_features_db,
                'vis_scene_features': scene_features.cpu()}, 
                os.path.join(scene_dir, f"{config.visual_model.name}_octree_map.pt"))

@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="sem_kitti_config",
)
def main(config: DictConfig):
    viser_server = viser.ViserServer()
    for scene in config.dataset.scenes:
        print(scene)
        plot_floor(scene, config, viser_server)
        process(viser_server, scene_name=scene, config=config, submap_size=500)


if __name__ == '__main__':
    main()