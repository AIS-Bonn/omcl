# Creates MAP from GT semantic information


import numpy as np
import os
from tqdm.auto import tqdm
import viser
import yaml
import torch
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig
from omcl.utils.spatial import crop_height
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
from tools import make_intrinsics, depth2pc
from omcl.models.odom import combine_odoms, estimate_odoms
from omcl.utils.mics import mp3d_load_poses
from omcl.models.map import increment_map, merge_submaps, init_map
from omcl.utils.plot import plot_floor
from omcl.models.image_encoders.minkowski.model import OpenSceneModel
from omcl.models.processing import get_unique_features


def get_map_dict_features(data_path, config, device):
    if config.visual_model.name == 'open_scene':
        # initialize empty features database
        features_db = torch.tensor([]).cuda()
        features_labels = [*range(len(features_db))]
    else:
        # use features database extracted on previous step
        features_data = torch.load(os.path.join(data_path, f"{config.visual_model.name}_features_db.pt"), weights_only=True)
        features_db = features_data['features']
        features_labels = features_data['labels']
    return features_db.to(device), features_labels


def generate_gt(viser_server: viser.ViserServer, scene_name: str, config: DictConfig, submap_size=500):
    scene_dir, poses44, T_init = mp3d_load_poses(scene_name, config)
    depth_dir = os.path.join(scene_dir, 'depth')
    semantic_dir = os.path.join(scene_dir, f'{config.visual_model.name}_semantic')

    assert os.path.exists(depth_dir), depth_dir
    depth_images = os.listdir(depth_dir)
    ids = sorted((depth_file[:-4] for depth_file in depth_images))
    
    PREDICT_ON_POINTS = False
    if config.visual_model.name == 'open_scene':
        model = OpenSceneModel()
        PREDICT_ON_POINTS = True
    else:
        assert os.path.exists(semantic_dir), semantic_dir
        assert len(depth_images) == len(os.listdir(semantic_dir)) // 2
    # configure
    intrinsics = make_intrinsics(config)
    odoms = estimate_odoms(poses44, T_init)
    min_height  = -config.simulate.camera_height + config.scene[scene_name].min_height
    device = 'cuda'
    submaps = []
    map_points, map_features, map_n_means = init_map(config, device, feature_size=config.visual_model.features_size)
    # initial states
    pose = T_init @ poses44[0]
    i_prev = 0
    features_db, labels_db = get_map_dict_features(scene_dir, config, device)
    sem_colors = d3_40_colors_rgb # for visualiztion
    last_data_id = len(ids) - 1
    for seq_num, id_str in enumerate(tqdm(ids, desc=scene_name)):
        i = int(id_str) + 1
        odom = combine_odoms(odoms, i_prev, i)
        i_prev = i
        pose = (pose @ odom)
        pose_dev = pose.to(device)  # cropping on cpu to save memory
        depth_img = np.load(os.path.join(depth_dir, f'{id_str}.npy'))            
        # transform and preprocess data
        xyz, depth_mask = depth2pc(depth_img=depth_img, intrinsics=intrinsics)
        xyz, crop_mask = crop_height(xyz, pose, config.scene[scene_name].max_height, min_height)
        if PREDICT_ON_POINTS:
            xyz_features = None
        else:
            obs_data = torch.load(os.path.join(semantic_dir, f'{id_str}.pt'))
            features_img = features_db[obs_data]
            xyz_features = features_img[depth_mask][crop_mask].to(device)
        xyz = (pose_dev[:3, :3] @ torch.from_numpy(xyz).to(device) + pose_dev[:3, -1][..., None]).T
        # mapping
        map_points, map_features, map_n_means = increment_map(map_points, map_features, xyz, xyz_features, map_n_means, 
                      max_level=config.scene[scene_name].max_level, resolution=config.scene[scene_name].resolution, device=device)
        
        
        if ((seq_num % submap_size == 0) and (seq_num > 0)) or (seq_num == last_data_id):
            print(f'new submap: {seq_num}/{len(ids)}')
            submaps.append((map_points.cpu(), map_features.cpu(), map_n_means.cpu()))
            # visualization
            if not PREDICT_ON_POINTS:
                if len(sem_colors) < len(features_db):
                    sem_colors = np.concatenate([sem_colors, generate_rgb_colors((len(features_db) - len(sem_colors))*10)], axis=0)
                colors = sem_colors[(map_features @ features_db.T).cpu().argmax(-1)]
                viser_server.add_point_cloud(name="submap", points=submaps[-1][0].numpy(), colors=colors, point_size=config.scene[scene_name].resolution)
            else:
                viser_server.add_point_cloud(name="submap", points=submaps[-1][0].numpy(), colors=np.ones((len(submaps[-1][0]), 3))* 0.5, point_size=config.scene[scene_name].resolution)
            # init next submap
            map_points, map_features, map_n_means = init_map(config, device, feature_size=config.visual_model.features_size)

    map_points, map_features, map_n_means = merge_submaps(submaps, config.scene[scene_name].max_level, config.scene[scene_name].resolution, 'cpu')
    print("submaps are merged")
    # viser_server.add_point_cloud(name="map", points=map_points.cpu().numpy(), colors=np.ones((len(map_points), 3))* 0.5, point_size=config.scene[scene_name].resolution)
    # breakpoint()
    if PREDICT_ON_POINTS:
        print('Features prediction on map points')
        map_features = model.forward(map_points, voxel_size=0.02).cuda()    # https://github.com/pengsongyou/openscene/blob/main/config/matterport/mink.yaml
        print(map_points.shape)
        print(map_features.shape)
        features_db = get_unique_features([], map_features, similarity_threshold=0.05, mean=True)
        labels_db = [*range(len(features_db))]
        print(f"New features_db size: {features_db.shape}")
    idx = (map_features.cuda() @ features_db.T.cuda()).cpu().argmax(-1)
    if len(sem_colors) < len(features_db):
        sem_colors = np.concatenate([sem_colors, generate_rgb_colors((len(features_db) - len(sem_colors))*10)], axis=0)  
    viser_server.add_point_cloud(name="map", points=map_points.cpu().numpy(), colors=sem_colors[idx], point_size=config.scene[scene_name].resolution)

    torch.save({'points': map_points, 
                'points_labels': idx,
                'features': features_db,
                'labels': labels_db}, 
                os.path.join(scene_dir, f"{config.visual_model.name}_octree_map.pt"))


@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="mp3d_config",
)
def main(config: DictConfig):
    viser_server = viser.ViserServer()
    for scene in config.paths.scenes:
        print(scene)
        plot_floor(scene, config, viser_server)
        generate_gt(viser_server, scene_name=scene, config=config, submap_size=100)


if __name__ == '__main__':
    main()
