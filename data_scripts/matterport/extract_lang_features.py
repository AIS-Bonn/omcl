# Extractes language features from RGB images

import numpy as np
import os
import shutil
from tqdm.auto import tqdm
import viser
import torch
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig
import cv2
from tools import make_intrinsics, depth2pc, get_scene_features, opengl_mat

from omcl.utils.plot import pose2viser_wxyz
from omcl.utils.spatial import voxel_down_sample
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
from omcl.utils.mics import mp3d_load_poses
from omcl.models.processing import get_unique_features
from omcl.models.image_encoders.lseg.lseg import LSeg


def get_similarity_ids(image_features, features_db):
    """
    returns the image with feature indexes per pixel
    """
    similarity_ids = (image_features.half().cuda() @ features_db.T.cuda()).argmax(-1)
    return similarity_ids

def process_frames(viser_server: viser.ViserServer, scene_name: str, config: DictConfig, lang_features_from_rgb):
    scene_dir, poses44, _, _ = mp3d_load_poses(scene_name, config)
    for p_i, p in enumerate(poses44):
        viser_server.scene.add_frame(f"/poses44/p_{p_i}", position = p[:3, -1], 
                                    wxyz=pose2viser_wxyz(p), origin_color=[0, 255,0],
                                    axes_radius=config.vis.axes_radius, axes_length=config.vis.axes_length, visible=True)
    depth_dir = os.path.join(scene_dir, 'depth')
    rgb_dir = os.path.join(scene_dir, 'rgb')
    # for visualiztion
    sem_colors = np.concatenate([d3_40_colors_rgb, generate_rgb_colors(config.vis.num_colors)], axis=0)
    assert os.path.exists(depth_dir), depth_dir
    depth_images = os.listdir(depth_dir)
    ids = sorted((depth_file[:-4] for depth_file in depth_images))
    intrinsics = make_intrinsics(config)
    
    all_features_db = torch.tensor([]).cuda()

    save_dir = os.path.join(scene_dir, f'{config.visual_model.name}_semantic')
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    
    scene_features = get_scene_features(scene_dir, config).cuda() # for visualization
    
    for _, id_str in enumerate(tqdm(ids, desc=scene_name)):
        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(rgb_dir, f'{id_str}.png')), cv2.COLOR_BGR2RGB)
        image_features = lang_features_from_rgb(rgb_image).reshape(-1,512)
        new_features = get_unique_features(all_features_db, image_features, similarity_threshold=0.1)
        if len(new_features) > 0:
            all_features_db = torch.cat((all_features_db.half(), new_features.half()))
            print(f"New features_db size: {all_features_db.shape}")
        similarity_ids = get_similarity_ids(image_features, all_features_db).reshape(540, 540).cpu()
        torch.save(similarity_ids, os.path.join(save_dir, f'{id_str}.pt'))
            
        # visualization 
        vis_ids = get_similarity_ids(all_features_db[similarity_ids.flatten()], scene_features).reshape(540, 540).cpu()
        sem_img = sem_colors[vis_ids]
        cv2.imwrite(os.path.join(save_dir, f'{id_str}.png'), cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB))
        depth_img = np.load(os.path.join(depth_dir, f'{id_str}.npy'))

        i = int(id_str)
        pose = poses44[i].numpy()
        xyz, depth_mask = depth2pc(depth_img=depth_img, intrinsics=intrinsics)
        xyz, downs_idx = voxel_down_sample(xyz, 0.05)
        vis_points = pose[:3, :3] @ opengl_mat.numpy() @ xyz + pose[:3, -1][..., None]
        colors = sem_img[depth_mask][downs_idx]
        viser_server.add_point_cloud(name="semantic_points", points=vis_points.T, colors=colors, point_size=0.01)
        _ = viser_server.scene.add_camera_frustum(name='semantic_image', fov=90, aspect=1,
                                                  position=pose[:3, -1], scale=0.25, wxyz=pose2viser_wxyz(torch.tensor(pose).float()),
                                                  image=sem_img)
    print(f"saving features database to {scene_dir}")
    features_labels = [*range(len(all_features_db))]
    torch.save({'features': all_features_db.cpu(),
                'labels': features_labels,
                'scene_features': scene_features.cpu() # for visualization
                },
               os.path.join(scene_dir, f"{config.visual_model.name}_features_db.pt"))
    print("Done!")


@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="mp3d_config",
)
def main(config: DictConfig):
    viser_server = viser.ViserServer()
    viser_server.scene.add_grid('floor', width=50, height=50, position=(0,0, -1.5))
    if config.visual_model.name == "lseg":
        rgb_encoder = LSeg(crop_size=512, base_size=540)
        def predict_image(rgb_image):
            pix_features, pred = rgb_encoder.get_lseg_feat(rgb_image, [""])
            image_features = pix_features[0].permute((1,2,0)).half()
            return image_features
        
    for scene in config.dataset.scenes:
        print("Scene: ", scene)
        process_frames(viser_server, scene_name=scene, config=config, lang_features_from_rgb=predict_image)


if __name__ == '__main__':
    main()
