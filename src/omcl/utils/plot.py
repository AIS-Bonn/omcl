import torch
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from omcl.utils.rays import ray_trace_pose_batched, transform_rays_batched
import math


def plot_floor(scene_name, config, viser_server):
    side_length = 2**config.dataset.scenes_config[scene_name].max_level * config.dataset.scenes_config[scene_name].resolution
    _ = viser_server.scene.add_grid('floor', width=side_length, height=side_length, width_segments=50, position=(0,0, -1.5),
                                    visible=config.vis.floor)


def pose2viser_wxyz(pose):
    quat_camera =  R.from_matrix(pose[:3,:3].cpu().float()).as_quat()
    return (quat_camera[-1], *quat_camera[:-1])


def rot2viser_wxyz(rot):
    quat_camera = R.from_matrix(rot.cpu().float()).as_quat()
    return (quat_camera[-1], *quat_camera[:-1])


def plot_camera_rgb(pose, rgb_image, hfov, aspect, viser_server, config):
    if rgb_image is None:
        image = None
    else:
        image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    _ = viser_server.scene.add_camera_frustum(name='cam', 
                                              fov=hfov, 
                                              aspect=aspect, 
                                              position=pose[:3, -1].cpu(),
                                              scale=config.vis.frustum_scale,
                                              wxyz=pose2viser_wxyz(pose),
                                              image=image,
                                            color=[255,0,0],
                                            line_width=6)


def plot_paricles(particles, hfov, aspect, viser_server, config, visible=True):
    for i in range(len(particles)):
        if i % config.vis.particles_stride == 0:
            p = particles[i]
            _ = viser_server.scene.add_camera_frustum(name=f'/particles/p_{i}', 
                                                fov=hfov, 
                                                aspect=aspect, 
                                                position=p[:3, -1].cpu(),
                                                scale=config.vis.frustum_scale,
                                                wxyz=pose2viser_wxyz(p),
                                                visible=visible)


def plot_camera_frame(name, pose, color, hfov, aspect, viser_server, config, image=None):
    viser_server.scene.add_camera_frustum(name=name, 
                                              fov=hfov, 
                                              aspect=aspect, 
                                              position=pose[:3, -1].cpu(),
                                              scale=config.vis.frustum_scale,
                                              wxyz=pose2viser_wxyz(pose),
                                              image=image,
                                              color=color)


def plot_map_nodes(vis_features, map_features_db, point_hierarchy, spc_labels, pyramid, scale, scene_config, d3_40_colors_rgb, viser_server, stride=1, visible=False):
    num_nodes = 2**scene_config.max_level
    node_size = 2/num_nodes
    gpts = -1. + (point_hierarchy.cpu()[pyramid[1, -2]:pyramid[1, -1]])  * node_size + 0.5*node_size
    if stride > 1:
        points = gpts[::stride].cpu().numpy()*scale
        vis_ids = (map_features_db[spc_labels[::stride]].cuda() @ vis_features.T).argmax(-1).cpu()
        colors = d3_40_colors_rgb[vis_ids.int()] / 255
    else:
        points = gpts.cpu().numpy()*scale
        vis_ids = (map_features_db[spc_labels].cuda() @ vis_features.T).argmax(-1).cpu()
        colors = d3_40_colors_rgb[vis_ids.int()] / 255
    _ = viser_server.scene.add_point_cloud(
        name="map_nodes",
        points=points,
        colors=colors,
        point_size=scene_config.resolution,
        visible=visible)
    
    
def plot_data_points(points, points_labels, d3_40_colors_rgb, viser_server):
    # for debug: should look the same as map_nodes
    _ = viser_server.scene.add_point_cloud(
        name="loaded_points",
        points=np.array(points),
        colors=d3_40_colors_rgb[points_labels] / 255,
        point_size=0.02,
        visible=False)
    
    
def plot_semantic_camera(image1, image2, pose, color, hfov, aspect, viser_server, config):
    if image1 is None or image2 is None:
        return
    image = np.zeros_like(image1)
    image[:, :image.shape[1]//2] =  cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)[:, :image.shape[1]//2]
    image[:, image.shape[1]//2:] =  cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)[:, image.shape[1]//2:]
    _ = viser_server.scene.add_camera_frustum(name='sem_cam', 
                                              fov=hfov, 
                                              aspect=aspect, 
                                              position=pose[:3, -1].cpu(),
                                              scale=config.vis.frustum_scale,
                                              wxyz=pose2viser_wxyz(pose),
                                              image=image,
                                            color=color,
                                            line_width=6)


def raycast_view(rays_o, 
                 rays_d, 
                 particle, 
                 spc,
                 point_hierarchy, 
                 spc_labels, 
                 scale,
                 colors_map,
                 map_features, vis_features):
    
    rays_o_pose_batched, rays_d_pose_batched = transform_rays_batched(rays_o, rays_d, particle[None])
    closest_rays_indx_batched, _, rays_ids_batched, batches_mask = ray_trace_pose_batched(rays_o_pose_batched / scale, 
                                                                        rays_d_pose_batched,
                                                                        spc, 
                                                                        point_hierarchy, 
                                                                        spc_labels)
    vis_ids = (map_features[rays_ids_batched].cuda() @ vis_features.T).argmax(-1).cpu()
    
    return colors_map[vis_ids], batches_mask[0], rays_o_pose_batched, rays_d_pose_batched


def plot_raycast_view(rays_o, 
                 rays_d, 
                 particle, 
                 spc,
                 point_hierarchy, 
                 spc_labels, 
                 scale,
                 map_features, vis_features,
                 colors_map,
                 viser_server,
                 config,
                 color):
    with torch.no_grad():
        rays_colors, image_mask, rays_o_pose_batched, rays_d_pose_batched = raycast_view(rays_o, rays_d, particle.to(rays_o.device), spc, point_hierarchy, spc_labels, scale, colors_map,
                                                                                         map_features, vis_features)
        rays_image = np.zeros((image_mask.shape[0], 3))
        rays_image[image_mask.cpu()] = rays_colors
        aspect = config.dataset.camera.w/config.dataset.camera.h
   
        _ = viser_server.scene.add_camera_frustum(name='raycast_view', 
                                                    fov=math.radians(config.dataset.camera.hfov), 
                                                    aspect=aspect, 
                                                    position=particle[:3, -1].cpu(),
                                                    scale=config.vis.frustum_scale,
                                                    wxyz=pose2viser_wxyz(particle),
                                                    image=rays_image.reshape((config.dataset.camera.h, config.dataset.camera.w, 3))/255,
                                                    color=color,
                                                    line_width=6)
             
        points = ((rays_o_pose_batched + rays_d_pose_batched)[0][image_mask]).cpu().numpy()
        colors = rays_colors
        _ = viser_server.scene.add_point_cloud(
            name="rays",
            points=points,
            colors=colors,
            point_size=0.02,
            visible=False)
        
        
def printRed(s): print("\033[91m {}\033[00m".format(s))
def printGreen(s): print("\033[92m {}\033[00m".format(s))
def printYellow(s): print("\033[93m {}\033[00m".format(s))
def printLightPurple(s): print("\033[94m {}\033[00m".format(s))
def printPurple(s): print("\033[95m {}\033[00m".format(s))
def printCyan(s): print("\033[96m {}\033[00m".format(s))
def printLightGray(s): print("\033[97m {}\033[00m".format(s))
def printBlack(s): print("\033[90m {}\033[00m".format(s))  # Corrected from 98 to 90 (standard ANSI)
