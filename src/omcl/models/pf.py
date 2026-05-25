# https://notebooks.gesis.org/binder/jupyter/user/rlabbe-kalman-a-lters-in-python-dh6seu3j/lab/tree/12-Particle-Filters.ipynb
#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
import torch
import numpy as np
from numpy.random import uniform
import scipy
import random
import kaolin as kal
from scipy.spatial.transform import Rotation as R
from omcl.utils.rays import ray_trace_pose_batched
from omcl.utils.rays import get_input_masks, get_input_masks_cropped
import os
import math
from omcl.utils.plot import (plot_camera_rgb, plot_paricles, plot_camera_frame, 
                             pose2viser_wxyz,
                             plot_floor, plot_map_nodes, plot_data_points, plot_semantic_camera,
                             plot_raycast_view)
from omcl.utils.rays import transform_rays_batched
from omcl.models.odom import get_mean_pose, combine_odoms, estimate_odoms
from omcl.utils.mics import read_pf_data, load_data, load_mesh_pts
from tqdm.auto import trange, tqdm
from omcl.models.map import points_features2octree
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
from scipy.spatial.transform import Rotation as R
import cv2
from pathlib import Path


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    dt = torch.empty((N, 2))
    dt[:, 0] = mean[0] + (torch.randn(N) * std[0])
    dt[:, 1] = mean[1] + (torch.randn(N) * std[1])
    
    phi = torch.zeros(N, 3)
    phi[:, 1] = torch.randn(N) * std[-1]
    angle = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = phi / angle
    dR =kal.math.quat.rot33_from_angle_axis(angle=angle, axis=axis)
    
    weights = torch.ones(N)
    return dt, dR, weights


def init_particles(pose, std, N):
    pose_cpu = pose.cpu()
    dt, dR, weights = create_gaussian_particles([pose_cpu[0,-1], pose_cpu[1,-1]], std, N)

    particles = torch.eye(4, 4)[None].repeat(N, 1, 1)
    particles[:, :3,:3] = pose_cpu[None, :3, :3] @ dR
    particles[:, :2, -1] = dt
    particles[:, 2, -1] = pose_cpu[2, -1]
    return particles.to(pose.device), weights.double().to(pose.device)

    
def predict(particles, odom, config):
    device = particles.device
    dtype = particles.dtype
    N = len(particles)

    phi = torch.randn(N, 3, device=device, dtype=dtype)
    phi[:, 0] *= config.odom.rot_x_std
    phi[:, 1] *= config.odom.rot_y_std
    phi[:, 2] *= config.odom.rot_z_std
    angle = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = phi / angle
    dR =kal.math.quat.rot33_from_angle_axis(angle=angle, axis=axis)
    
    n_odom = odom[None].repeat(len(particles), 1, 1).clone()
    n_odom[:, :3, :3] = n_odom[:, :3, :3]  @ dR
    n_odom[:, 0, -1] += torch.randn(N, device=device, dtype=dtype)  * config.odom.x_std
    n_odom[:, 1, -1] += torch.randn(N, device=device, dtype=dtype)  * config.odom.y_std
    n_odom[:, 2, -1] += torch.randn(N, device=device, dtype=dtype)  * config.odom.z_std

    return particles @ n_odom

    
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    
    
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var


def stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (torch.rand(N) + torch.arange(0, N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def torch_stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = torch.clip((torch.rand(N, dtype=torch.float64) + torch.arange(0, N, dtype=torch.float64)) / N, 0, 1- 1e-12)

    indexes = torch.zeros(N, dtype=torch.int)
    cumulative_sum = torch.cumsum(weights, 0)
    # cumulative_sum = torch.clip(cumulative_sum, 0, 1 - 1e-12)
    # cumulative_sum[-1] = 1   # Prevent overshoot due to numerical issues
    cumulative_sum[-1]  = max(cumulative_sum[-1], 1)  # Prevent overshoot due to numerical issues
    i, j = 0, 0
    try:
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
    except Exception as e:
        print(f'i: {i}, N: {N}, j: {j}')
        print('positions')
        print(positions.shape)
        print(positions)
        print('cumulative_sum')
        print(cumulative_sum.shape)
        print(cumulative_sum[-10:])
        print(cumulative_sum)
        print('weights')
        print(weights.sum())
        print(weights.shape)
        print(weights.max())
        print(weights.min())
        
        raise e
    return indexes


def estimate_weights(weights, batches_loss):
    weights = weights * batches_loss
    # weights[batches_mask.any(-1).logical_not()] = 1.e-5
    weights /= (weights.sum() + 1.e-9)
    return weights


#TODO: clip loss to [0,1]
def estimate_loss(batches_mask, rays_ids_batched, rays_features, closest_rays_indx_batched, encodings_device, bs, output_tensor, output_slice):
    loss = torch.zeros(closest_rays_indx_batched.shape[0], device=encodings_device.device)
    n_bs = closest_rays_indx_batched.shape[0] // bs
    for i in range(bs):
        st = i*n_bs
        end = (i+1)*n_bs
        if end >= closest_rays_indx_batched.shape[0]:
            end = -1
        # gen_features = encodings_device[rays_ids_batched[st:end].int()]
        # gt_features = rays_features[closest_rays_indx_batched[st:end]]
        # loss[st:end] = torch.nn.functional.cosine_similarity(gen_features, gt_features.detach(), dim=1)
        loss[st:end] = torch.nn.functional.cosine_similarity(encodings_device[rays_ids_batched[st:end]],  # ray-traced features
                                                             rays_features[closest_rays_indx_batched[st:end]],  # measured features by sensor (gt)
                                                             dim=1)

        # loss[st:end] = torch.einsum(
        #                     'ij,ij->i',
        #                     encodings_device[rays_ids_batched[st:end]],
        #                     rays_features[closest_rays_indx_batched[st:end]]
        #                         )


    batches_loss = torch.zeros_like(batches_mask, dtype=loss.dtype, device=loss.device)
    batches_loss[batches_mask] = loss
    output_tensor[output_slice] = batches_loss.sum(-1) / torch.maximum(batches_mask.count_nonzero(-1), torch.tensor(1, device=batches_mask.device))
    # batch_indices, _ = torch.nonzero(batches_mask, as_tuple=True)
    # batches_loss = torch.zeros(batches_mask.size(0), dtype=loss.dtype, device=loss.device)
    # batches_loss.scatter_add_(0, batch_indices, loss)
    # output_tensor[output_slice] = batches_loss / batches_mask.count_nonzero(dim=-1).clamp(min=1)
    # return batches_loss


def get_masks_features(input_data_mask, obs_features, encodings_device, classes_ids, device, config, num_samples=1024):
    if config.sampling == 'default':
        reg_mask = torch.zeros(input_data_mask.shape[0], input_data_mask.shape[1], dtype=bool, device=device)
        # sample random rays
        mask_ij = get_input_masks(input_data_mask, obs_features.cuda(), encodings_device, classes_ids.cuda(), num_samples=num_samples)
        # ids = mask_ij[:, 0] * input_data_mask.shape[0] + mask_ij[:, 1]
        success = mask_ij is not None
        if success:
            reg_mask[mask_ij[:,0], mask_ij[:, 1]] = 1
        rays_mask = reg_mask.flatten()
        return rays_mask.cpu(), rays_mask, success
    
    elif config.sampling == 'cropped':
        mask_ij = get_input_masks_cropped(input_data_mask, obs_features.cuda(), encodings_device, classes_ids.cuda(), num_samples=num_samples)
        rays_ids = mask_ij[:, 0] * input_data_mask.shape[0] + mask_ij[:, 1]
        return rays_ids.cpu(), rays_ids.cuda()
    elif config.sampling == 'equall':
        mask_ij = get_input_masks(input_data_mask, obs_features.cuda(), encodings_device, classes_ids.cuda(), num_samples=num_samples)
        success = mask_ij is not None
        if success:
            rays_ids = mask_ij[:, 0] * input_data_mask.shape[1] + mask_ij[:, 1]
            return rays_ids.cpu(), rays_ids.cuda(), success
        return torch.tensor([0]), torch.tensor([0]).cuda(), False
    raise NotImplementedError
    # if False:
    #     mask_ij = get_input_masks(input_data_mask, obs_features.cuda(), encodings_device, classes_ids.cuda(), num_samples=num_samples)
    #     rays_ids = mask_ij[:, 0] * input_data_mask.shape[0] + mask_ij[:, 1]
    #     return rays_ids, rays_ids.cuda()
    return rays_mask, rays_mask.cuda()

    
def measurements_info(config, data_path):
    if config.visual_model.name == 'open_scene':
        scene_dir = os.path.join(data_path,  f"lseg_semantic")
    else:
        scene_dir = os.path.join(data_path,  f"{config.visual_model.name}_semantic")
    print(f"{scene_dir} is ised")
    aspect = config.dataset.camera.w / config.dataset.camera.h
    hfov = math.radians(config.dataset.camera.hfov)
    return scene_dir, aspect, hfov

def load_octree_map(points, points_labels, scene_name, config):
    print("Create octree map")
    spc, spc_labels, scale =  points_features2octree(points.cuda(), points_labels, 
                                                    config.scene[scene_name].max_level, 
                                                    config.scene[scene_name].resolution)
    return (spc, spc_labels, scale)
        
        
def run(scene_name, config, first_pose_id=0, device='cuda', viser_server=None, octree_map=None, batch_size=1000, inital_particles=None, global_localization=False, plot_map=True):
    torch.set_grad_enabled(False)
    print("loading the data")
    scene_config = config.dataset.scenes_config[scene_name]
    (data_path, points, points_labels, poses44, rays_o, rays_d, 
      map_features_db, rgb_features_db, vis_scene_features) = load_data(scene_name, config, device)
    scene_dir, aspect, hfov = measurements_info(config, data_path)
    rays_d = rays_d.to(device=device)
    rays_o = rays_o.to(device=device)
    if config.semantic_grounding:
        # Semantic Grounding remaps automatically extracted features to the user prompt features.
        # For simplicity, we demonstrate this using scene labels as the user prompt.
        # vis_scene_features was created from the scene labels, and its direct clone saves resources.
        # https://github.com/AIS-Bonn/omcl/blob/4483b9ae098f79a5631b04802258d6b62120a4a0/data_scripts/matterport/tools.py#L83
        user_prompt_features = vis_scene_features.clone().detach()  # get features from the user prompt (e.g., scene labels)
        remap_ids = (map_features_db @ user_prompt_features.T).argmax(-1).cpu()
        map_features_db = user_prompt_features.clone().detach()
        points_labels = remap_ids[points_labels]
    
    map_features_db_device = map_features_db.to(device)
    rgb_features_db_device = rgb_features_db.to(device)
    
    classes_ids = torch.arange(0, rgb_features_db.shape[0])
    odoms = estimate_odoms(poses44).to(device=device)

    if octree_map is None:
        print("Create octree map")
        points = points.cuda()
        spc, spc_labels, scale =  points_features2octree(points, points_labels, 
                                                        scene_config.max_level, 
                                                        scene_config.resolution,
                                                        vectorize=True)
        points = points.cpu()
        octree_map = (spc, spc_labels, scale)
    else:
        spc, spc_labels, scale = octree_map
    point_hierarchy = kal.ops.spc.generate_points(spc.octrees, spc.pyramids, spc.exsum)
    pyramid = spc.pyramids[0]

    # visualization
    if viser_server is not None:
        colors_map = np.concatenate([d3_40_colors_rgb, generate_rgb_colors(config.vis.num_colors)], axis=0)
        if config.vis.plot_data:
            plot_data_points(points.cpu(), points_labels, colors_map, viser_server)
        plot_floor(scene_name, config, viser_server)
        # remap map colors to rgb_features_db  colors
        plot_map_nodes(vis_scene_features, map_features_db, point_hierarchy, spc_labels, pyramid, scale, scene_config, colors_map, viser_server, stride=config.vis.map_stride, visible=plot_map)
        if config.vis.odom_stride > 0:
            _i_prev = 0
            _p0 = poses44[_i_prev]
            for p_i, p in enumerate(poses44[::config.vis.odom_stride]):
                _p_odom =  _p0 @ combine_odoms(odoms.cpu(), _i_prev, p_i)
                _i_prev = p_i
                viser_server.scene.add_frame(f"/poses44/p_{p_i}", position = p[:3, -1], 
                                            wxyz=pose2viser_wxyz(p), origin_color=[0, 255,0],
                                            axes_radius=config.vis.axes_radius, axes_length=config.vis.axes_length, visible=False)
                viser_server.scene.add_frame(f"/odoms/p_{p_i}", position = p[:3, -1], 
                                            wxyz=pose2viser_wxyz(_p_odom), origin_color=[0, 255,0],
                                            axes_radius=config.vis.axes_radius, axes_length=config.vis.axes_length, visible=False)
             
    print("run")
    i_prev = first_pose_id
    particles, weights = init_particles(poses44[i_prev].detach().to(device), std=[config.init_std_x, config.init_std_y, config.init_std_th], N=config.num_particles)
    
    plot_paricles(particles, hfov, aspect, viser_server, config)
    plot_camera_rgb(poses44[i_prev], None, hfov, aspect, viser_server, config)
    # visualization
    rays_origin_colors, _, _, _ = read_pf_data(scene_dir, 0, rgb_features_db, config)
    _ = viser_server.scene.add_point_cloud(
        name="rays_origin",
        points=(rays_o + rays_d * 1.0).cpu().numpy(),
        colors=rays_origin_colors.reshape(-1, 3),
        point_size=0.02,
        visible=False)

    estimated_poses = []
    precision_steps = np.array([0 for _ in config.global_localization.distances])
    all_losses = torch.zeros(particles.shape[0], device=device)
    if inital_particles is not None:
        particles = inital_particles.to(particles.device)
    with torch.no_grad():
        for i in trange(first_pose_id, len(os.listdir(scene_dir))//2):
            rgb_image, obs_features, input_data_mask, _ = read_pf_data(scene_dir, i, rgb_features_db, config)
            rays_mask_cpu, rays_mask, _features_masks_success = get_masks_features(input_data_mask, obs_features, rgb_features_db_device, classes_ids, device, config, num_samples=config.num_samples)
            rays_features = obs_features.reshape(-1,512)[rays_mask_cpu].to(device)
            odom = combine_odoms(odoms, i_prev, i+1)
            i_prev = i+1
            particles = predict(particles, odom, config=config)

            if not _features_masks_success: # only prediction step if no features are found
                print(f"Corrupted features/masks for frame {i}, skip it")
                continue
            # iterate over the batches of particless
            all_losses.zero_()
            rays_o_masked = rays_o[rays_mask]
            rays_d_masked = rays_d[rays_mask]
            for batch_idx in range(math.ceil(particles.shape[0] / batch_size)):
                st = batch_idx*batch_size
                end = min((batch_idx+1)*batch_size, particles.shape[0])
                    
                rays_o_pose_batched, rays_d_pose_batched = transform_rays_batched(rays_o_masked, rays_d_masked, particles[st:end])
                if rays_o_pose_batched.nelement() == 0: # corrupted semantic (based on rays_mask?)
                    continue
                closest_rays_indx_batched, _, rays_ids_batched, batches_mask = ray_trace_pose_batched(rays_o_pose_batched / scale, 
                                                                                    rays_d_pose_batched,
                                                                                    spc, 
                                                                                    point_hierarchy, 
                                                                                    spc_labels)
        
                if len(rays_ids_batched) == 0:
                    continue  # no intersections are found -> zero weights
                estimate_loss(batches_mask, rays_ids_batched, rays_features, closest_rays_indx_batched, map_features_db_device, bs=config.rays_loss_batch_size, output_tensor=all_losses, output_slice=slice(st, end))

            weights = estimate_weights(weights, all_losses)
            mean_pose, p_rot33_mean = get_mean_pose(particles, weights)
            # resample
            samples = torch_stratified_resample(weights)
            particles = particles[samples]
            weights = weights[samples]

            # save trajectory
            estimated = torch.eye(4, device='cpu')
            estimated[:3,:3] = p_rot33_mean.cpu()
            estimated[:3, -1] = mean_pose.cpu()
            estimated_poses.append(estimated.numpy())
            
            if global_localization:
                distance = np.linalg.norm((estimated[:3, -1] - poses44[i][:3, -1]).numpy())
                for precision_idx in range(len(precision_steps)):
                    if (distance <= config.global_localization.distances[precision_idx]) and (precision_steps[precision_idx] == 0):
                        precision_steps[precision_idx] = i + 1
                        print(f"Achieved {distance} m after {precision_steps[precision_idx] - first_pose_id} steps")
                if np.all(precision_steps > 0):
                    return poses44, estimated_poses, octree_map, precision_steps
                    
            
            # visualize
            if viser_server is not None:
                if config.vis.plot_particles:
                    plot_paricles(particles, hfov, aspect, viser_server, config)

                if config.vis.plot_gt:
                    if config.vis.demo: # only red frame
                        plot_camera_rgb(poses44[i], None, hfov, aspect, viser_server, config)
                    else:
                        plot_camera_rgb(poses44[i], rgb_image, hfov, aspect, viser_server, config)
                
                if config.vis.plot_rays:
                    plot_raycast_view(rays_o, rays_d, estimated, spc, point_hierarchy, spc_labels, scale, 
                                      map_features_db, vis_scene_features,
                                      colors_map, viser_server, config, [0,0,255])
                        
                if config.vis.demo:
                    image1 =  cv2.imread(os.path.join(str(Path(scene_dir).parent.absolute()), 'rgb', f"00000{i}"[-6:]+'.png'))
                    plot_semantic_camera(image1, rgb_image, estimated, [0,255,0], hfov, aspect, viser_server, config)
                else:
                    plot_camera_frame('mean_cam', estimated, [0,255,0], hfov, aspect, viser_server, config)
        
    return poses44, estimated_poses, octree_map, precision_steps
