# https://notebooks.gesis.org/binder/jupyter/user/rlabbe-kalman-a-lters-in-python-dh6seu3j/lab/tree/12-Particle-Filters.ipynb
#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
import torch
import numpy as np
from numpy.random import uniform, randn
import scipy
import random
import kaolin as kal
from scipy.spatial.transform import Rotation as R
from omcl.utils.rays import ray_trace_pose_batched
from omcl.utils.rays import get_input_masks, get_input_masks_cropped
import os
import math
from omcl.utils.plot import (plot_camera_rgb, plot_paricles, plot_camera_frame, 
                             mp3d_rot2viser_wxyz, mp3d_rot2viser_wxyz2,
                             plot_floor, plot_map_nodes, plot_data_points, plot_semantic_camera,
                             raycast_view, plot_raycast_view)
from omcl.utils.rays import create_rays, transform_rays_batched
from omcl.models.odom import get_mean_pose, combine_odoms, estimate_odoms
from omcl.utils.mics import read_pf_data, load_data, load_mesh_pts
from tqdm.auto import trange, tqdm
from omcl.models.map import points_features2octree
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
from copy import copy
from omcl.models.processing import get_unique_features
from scipy.spatial.transform import Rotation as R
import cv2
from pathlib import Path
import shutil

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = torch.empty((N, 3))
    particles[:, 0] = mean[0] + (torch.randn(N) * std[0])
    particles[:, 1] = mean[1] + (torch.randn(N) * std[1])
    particles[:, 2] = mean[2] + (torch.randn(N) * std[2])
    particles[:, 2] %= 2 * torch.pi
    weights = torch.ones(N)
    return particles, weights


def init_particles(pose, T_init, std, N): #TODO: change scipy to pypose
    pose_cpu = pose.cpu()
    orientation = R.from_matrix(pose_cpu[:3,:3] @ T_init[:3, :3]).as_euler('xyz', degrees=False)
    particles2d, weights = create_gaussian_particles([pose_cpu[0,-1], pose_cpu[1,-1], orientation[-1]], std, N)
    axis = torch.zeros(len(particles2d), 3)
    axis[:, 2] = 1
    particles_rot33 = kal.math.quat.rot33_from_angle_axis(angle=particles2d[:, -1][..., None], axis=axis)
    particles = torch.zeros(len(particles2d), 4, 4)
    particles[:, :3,:3] = particles_rot33 @ T_init[:3, :3]
    particles[-1,-1] = 1
    particles[:, :2, -1] = particles2d[:, :2]
    particles[:, 2, -1] = pose_cpu[2, -1]
    return particles.to(pose.device), weights.double().to(pose.device)



angle_diff = torch.pi / 180 * 30
trans_diff = 1
num_variants = 100
rot_perturbes = torch.zeros(num_variants, 3)
rot_perturbes[:, 1] = torch.linspace(-angle_diff, angle_diff, num_variants)
rot_perturbes[:3]
rot_perturbes_mat44 = torch.zeros(num_variants, 4,4)
rot_perturbes_mat44[:,-1,-1] = 1
for i, rp in enumerate(rot_perturbes):
    rot_perturbes_mat44[i, :3, :3] = torch.tensor(R.from_euler('xyz', rp, degrees=False).as_matrix()) 
trans_pertrube = torch.linspace(-trans_diff, trans_diff, 100) 

perturbations_mat44 = rot_perturbes_mat44.clone()[torch.randperm(num_variants)]
perturbations_mat44[:,0,-1] = trans_pertrube.clone()[torch.randperm(num_variants)]
perturbations_mat44[:,2,-1] = trans_pertrube.clone()[torch.randperm(num_variants)]

def predict_predefined(particles, odom, std_t, std_rot):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""
    delta_R = rot_perturbes_mat44[torch.randint(0,len(rot_perturbes_mat44), (len(particles),))]
    particles = particles @ delta_R
    moved = particles @ odom[None]
    # moved[:, :2, -1] += torch.randn(len(particles), 2)  * std_t
    return moved

def predict(particles, odom, std_t, std_rot, config):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""
    delta_R = torch.randn(len(particles), device=particles.device, dtype=particles.dtype)  * std_rot
    delta_R %= 2 * torch.pi
    _axis = torch.zeros(len(particles), 3, device=particles.device, dtype=particles.dtype)
    _axis[:, 1] = 1
    delta_R = kal.math.quat.rot33_from_angle_axis(angle=delta_R[..., None], axis=_axis)

    particles[:, :3,:3] = particles[:, :3,:3] @ delta_R
    moved = particles @ odom[None]
    if config.in_3D:
        moved[:, :3, -1] += torch.randn(len(particles), 3, device=particles.device, dtype=particles.dtype)  * std_t
    else:
        moved[:, :2, -1] += torch.randn(len(particles), 2, device=particles.device, dtype=particles.dtype)  * std_t
    return moved

    
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
    weights = weights * batches_loss + 1.e-5
    # weights[batches_mask.any(-1).logical_not()] = 1.e-5
    weights /= weights.sum()
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

    
def prepare_pf(first_pose_id, odoms, poses44, T_init, config, device, data_path):
    i_prev = first_pose_id

    print('prepare data')
    odoms = odoms.to(device)
    pose = (T_init @ poses44[0]).detach().to(device)
    if i_prev > 0:
        pose = pose @ combine_odoms(odoms, 0, i_prev)
    
    particles, weights = init_particles(pose, T_init, std=[config.init_std_x, config.init_std_y, 0.], N=config.num_particles)
    h = config.simulate.resolution.h
    w = config.simulate.resolution.w
    
    if config.visual_model.name == 'open_scene':
        scene_dir = os.path.join(data_path,  f"lseg_semantic")
    else:
        scene_dir = os.path.join(data_path,  f"{config.visual_model.name}_semantic")
    print(f"{scene_dir} is ised")

    aspect = config.simulate.resolution.w/config.simulate.resolution.h
    hfov = math.radians(config.simulate.hfov)

    return i_prev, odoms, pose, particles, weights, scene_dir, aspect, hfov

def load_octree_map(points, points_labels, scene_name, config):
    print("Create octree map")
    spc, spc_labels, scale =  points_features2octree(points.cuda(), points_labels, 
                                                    config.scene[scene_name].max_level, 
                                                    config.scene[scene_name].resolution)
    return (spc, spc_labels, scale)
        
def run(scene_name, config, first_pose_id=0, device='cuda', viser_server=None, octree_map=None, batch_size=1000, inital_particles=None):
    torch.set_grad_enabled(False)
    print("loading the data")
    (data_path, points, points_labels, poses44, T_init, 
      map_features_db, rgb_features_db, vis_scene_features) = load_data(scene_name, config, device)
    map_features_db_device = map_features_db.to(device)
    rgb_features_db_device = rgb_features_db.to(device)
    
    classes_ids = torch.arange(0, rgb_features_db.shape[0])
    odoms = estimate_odoms(poses44, T_init)

    if octree_map is None:
        print("Create octree map")
        points = points.cuda()
        spc, spc_labels, scale =  points_features2octree(points, points_labels, 
                                                        config.scene[scene_name].max_level, 
                                                        config.scene[scene_name].resolution,
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
        plot_data_points(points.cpu(), points_labels, colors_map, viser_server)
        plot_floor(scene_name, config, viser_server)
        # remap map colors to rgb_features_db  colors
        plot_map_nodes(vis_scene_features, map_features_db, point_hierarchy, spc_labels, pyramid, scene_name, scale, config, colors_map, viser_server, stride=2)
    print("run")

    i_prev, odoms, pose, particles, weights, scene_dir, aspect, hfov = prepare_pf(first_pose_id, odoms, poses44, T_init, config, device, data_path)    
    # rays can always be on GPU
    rays_o, rays_d, cam = create_rays(poses44[0].clone().detach().cuda(), config, T_init)
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
            pose = pose @ odom
            particles = predict(particles, odom, std_t=config.std_t, std_rot=config.std_rot, config=config)

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
                # TODO: description localizatoon:
                # TODO:     3. choose lowest possible lost for every particle
                estimate_loss(batches_mask, rays_ids_batched, rays_features, closest_rays_indx_batched, map_features_db_device, bs=config.rays_loss_batch_size, output_tensor=all_losses, output_slice=slice(st, end))

            weights = estimate_weights(weights, all_losses)
            mean_pose, p_rot33_mean = get_mean_pose(particles, weights)
            # resample
            samples = torch_stratified_resample(weights)
            particles = particles[samples]
            weights = weights[samples]

            # save trajectory
            estimated = torch.eye(4, device=particles.device)
            estimated[:3,:3] = p_rot33_mean
            estimated[:3, -1] = mean_pose
            estimated_poses.append(estimated.cpu().numpy())
            
            if config.global_localization.active:
                distance = np.linalg.norm(mean_pose.cpu().numpy() - pose[:3, -1].cpu().numpy())
                for precision_idx in range(len(precision_steps)):
                    if (distance <= config.global_localization.distances[precision_idx]) and (precision_steps[precision_idx] == 0):
                        precision_steps[precision_idx] = i + 1
                        print(f"Achieved {distance} m after {precision_steps[precision_idx] - first_pose_id} steps")
                if np.all(precision_steps > 0):
                    return poses44, estimated_poses, octree_map, precision_steps
                    
            
            # visualize
            if viser_server is not None:
                if config.vis.plot_particles:
                    plot_paricles(particles, hfov, aspect, viser_server, stride=config.vis.particles_stride)

                if config.vis.plot_gt:
                    if config.vis.demo: # only red frame
                        plot_camera_rgb(pose, None, hfov, aspect, viser_server)
                    else:
                        plot_camera_rgb(pose, rgb_image, hfov, aspect, viser_server)
                
                if config.vis.plot_estimated:
                    plot_raycast_view(rays_o, rays_d, estimated, spc, point_hierarchy, spc_labels, scale, 
                                      map_features_db, vis_scene_features,
                                      colors_map, viser_server, hfov, 
                                      config.simulate.resolution.w, config.simulate.resolution.h, [0,0,255])
                        
                    if config.vis.demo:
                        __estimated_semantic_pose = mean_pose
                        image1 =  cv2.imread(os.path.join(str(Path(scene_dir).parent.absolute()), 'rgb', f"00000{i}"[-6:]+'.png'))
                        plot_semantic_camera(image1, rgb_image, __estimated_semantic_pose, pose[:3,:3], [0,255,0], hfov, aspect, viser_server)
                    else:
                        plot_camera_frame('mean_cam', mean_pose, p_rot33_mean, [0,255,0], hfov, aspect, viser_server)
        
    return poses44, estimated_poses, octree_map, precision_steps
