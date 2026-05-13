import kaolin as kal
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def get_sim_cam_mat_with_fov(h, w, fov):
    # https://github.com/vlmaps/vlmaps/blob/master/vlmaps/utils/mapping_utils.py
    # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html#gsc.tab=0
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def create_rays(pose, height, width, fov, T_init):
    pose2 = T_init.to(device=pose.device, dtype=pose.dtype) @ pose.clone().detach()
    pose2[:3,:3] = pose[:3,:3].clone().detach()
    calib_mat = get_sim_cam_mat_with_fov(h=height, w=width, fov=fov)
    print(calib_mat)
    print(calib_mat[0][0])
    print(calib_mat[1][1])
    cam=kal.render.camera.Camera.from_args(
        view_matrix=pose2,
        # fov=90,
        focal_x=calib_mat[0][0],
        # x0=config.simulate.cam_calib_mat[2],
        focal_y=calib_mat[1][1],
        # y0=config.simulate.cam_calib_mat[5],
        width=width,
        height=height,
        device=pose.device)
    
    rays_o, rays_d = kal.render.camera.generate_pinhole_rays(camera=cam)

    return rays_o, rays_d , cam

def transform_rays(rays_o, rays_d, pose):
    rays_d_new = rays_d @ pose[:3,:3].T
    rays_o_new = rays_o + pose[:3,-1]
    return rays_o_new, rays_d_new

def transform_rays_batched(rays_o, rays_d, batched_poses):
    rays_d_new = rays_d @ batched_poses[:, :3, :3].mT
    rays_o_new = rays_o + batched_poses[:, :3,-1][:, None, ...]
    return rays_o_new, rays_d_new

def make_sem_image(closest_rays_indx, rays_labels):
    raise NotImplementedError("fix dimensions")
    sem_image = torch.zeros((1080*1080), dtype=torch.int)
    sem_image[closest_rays_indx] = rays_labels
    return sem_image


# def init_rays(pose, rot_init, pose_init, config):
#     position = rot_init @ (pose[:3] - pose_init)
#     rotation = R.from_matrix(rot_init @ R.from_quat(pose[3:]).as_matrix()).as_quat()
#     first_pose = torch.eye(4, dtype=float)
#     first_pose[:3,:3] = torch.from_numpy(R.from_quat(rotation).as_matrix()).float()
#     first_pose[:3, -1] = position
#     rays_o, rays_d, cam = create_rays(first_pose, config)
#     return rays_o, rays_d, cam

def ray_trace_pose(rays_o_pose, rays_d_pose, spc, point_hierarchy, spc_features):
    ray_indx, point_indx, depth = kal.render.spc.unbatched_raytrace(
        octree=spc.octrees, 
        point_hierarchy=point_hierarchy,
        pyramid=spc.pyramids[0],
        exsum=spc.exsum,
        origin=rays_o_pose.cuda() + torch.tensor([1e-7,1e-7,1e-7], device='cuda'),
        direction=rays_d_pose.cuda(),
        level=spc.max_level,
        with_exit=True)
    # assert len(ray_indx) > 0, f"that same error {ray_indx.max().cpu()}, {point_indx.max().cpu()}, {depth.max().cpu()}"
    # assert len(point_indx) > 0, f"that same error {ray_indx.max().cpu()}, {point_indx.max().cpu()}, {depth.max().cpu()}"
    # assert len(depth) > 0, f"that same error {ray_indx.max().cpu()}, {point_indx.max().cpu()}, {depth.max().cpu()}"
    if len(ray_indx) <= 0:
        return [], [], []
    # print(ray_indx.requires_grad)
    # print(point_indx.requires_grad)
    # print(depth.requires_grad)
    
    mask = kal.render.spc.mark_pack_boundaries(ray_indx)
    
    closest_rays_indx = ray_indx[mask]
    closest_points_indx = point_indx[mask]
    rays_ids = spc_features[closest_points_indx.cpu()- spc.pyramids[0,1,-2]]
    return closest_rays_indx, closest_points_indx, rays_ids

def get_features_masks(input_data_mask, features_image, encodings, classes_ids):
    obs_features = features_image.reshape(features_image.shape[0]*features_image.shape[1], -1)
    prod = obs_features @ encodings.detach().T  # both assumed normalized before
    # reshape to image size with ids per pixel
    cls_ids = prod.argmax(-1).reshape(features_image.shape[:2])
    # get mask for each cls
    observation_masks = cls_ids[None] == classes_ids[..., None, None]
    return torch.logical_and(observation_masks, input_data_mask[None].to(observation_masks.device))

def get_input_masks(input_data_mask, features_image, encodings, classes_ids, num_samples):
    observation_masks = get_features_masks(input_data_mask, features_image, encodings, classes_ids)
    res = [] # i,j pixels
    for i in range(len(observation_masks)):
        ids = observation_masks[i].argwhere() # TODO: APPLY for all masks at once
        if len(ids) > 0:
            res.append(ids[torch.randint(low=0, high=len(ids), size=(num_samples,))]) # TODO: randint in advace all
    if len(res) == 0:   # could be because of all zeros in input_data_mask
        print("no masks")
        return None
    return torch.cat(res)


def get_input_masks_cropped(input_data_mask, features_image, encodings, classes_ids, num_samples):
    observation_masks = get_features_masks(input_data_mask, features_image, encodings, classes_ids)
    res = [] # i,j pixels
    for i in range(len(observation_masks)):
        ids = observation_masks[i].argwhere() # TODO: APPLY for all masks at once
        if len(ids) > 0:
            res.append(ids[torch.randperm(len(ids))[:min(len(ids), num_samples)]])
    return torch.cat(res)

def ray_trace_pose_batched(rays_o_pose, rays_d_pose, spc, point_hierarchy, spc_features):
    closest_rays_indx, closest_points_indx, rays_ids = ray_trace_pose(rays_o_pose.reshape(-1,3), rays_d_pose.reshape(-1,3), spc, point_hierarchy, spc_features)

    batches_mask = torch.zeros(rays_o_pose.shape[0], rays_o_pose.shape[1], dtype=bool, device=rays_o_pose.device)
    batches_mask.view(-1)[closest_rays_indx] = True
    # ALL_RAYS_IDS = (torch.arange(rays_o_pose.shape[1], device='cuda', dtype=int)).repeat(rays_o_pose.shape[0],1)
    # rays_ids_batched = torch.zeros(batches_mask.shape, dtype=torch.int)
    # rays_ids_batched[batches_mask] = rays_ids.int()
    # rays_ids_batched = masked_tensor(rays_ids_batched, batches_mask)
    # _, closest_rays_indx_batched = torch.nonzero(batches_mask, as_tuple=True)
    return torch.nonzero(batches_mask, as_tuple=True)[1], closest_points_indx, rays_ids, batches_mask
    # return ALL_RAYS_IDS[batches_mask], closest_points_indx, rays_ids, batches_mask

def ray_trace_particles():
    pass