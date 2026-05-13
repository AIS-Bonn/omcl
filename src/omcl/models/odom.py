import torch
from torch import Tensor
import kaolin as kal


def combine_odoms(odoms, start, end):
    mat = kal.math.quat.euclidean_identity(1, device=odoms.device)[0]
    for odom in odoms[start: end]:
        mat = mat @ odom
    return mat


def estimate_odoms(poses44, T_init):
    # estimate all odoms
    poses44 = T_init @ poses44
    # Ti = Ti-1 @ T - > T = Ti-1 ` @ Ti
    odoms = torch.cat((kal.math.quat.euclidean_identity(1, device=poses44.device), euclidean_inverse(poses44[:-1]) @ poses44[1:]))
    return odoms


def euclidean_inverse(x: Tensor) -> Tensor:
    """https://kaolin.readthedocs.io/en/latest/modules/kaolin.math.quat.html#kaolin.math.quat.euclidean_inverse"""
    mat = kal.math.quat.euclidean_identity(len(x), device=x.device)
    inv_rot = kal.math.quat.rot33_inverse(kal.math.quat.euclidean_rotation_matrix(x))
    translations = kal.math.quat.euclidean_translation_vector(x).squeeze()
    mat[..., :3, :3] = inv_rot
    for i in range(len(inv_rot)):
        inv_trans = -inv_rot[i] @ translations[i]
        mat[i, :3, 3] = inv_trans
    return mat


def get_mean_pose(particles, weights):
    mean_pose = (particles[:, :3, -1]*weights[..., None]).sum(0)
    
    # p_angles, p_axis = kal.math.quat.angle_axis_from_rot33(particles[..., :3, :3])
    # mean_rot_vec = ((p_angles * p_axis)*weights[..., None]).sum(0)
    # mean_angle = mean_rot_vec.norm(closest_rays_indx_batched.shape)
    # mean_axis = mean_rot_vec / mean_angle 
    # print(mean_angle, mean_axis)
    # p_rot33_mean = kal.math.quat.rot33_from_angle_axis(mean_angle, mean_axis)[0]
    p_quats = kal.math.quat.quat_from_rot33(particles[..., :3, :3])
    mean_p_quat = kal.math.quat.quat_unit_positive((p_quats * weights[..., None]).sum(0))
    p_rot33_mean = kal.math.quat.rot33_from_quat(mean_p_quat)
    return mean_pose, p_rot33_mean
