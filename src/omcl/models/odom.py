import torch
from torch import Tensor
import kaolin as kal


def combine_odoms(odoms, start, end):
    mat = kal.math.quat.euclidean_identity(1, device=odoms.device)[0]
    for odom in odoms[start: end]:
        mat = mat @ odom
    return mat


def estimate_odoms(poses44):
    # estimate all odoms
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
    mean_t = (particles[:, :3, -1] * weights[:, None]).sum(dim=0)

    q = kal.math.quat.quat_from_rot33(particles[:, :3, :3])   # [N,4]

    q = q.clone()
    ref = q[0]
    signs = torch.sign((q * ref[None]).sum(dim=-1, keepdim=True))
    signs[signs == 0] = 1.0
    q = q * signs

    A = (weights[:, None, None] * (q[:, :, None] @ q[:, None, :])).sum(dim=0)
    eigvals, eigvecs = torch.linalg.eigh(A)
    q_mean = eigvecs[:, -1]
    q_mean = q_mean / torch.linalg.norm(q_mean)
    q_mean = kal.math.quat.quat_unit_positive(q_mean[None])[0]

    R_mean = kal.math.quat.rot33_from_quat(q_mean[None])[0]
    return mean_t, R_mean
