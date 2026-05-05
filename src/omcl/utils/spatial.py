import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor
import kaolin as kal


def voxel_down_sample(points, voxel_size):
    is_o3d = isinstance(points, o3d.geometry.PointCloud)
    if is_o3d:
        frame = points
    else:
        channel_first = points.shape[0] == 3
        if channel_first:
            pts = points.T
        else:
            pts = points
        frame = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    frame, _, trace = frame.voxel_down_sample_and_trace(voxel_size, frame.get_min_bound(), frame.get_max_bound())
    # trace_f = lambda i, trace=trace: trace[i][random.randint(0, len(trace[i]) - 1)]
    trace_f = lambda i, trace=trace: trace[i][0]
    idx = np.array(list(map(trace_f, range(len(trace)))))
    
    if is_o3d:
        channel_first = False
        points = np.asarray(points.points)
    
    if channel_first:
        return points[:, idx], idx
    else:
        return points[idx], idx
    
    
def crop_height(local_points, pose, height, min_height):
    global_points = pose[:3, :3] @ local_points + pose[:3, -1][..., None]
    height_mask1 = global_points[-1,:] < height
    height_mask2 = global_points[-1,:] > min_height
    height_mask = torch.logical_and(height_mask1, height_mask2)
    return local_points[:, height_mask], height_mask


