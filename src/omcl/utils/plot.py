import torch
from scipy.spatial.transform import Rotation as R

def plot_floor(scene_name, config, viser_server):
    side_length = 2**config.scene[scene_name].max_level * config.scene[scene_name].resolution
    _ = viser_server.scene.add_grid('floor', width=side_length, height=side_length, width_segments=50, position=(0,0, -1.5),
                                    visible=config.vis.floor)


opengl_mat = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0,  -1]], dtype=torch.float32)

def mp3d_pose2viser_wxyz(pose):
    quat_camera =  R.from_matrix(pose[:3,:3].cpu().float() @ opengl_mat).as_quat()
    return (quat_camera[-1], *quat_camera[:-1])
