import numpy as np
import torch
import clip
import os
import yaml


opengl_mat = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0,  -1]], dtype=torch.float32) # for visualization


def get_sim_cam_mat_with_fov(h, w, fov):
    # https://github.com/vlmaps/vlmaps/blob/master/vlmaps/utils/mapping_utils.py
    # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html#gsc.tab=0
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def make_intrinsics(config, hfov=90):
    hfov = hfov * np.pi / 180.
    W = max(config.dataset.simulation.resolution.w, config.dataset.simulation.resolution.h)
    K = np.array([
    [1 / np.tan(hfov / 2.), 0., 0., 0.],
    [0., 1 / np.tan(hfov / 2.), 0., 0.],
    [0., 0.,  1, 0],
    [0., 0., 0, 1]])

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,W))
    xs = xs.reshape(1,W,W)
    ys = ys.reshape(1,W,W)

    return {'inv_K': np.linalg.inv(K), 'xs': xs, 'ys': ys, 'W': W}


def depth2pc(depth_img, intrinsics, min_depth=0.1, max_depth=10):
    """
    Return 3xN array and the mask of valid points in [min_depth, max_depth]
    """
    depth_mask = np.logical_and(depth_img > min_depth, depth_img < max_depth)
    
    depth = depth_img.reshape(1, intrinsics['W'], intrinsics['W'])
    # x = (intrinsics['x'] * depth_img)[depth_mask]
    # y = (intrinsics['y'] * depth_img)[depth_mask]
    # z = depth_img[depth_mask]
    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((intrinsics['xs'] * depth , intrinsics['ys'] * depth, -depth, np.ones(depth.shape)))
    xys = xys[:, depth_mask]
    # xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(intrinsics['inv_K'], xys)
    assert np.all(xy_c0[-1, :] == 1)    # just check the values
    return xy_c0[:3, :].astype(np.float32), depth_mask  #TODO: is float 32 enough for kaolin coordinates?



def encode_text(encoder, text_list, device='cuda'):
    text = clip.tokenize(text_list)  
    text = text.to(device)
    text_features = encoder.encode_text(text).half()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
    return text_features


def get_scene_labels(scene_dir, config):
    with open(os.path.join(scene_dir, 'classes_map.yaml'), 'r') as classes_file:
        classes_map = yaml.safe_load(classes_file)
    labels = []
    ids = []
    ids2labels = {}
    for id, label in classes_map.values():
        if label == "" or id == config.dataset.simulation.ignore_indx:
            continue
        if id not in ids:
            ids.append(id)
            labels.append(label)
            ids2labels[id] = label
    sorted_indexes = np.argsort(ids)
    labels_ids = np.array(ids)[sorted_indexes]
    labels = [labels[i] for i in sorted_indexes]
    return labels, labels_ids, ids2labels

def get_scene_features(scene_dir, config):
    # used for visualization to colorize semantic
    labels, _, _ = get_scene_labels(scene_dir, config)
    with torch.no_grad():
        text_encoder, _ = clip.load("ViT-B/32", device='cpu', jit=False)
        labels_features = encode_text(text_encoder, labels, 'cpu')
    return labels_features
