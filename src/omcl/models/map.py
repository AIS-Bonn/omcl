import kaolin as kal
import torch
from tqdm.auto import trange, tqdm


def points_features2octree(points, points_labels, max_level, resolution, vectorize=False):
    num_nodes = 2**max_level
    node_size = 2 / num_nodes
    assert points.is_cuda, "`points` should be on gpu otherwise spc raises exceptions"
    length, scale = get_length_scale(max_level, resolution)

    # points_mask = torch.logical_and(
    #     torch.logical_and(points[:, 0] < length / 2, points[:, 0] > -length / 2),
    #     torch.logical_and(points[:, 1] < length / 2, points[:, 1] > -length / 2)
    #     )
    # points = points[points_mask]
    # points_labels = points_labels[points_mask.to(points_labels.device)]
    
    points = points / scale
    print(f"Node size: {node_size*scale}")
    # create octree structure
    spc = kal.ops.conversions.unbatched_pointcloud_to_spc(points, max_level)
    octree = spc.octrees
    max_level, pyramids, exsum = kal.ops.spc.scan_octrees(
        octree, torch.tensor([len(octree)], dtype=torch.int32, device='cpu'))
    # find corresponding points_labels for octree nodes TODO: average values in nodes?
    q_points = kal.ops.spc.quantize_points(points, max_level)
    pidx = kal.ops.spc.unbatched_query(spc.octrees, exsum, query_coords=q_points, level=max_level, with_parents=False)
    points_indixes = pidx - pyramids[0,1,-2] # To make min == 0
    assert points.shape[0] == points_indixes.shape[0]

    spc_labels = torch.zeros(points_indixes.max() - points_indixes.min()+1, dtype=points_labels.dtype)
    if vectorize:
        spc_labels[points_indixes - points_indixes.min()] = points_labels
    else:
        # rays iteration
        for j, pindx in enumerate(points_indixes):
            spc_labels[pindx] = points_labels[j]
    
    return spc, spc_labels.int(), scale


def increment_map(map_pts, map_features, new_pts, new_featurs, map_n_means, max_level, resolution, device):
    length, scale = get_length_scale(max_level, resolution)
    
    nodes_coords, points_indixes = merge_points_octree(map_pts, new_pts, scale=scale, max_level=max_level, res_device=device)
    if new_featurs is None:
        return nodes_coords, torch.tensor([]), torch.tensor([])
    # sum all features
    average_features = torch.zeros((nodes_coords.shape[0], new_featurs.shape[-1]), dtype=map_features.dtype, device=device)
    all_features = torch.cat((map_features * map_n_means[..., None], new_featurs))
    average_features.index_add_(0, points_indixes, all_features)
    denom_n = torch.zeros(nodes_coords.shape[0], dtype=map_n_means.dtype, device=device)
    denom_n[points_indixes[:map_features.shape[0]]] = map_n_means - 1
    denom_n.index_add_(0, points_indixes, torch.ones(all_features.shape[0], dtype=map_n_means.dtype, device=device))   
    average_features /= denom_n[..., None]
    return nodes_coords, average_features / average_features.norm(dim=-1, keepdim=True), denom_n

def increment_map_reduce(map_pts, map_features, new_pts, new_featurs, map_n_means, max_level, resolution, device):
    length, scale = get_length_scale(max_level, resolution)
    
    nodes_coords, points_indixes = merge_points_octree(map_pts, new_pts, scale=scale, max_level=max_level, res_device=device)
    if new_featurs is None:
        return nodes_coords, torch.tensor([]), torch.tensor([])
    # sum all features
    average_features = torch.zeros((nodes_coords.shape[0], new_featurs.shape[-1]), dtype=map_features.dtype, device=device)
    # add old features
    old_indexes = points_indixes[:len(map_features)]
    average_features.index_add_(0, old_indexes, map_features * map_n_means[..., None])
    # reduce new features by averaging
    new_indexes = points_indixes[len(map_features):]
    average_features_new = torch.ones_like(average_features) # ones to make normalization possible
    average_features_new.index_reduce_(dim=0, index=new_indexes, source=new_featurs, reduce='mean', include_self=False)
    average_features_new = average_features_new
    # Per-node flag: did this node receive any new points?
    has_new = torch.zeros(nodes_coords.shape[0], dtype=torch.bool, device=device)
    has_new.index_fill_(0, new_indexes, True)

    average_features = average_features + average_features_new * has_new[..., None]    
    denom_n = torch.zeros(nodes_coords.shape[0], dtype=map_n_means.dtype, device=device)
    denom_n.index_add_(0, old_indexes, map_n_means)
    denom_n = denom_n + has_new
    average_features /= denom_n[..., None]
    return nodes_coords, average_features, denom_n


def merge_submaps(submaps, max_level, resolution, device):
    if len(submaps) == 1:
        return submaps[0][0], submaps[0][1], submaps[0][2]
    map_points, map_features, map_n_means = submaps[0]
    map_points = map_points.to(device=device)
    map_features = map_features.to(device=device)
    map_n_means = map_n_means.to(device=device)
        
    length, scale = get_length_scale(max_level, resolution)
    
    for sm in tqdm(submaps[1:]):
        sm_points, sm_features, sm_n_means = sm
        sm_points = sm_points.to(device=map_points.device)
        sm_features = sm_features.to(device=map_features.device)
        sm_n_means = sm_n_means.to(device=map_n_means.device)
        
        nodes_coords, points_indixes = merge_points_octree(map_points, sm_points, scale=scale, max_level=max_level, res_device=device)
        map_points = nodes_coords
        
        if sm_features.numel() == 0:
            continue
        
        # sum all features
        average_features = torch.zeros((nodes_coords.shape[0], sm_features.shape[-1]), dtype=map_features.dtype, device=device)
        all_features = torch.cat((map_features * map_n_means[..., None], sm_features * sm_n_means[..., None] )).to(device=device)
        average_features.index_add_(0, points_indixes, all_features)
        # sum denominators
        denom_n = torch.zeros(nodes_coords.shape[0], dtype=map_n_means.dtype, device=device)
        denom_n.index_add_(0, points_indixes[:map_features.shape[0]], map_n_means.to(device))
        denom_n.index_add_(0, points_indixes[map_features.shape[0]:], sm_n_means.to(device))
        # breakpoint()
        # denom_n[points_indixes[:map_features.shape[0]]] += map_n_means.to(device)
        # denom_n[points_indixes[map_features.shape[0]:]] += sm_n_means.to(device)
        # denom_n[points_indixes[:map_features.shape[0]]] = map_n_means - 1
        # denom_n.index_add_(0, points_indixes, torch.ones(all_features.shape[0], dtype=map_n_means.dtype, device=device))  
         
        average_features /= denom_n[..., None]
        
        map_features = average_features
        map_n_means = denom_n
        
    return map_points, map_features, map_n_means


def init_map(config, device, feature_size):
    map_points = torch.tensor([], device=device)
    map_features = torch.tensor([], dtype=torch.float16, device=device).reshape(-1, feature_size)
    map_n_means = torch.tensor([], dtype=int, device=device)
    return map_points, map_features, map_n_means

def get_length_scale(max_level, resolution):
    length = 2**(max_level) * resolution
    scale = length / 2
    return length, scale

def merge_points_octree(points_1, points_2, scale, max_level, res_device):
    points = torch.cat((points_1, points_2)).cuda() / scale
    spc = kal.ops.conversions.unbatched_pointcloud_to_spc(points, max_level, features=None)
    octree = spc.octrees
    max_level, pyramids, exsum = kal.ops.spc.scan_octrees(
        octree, lengths=torch.tensor([len(octree)], dtype=torch.int32, device='cpu'))   # lengths must be a cpu tensor
    # get sorted octree coords
    node_coords = kal.ops.spc.generate_points(octree, pyramids, exsum)[pyramids[0,1,-2]:]
    num_nodes = 2**max_level
    node_size = 2 / num_nodes
    
    # get nodes idx per each point
    pidx = kal.ops.spc.unbatched_query(spc.octrees, spc.exsum, 
                                        query_coords=kal.ops.spc.quantize_points(points, max_level),
                                        level=max_level, with_parents=False
                                        )
    points_indixes = pidx - spc.pyramids[0,1,-2] # To make min == 0
    assert points.shape[0] == points_indixes.shape[0]
        
    return ((-1. + node_coords*node_size + 0.5*node_size) * scale).to(res_device), points_indixes.to(res_device)

