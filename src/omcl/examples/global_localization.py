
import numpy as np
import os
from tqdm.auto import tqdm, trange
import viser
import torch
import hydra
from omegaconf import DictConfig
from omcl.models.pf import run
import clip
from omcl.utils.mics import load_data
from omcl.models.map import points_features2octree
from omcl.utils.plot import *
from omcl.utils.colors import d3_40_colors_rgb, generate_rgb_colors
import kaolin as kal
from omcl.models.pf import init_particles
import math
import trimesh


def read_gt_mesh(scene_name, config):
    print("LOADING MESH")
    gt_dir = os.path.join(os.path.expanduser('~'), config.paths.all_data, 'mp3d')
    scene_name_raw = scene_name.split('_')[0]
    mesh = trimesh.load(os.path.join(gt_dir, scene_name_raw, f"{scene_name_raw}_semantic.ply"))
    print("MESH IS LOADED")
    data_path = os.path.join(os.path.expanduser('~'), config.paths.all_data,
                         config.dataset.name, scene_name)
    dt = torch.from_numpy(np.loadtxt( os.path.join(data_path,  "poses.txt"))).float()[0, :3]
    mesh.vertices = np.array(mesh.vertices) - np.array([
                                                dt[0], 
                                                dt[1], # pose_init_global[1, -1], 
                                                dt[2] + config.dataset.simulation.camera_height # config.dataset.simulation.camera_height + pose_init_global[2, -1]
                                                ])
    
    cropped_mesh = crop_mesh_by_height(mesh, 
                                       config.dataset.scenes_config[scene_name].min_height - 0.5, 
                                       config.dataset.scenes_config[scene_name].max_height)
    return cropped_mesh


def crop_mesh_by_height(mesh, z_min, z_max):
    z_values = mesh.vertices[:, 2]
    valid_vertices_mask = (z_values >= z_min) & (z_values <= z_max)
    valid_vertex_indices = np.where(valid_vertices_mask)[0]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertex_indices)}
    valid_faces = []
    for face in mesh.faces:
        if all(v in index_map for v in face):
            new_face = [index_map[v] for v in face]
            valid_faces.append(new_face)
    if not valid_faces:
        return None  # No geometry left
    cropped_vertices = mesh.vertices[valid_vertex_indices]
    cropped_faces = np.array(valid_faces)
    # Handle vertex colors
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        cropped_colors = mesh.visual.vertex_colors[valid_vertex_indices]
        visual = trimesh.visual.ColorVisuals(vertex_colors=cropped_colors)
    else:
        visual = None
    cropped_mesh = trimesh.Trimesh(
        vertices=cropped_vertices,
        faces=cropped_faces,
        visual=visual,
        process=False
    )
    return cropped_mesh


def encode_text(encoder, text_list, device='cuda'):
    with torch.no_grad():
        text = clip.tokenize(text_list)  
        text = text.to(device)
        text_features = encoder.encode_text(text).half()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
        return text_features


def detect_floor_walls(map_features_db, map_points_labels, text_encoder, th=0.9):
    with torch.no_grad():
        floor_wall_features = encode_text(text_encoder, ['floor', 'wall'], 'cpu').to(map_features_db.device,)
        surrounding_mask = (torch.nn.functional.cosine_similarity(
            map_features_db[map_points_labels][..., None, :].cuda(), floor_wall_features[None,...].cuda(), dim=-1
            ).max(-1)[0] > th).logical_not().cpu()
        floor_mask = (torch.nn.functional.cosine_similarity(
            map_features_db[map_points_labels][..., None, :].cuda(), floor_wall_features[0][None,...].cuda(), dim=-1
            ).max(-1)[0] > th).cpu()
        return floor_mask, surrounding_mask


def get_prompt_locations(floor_points, surrounding_points, map_features_db, surrounding_labels, prompt_encodings, 
                         th=0.9, step=10, prompt_radious=2., map_prompt_match_th=0.5, min_matches=500):
    with torch.no_grad():
        bs = 10000
        map_features = map_features_db[surrounding_labels].cuda()
        prompt_enc = prompt_encodings[None, ...].cuda()
        map_prompt_sims = []
        for b_id in range(math.ceil(len(surrounding_labels) / bs)):
            st = b_id * bs
            end = min((b_id+1)*bs, len(surrounding_labels))
            sims = (torch.nn.functional.cosine_similarity(map_features[st:end, None, :], prompt_enc, -1) > th).cpu()
            map_prompt_sims.append(sims)
        map_prompt_sims = torch.cat(map_prompt_sims, dim=0).cuda()
        point_scores = []
        sparse_floor_points = floor_points[::step].cuda()
        surrounding_points = surrounding_points.cuda()
        for point in tqdm(sparse_floor_points):
            close_points_mask = (surrounding_points - point).norm(2, -1) < prompt_radious
            # sum all surroinding points and check there are more that min_matches per prompt
            score = (map_prompt_sims[close_points_mask].sum(0) > min_matches).sum() / map_prompt_sims.shape[-1]
            point_scores.append(score)
        scores = torch.stack(point_scores)
        return sparse_floor_points[scores > map_prompt_match_th].cpu()


def make_particles_at_locations(pose0, locations, num_particles, rot_std=1.):
    particles = []
    for i in range(num_particles):
        p = pose0.detach().clone()
        # p[:3, :3] = p[:3, :3] @ get_random_rot(0, 0.5, 0., device=p.device, dtype=p.dtype)
        p[:2, -1] = locations[torch.randint(len(locations), (1,))][0][:2]
        particles.append(init_particles(p, std=[0, 0, rot_std], N=1)[0]) # apply random rotation and store the particle
    particles = torch.cat(particles, 0)
    return particles



@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="mp3d_config",
)
def main(config: DictConfig):
    config.vis.particles_stride = 2
    viser_server = viser.ViserServer()
    scene = 'JmbYfDe2QKZ_2'
    scene_config = config.dataset.scenes_config[scene]
    aspect = config.dataset.simulation.resolution.w / config.dataset.simulation.resolution.h
    hfov = math.radians(config.dataset.simulation.hfov)
    plot_floor(scene, config, viser_server)
    # initial pose for localization without prompts
    pose0_id = 150
    # Prepare prompts
    text_encoder, _ = clip.load("ViT-B/32", device='cpu', jit=False)
    prompt1 =  ['toilet', 'mirro', 'towel', 'sink']
    prompt1_features = encode_text(text_encoder, prompt1, device='cpu')
    pose1_id = 40
    prompt2 = ['table', 'chair', 'picture', 'door', 'tv monitor']
    prompt2_features = encode_text(text_encoder, prompt2, device='cpu')
    pose2_id = 393
    # Load Map
    (data_path, points, points_labels, poses44, _, _, 
      map_features_db, _, vis_scene_features) = load_data(scene, config, 'cuda')
    scene_mesh = read_gt_mesh(scene, config)
    viser_mesh = viser_server.scene.add_mesh_trimesh('mesh', scene_mesh)
    print("Create octree map")
    points = points.cuda()
    spc, spc_labels, scale =  points_features2octree(points, points_labels, 
                                                    scene_config.max_level, 
                                                    scene_config.resolution,
                                                    vectorize=True)
    points = points.cpu()
    octree_map = (spc, spc_labels, scale)
    point_hierarchy = kal.ops.spc.generate_points(spc.octrees, spc.pyramids, spc.exsum)
    pyramid = spc.pyramids[0]
    
    # visualize map
    sem_colors = np.concatenate([d3_40_colors_rgb, generate_rgb_colors(config.vis.num_colors)], axis=0)
    vis_ids = (map_features_db[spc_labels].cuda() @ vis_scene_features.T).argmax(-1).cpu()
    plot_data_points(points.cpu(), points_labels, sem_colors, viser_server)
    plot_map_nodes(vis_scene_features, map_features_db, point_hierarchy, spc_labels, pyramid, scale, scene_config, sem_colors, viser_server, stride=3, visible=False)
    # detect floor
    floor_mask, surrounding_mask = detect_floor_walls(map_features_db=map_features_db, map_points_labels=points_labels, text_encoder=text_encoder, th=0.9)
    floor_points = points[floor_mask]
    surrounding_points = points[surrounding_mask]
    surrounding_labels = points_labels[surrounding_mask]
    _ = viser_server.scene.add_point_cloud(
            name="floor_points",
            points=floor_points.numpy(),
            colors=sem_colors[vis_ids[floor_mask]],
            point_size=scene_config.resolution,
            visible=False)
    _ = viser_server.scene.add_point_cloud(
            name="surrounding",
            points=surrounding_points.numpy(),
            colors=sem_colors[vis_ids[surrounding_mask]],
            point_size=scene_config.resolution,
            visible=False)
    
    
    def run_global_localization(prompt, prompt_features, pose_id, config, prompt_color):
        # get access to the constant outer variables
        nonlocal floor_points, surrounding_points, surrounding_labels, map_features_db
        nonlocal hfov, aspect, viser_server, scene, scene_config
        nonlocal poses44, octree_map
        printRed("The camera pose is ready.")
        plot_camera_rgb(poses44[pose_id], None, hfov, aspect, viser_server, config)
        printGreen(f"The corresponding prompt is {prompt}.")
        
        possible_coordinates = get_prompt_locations(floor_points=floor_points,
                                    surrounding_points=surrounding_points,
                                    map_features_db=map_features_db, 
                                    surrounding_labels=surrounding_labels, 
                                    prompt_encodings=prompt_features, 
                                    th=0.9,
                                    map_prompt_match_th=0.7
                                    )
        printYellow(f"Initialization area is {possible_coordinates.shape[0]} voxels.")
        viser_server.scene.add_point_cloud(
            name="prompt area",
            points=possible_coordinates + torch.tensor([0., 0., 0.2]),
            colors=prompt_color,
            point_size=scene_config.resolution * 10,
            visible=True)
        printYellow("Press Enter to continue.")
        input()
        # create particles in the prompt area
        particles = make_particles_at_locations(poses44[0], possible_coordinates, config.num_particles)
        plot_paricles(particles, hfov, aspect, viser_server, config, visible=True)
        
        printGreen(f"Particles are initialized. Press Enter to continue.")
        input()
        
        
        _, _, _, _ = run(scene, config, 
                        first_pose_id=pose_id, # different initial  pose
                        device='cuda', 
                        viser_server=viser_server, 
                        octree_map=octree_map,
                        batch_size=config.particles_batch_size,
                        inital_particles=particles,
                        global_localization=True, # activate global localization,
                        plot_map=False
                        )
    
    # run interactive demonstration
    printGreen("We firstly demonstrate global localization without prompts. Open the viewer and press Enter to continue.")
    input()
    # create random particles around the whole map
    print(f"Initialization area is {floor_points.shape[0]} voxels")
    particles = make_particles_at_locations(poses44[0], floor_points[::10], config.num_particles)
    
    
    plot_paricles(particles, hfov, aspect, viser_server, config, visible=True)
    plot_camera_rgb(poses44[pose0_id], None, hfov, aspect, viser_server, config)
    
    printCyan("Particles are initialized. Press Enter to continue.")
    input()
    _, _, _, _ = run(scene, config,
                     first_pose_id=pose0_id, # define the first initial pose 
                     device='cuda', viser_server=viser_server, 
                    octree_map=octree_map,
                    batch_size=config.particles_batch_size,
                    inital_particles=particles, 
                    global_localization=True, # activate global localization
                    plot_map=False
                    )
    printYellow("Initialization is finished. Press Enter to continue with the next demonstration.")
    input()
    
    
    run_global_localization(prompt1, prompt1_features, pose1_id, config, prompt_color=(0,255,0))
    printYellow("Initialization is finished. Press Enter to continue with the next demonstration.")
    input()
    
    
    run_global_localization(prompt2, prompt2_features, pose2_id, config, prompt_color=(0,255,0))
    printGreen("Global Localization demonstration is finished. Press Enter to exit.")
    input()
    
    
if __name__ == '__main__':
    main()
