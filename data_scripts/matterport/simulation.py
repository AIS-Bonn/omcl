# Creates GT RGB, DEPTH and SEMANTIC images
# original code: https://github.com/vlmaps/vlmaps/tree/master
import os
import subprocess
from pathlib import Path
import shutil
import gdown
from typing import Dict, List, Union
import cv2
# import sys
# sys.setrecursionlimit(10)
import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from omcl.utils.colors import d3_40_colors_rgb
import yaml 


def plot_semantic(path, semantic_obs=np.array([])):
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata(semantic_obs.flatten().astype(np.uint8))   # somehow colors ignore_indx but it doesn't matter how
        semantic_img = semantic_img.convert("RGBA")
        semantic_img.save(path)
        np.save(str(path)[:-3] + 'npy',semantic_obs)

def make_sensor_spec(
    uuid: str,
    sensor_type: str,
    h: int,
    w: int,
    position: Union[List, np.ndarray],
    orientation: Union[List, np.ndarray] = None,
) -> Dict:
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = uuid
    sensor_spec.sensor_type = sensor_type
    sensor_spec.resolution = [h, w]
    sensor_spec.position = position
    if orientation:
        sensor_spec.orientation = np.array(orientation)

    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return sensor_spec


def get_obj2cls_dict(sim: habitat_sim.Simulator, config) -> Dict:
    """
    get the dictionary mapping from object id to semantic id
    """
    scene = sim.semantic_scene
    obj2cls = {int(obj.id.split("_")[-1]): (obj.category.index(), obj.category.name()) for obj in scene.objects}
    object_ids = []
    classes_ids_labels = []
    for k, v in obj2cls.items():
        if v[0] == -1:  # (-1, '')
            # new_obj2cls[k] = (config.ignore_indx, v[1]) # empry string '', ingore_indx for broken data
            object_ids.append(k)
            classes_ids_labels.append((config.ignore_indx, v[1]))   # empry string '', ingore_indx for broken data
        else:
            # new_obj2cls[k] = v
            object_ids.append(k)
            classes_ids_labels.append(v)
    unique_sorted, inverse = np.unique([v[0] for v in classes_ids_labels], return_inverse=True)
    id_counter = 0
    for i in range(len(unique_sorted)):
        if unique_sorted[i] >= 0:
            unique_sorted[i] = id_counter
            id_counter += 1
    
    for i in range(len(classes_ids_labels)):
        inv_id = inverse[i]
        classes_ids_labels[i] = [int(unique_sorted[inv_id]), classes_ids_labels[i][1]]
    
    new_obj2cls = {}
    for obj_id, cls_id_label in zip(object_ids, classes_ids_labels):
        new_obj2cls[obj_id] = cls_id_label
    return new_obj2cls


def make_cfg(settings: Dict) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.scene_dataset_config_file = settings['mp3d_scene_config_path']

    sensor_spec = []
    back_rgb_sensor_spec = make_sensor_spec(
        "back_color_sensor",
        habitat_sim.SensorType.COLOR,
        settings["height"],
        settings["width"],
        [0.0, settings["sensor_height"], 1.3],
        orientation=[-np.pi / 8, 0, 0],
    )
    sensor_spec.append(back_rgb_sensor_spec)

    if settings["color_sensor"]:
        rgb_sensor_spec = make_sensor_spec(
            "color_sensor",
            habitat_sim.SensorType.COLOR,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(rgb_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = make_sensor_spec(
            "depth_sensor",
            habitat_sim.SensorType.DEPTH,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = make_sensor_spec(
            "semantic_sensor",
            habitat_sim.SensorType.SEMANTIC,
            settings["height"],
            settings["width"],
            [0.0, settings["sensor_height"], 0.0],
        )
        sensor_spec.append(semantic_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_spec
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=settings["move_forward"]),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["turn_right"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["turn_right"])
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def cvt_obj_id_2_cls_id(semantic: np.ndarray, obj2cls: Dict) -> np.ndarray:
    h, w = semantic.shape
    semantic = semantic.flatten()
    u, inv = np.unique(semantic, return_inverse=True)
    return np.array([obj2cls[x][0] for x in u])[inv].reshape((h, w))


def save_obs(
    root_save_dir: Union[str, Path], sim_setting: Dict, observations: Dict, save_id: int, obj2cls: Dict
) -> None:
    """
    save rgb, depth, or semantic images in the observation dictionary according to the sim_setting.
    obj2cls is a dictionary mapping from object id to semantic id in habitat_sim.
    rgb are saved as .png files of shape (width, height) in sim_setting.
    depth are saved as .npy files where each pixel stores depth in meters.
    semantic are saved as .npy files where each pixel stores semantic id.

    """
    root_save_dir = Path(root_save_dir)
    if sim_setting["color_sensor"]:
        # save rgb
        save_name = f"{save_id:06}.png"
        save_dir = root_save_dir / "rgb"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / save_name
        obs = observations["color_sensor"][:, :, [2, 1, 0]] / 255
        cv2.imwrite(str(save_path), observations["color_sensor"][:, :, [2, 1, 0]])

    if sim_setting["depth_sensor"]:
        # save depth
        if sim_setting["depth_sensor"]:
            save_name = f"{save_id:06}.npy"
            save_dir = root_save_dir / "depth"
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / save_name
            obs = observations["depth_sensor"]
            with open(save_path, "wb") as f:
                np.save(f, obs)

    if sim_setting["semantic_sensor"]:
        # save semantic
        if sim_setting["semantic_sensor"]:
            # save_name = f"{save_id:06}.npy"
            save_dir_image = root_save_dir / "gt_semantic" 
            os.makedirs(save_dir_image, exist_ok=True)
            # save_dir = root_save_dir / "semantic"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = save_dir_image / save_name
            obs = observations["semantic_sensor"]
            obs = cvt_obj_id_2_cls_id(obs, obj2cls)
            # with open(save_path, "wb") as f:    #TODO: is it saved second time in plot_semantic?
            #     np.save(f, obs)
            plot_semantic(save_dir_image / f"{save_id:06}.png", obs)


def generate_scene_data(save_dir: Union[Path, str], config: DictConfig, scene_path: Path, poses: np.ndarray, mp3d_scene_config_path: str) -> None:
    """
    config: config for the sensors of the collected data
    scene_path: path to the Matterport3D scene file *.glb
    poses: (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
    """
    image_dim = max(config.resolution.w, config.resolution.h)
    sim_setting = {
        "scene": str(scene_path),
        "default_agent": 0,
        "sensor_height": config.camera_height,
        "color_sensor": config.rgb,
        "depth_sensor": config.depth,
        "semantic_sensor": config.semantic,
        "move_forward": 0.1,
        "turn_left": 5,
        "turn_right": 5,
        "width": image_dim,
        "height": image_dim,
        "enable_physics": False,
        "seed": 42,
        'mp3d_scene_config_path': mp3d_scene_config_path
    }
    cfg = make_cfg(sim_setting)
    sim = habitat_sim.Simulator(cfg)

    # get the dict mapping object id to semantic id in this scene
    obj2cls = get_obj2cls_dict(sim, config)

    # initialize the agent in sim
    agent = sim.initialize_agent(sim_setting["default_agent"])  #TODO: agent not used?
    pbar = tqdm(poses, leave=False)
    with open(os.path.join(save_dir, 'classes_map.yaml'), 'w') as classes_file:
        yaml.safe_dump(obj2cls, classes_file, default_flow_style=False)
    for pose_i, pose in enumerate(pbar):
        # if pose_i % config.step != 0:
        #     continue
        pbar.set_description(desc=f"Frame {pose_i:06}")
        agent_state = habitat_sim.AgentState()
        agent_state.position = pose[:3]
        agent_state.rotation = pose[3:]
        sim.get_agent(0).set_state(agent_state)
        obs = sim.get_sensor_observations(0)
        save_obs(save_dir, sim_setting, obs, pose_i, obj2cls)

    sim.close()


@hydra.main(
    version_base=None,
    config_path="../../src/omcl/configs",
    config_name="mp3d_config",
)
def main(config: DictConfig) -> None:
    # config = config.matterport
    print(config.paths)
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ['HYDRA_FULL_ERROR'] = '1'
    mp3d_dir = os.path.join(os.path.expanduser('~'), config.paths.all_data, 'mp3d')
    target_dir = os.path.join(os.path.expanduser('~'), config.paths.all_data)
    os.makedirs(target_dir, exist_ok=True)
    dataset_dir = Path(target_dir) / f'{config.dataset.name}'
    print("PATH: ", dataset_dir)
    if not dataset_dir.exists() or config.simulate.download_poses:
        zip_filepath = dataset_dir.parent / "vlmaps_dataset.zip"
        gdown.download(
            "https://drive.google.com/file/d/1KaRi1VnY7C_TT1WckDWxHvP4v3MTNu1a/view?usp=sharing",
            zip_filepath.as_posix(),
            fuzzy=True,
        )
        print("ZIP path:", zip_filepath.as_posix())
        print("UNZIP path:", dataset_dir.parent.as_posix())
        subprocess.run(["unzip", zip_filepath.as_posix(), "-d", dataset_dir.parent.as_posix()])
    
    data_dirs = sorted([x for x in dataset_dir.iterdir() if x.is_dir()])
    if config.paths.scenes:
        data_dirs = sorted([dataset_dir / x for x in config.paths.scenes])
    pbar = tqdm(data_dirs)
    mp3d_scene_config_path = os.path.join(mp3d_dir, 'mp3d.scene_dataset_config.json')
    assert os.path.exists(mp3d_scene_config_path), f'{mp3d_scene_config_path} does not exist'
    for data_dir_i, data_dir in enumerate(pbar):
        pbar.set_description(desc=f"Scene {data_dir.name:14}")
        scene_name = data_dir.name.split("_")[0]
        scene_path = Path(mp3d_dir) / scene_name / (scene_name + ".glb")
        pose_path = data_dir / "poses.txt"
        poses = np.loadtxt(pose_path)  # (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
        for old_data in os.listdir(data_dir):
            if old_data != 'poses.txt':
                if old_data in ['rgb', 'depth', 'gt_semantic']:
                    rm_old_data = os.path.join(data_dir, old_data)
                    if os.path.isdir(rm_old_data):
                        shutil.rmtree(rm_old_data, ignore_errors=True)
                    else:
                        os.remove(rm_old_data)
        # save_dir = dataset_dir.parent / config.dataset.name / data_dir.name
        # breakpoint()
        generate_scene_data(data_dir, config.simulate, scene_path, poses, mp3d_scene_config_path)


if __name__ == "__main__":
    main()
    