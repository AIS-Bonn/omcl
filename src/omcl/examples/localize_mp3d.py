
import numpy as np
import os
from tqdm.auto import tqdm
from omcl.models.odom import estimate_odoms
import viser
import yaml
import torch
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import DictConfig
from omcl.utils.plot import plot_floor
from omcl.models.pf import estimate_loss, run


@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="mp3d_config",
)
def main(config: DictConfig):
    viser_server = viser.ViserServer()
    for scene in config.dataset.scenes:
        print(scene)
        plot_floor(scene, config, viser_server)
        # data_path, points, points_labels, poses44, pose_init, rot_init, features_db, features_labels = load_data('5LpN3gDmAk7_1', config)
        # odoms = estimate_odoms(poses44, pose_init, rot_init)

        poses44, estimated_poses, octree_map, precision_steps = run(scene, config, first_pose_id=0, device='cuda', viser_server=viser_server, batch_size=config.particles_batch_size)
    


if __name__ == '__main__':
    main()
