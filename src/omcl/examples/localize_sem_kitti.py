
import numpy as np
import viser
import hydra
from omegaconf import DictConfig
from omcl.utils.plot import plot_floor
from omcl.models.pf import run


@hydra.main(
    version_base=None,
    config_path="../../omcl/configs",
    config_name="sem_kitti_config",
)
def main(config: DictConfig):
    viser_server = viser.ViserServer()
    for scene in config.dataset.scenes:
        print(scene)
        plot_floor(scene, config, viser_server)
        # data_path, points, points_labels, poses44, pose_init, rot_init, features_db, features_labels = load_data('5LpN3gDmAk7_1', config)
        # odoms = estimate_odoms(poses44, pose_init, rot_init)
        # viser_server.initial_camera.position = np.array([ 36.59388359, -59.70568682, 246.16176265])
        # viser_server.initial_camera.look_at = np.array([172.51614205, -41.75260087, -66.64686478])
        poses44, estimated_poses, octree_map, precision_steps = run(scene, config, first_pose_id=0, device='cuda', viser_server=viser_server, batch_size=config.particles_batch_size)
    


if __name__ == '__main__':
    main()
