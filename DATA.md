We use <a href="https://pixi.prefix.dev/latest/"> pixi</a> for easier datasets management:
    
    curl -fsSL https://pixi.sh/install.sh | sh

All commands work from this `.` directory.

## Matterport 3D

Download `download_mp.py` script from <a href="https://niessner.github.io/Matterport/">Matterport3D website</a> and place it into `data_scripts/` direcroty.

        .
        ├── data_scripts 
        │   ├── download_mp.py 
        │   ├── ... 
        │   ...


Use the following commands to prepare the dataset:
    
    mkdir -p ~/data
    pixi run -e legacy python data_scripts/download_mp.py -o . --task habitat # Enter->CTRL-C
    pixi run unzip_mp3d
    pixi run simulation_mp3d
    pixi run download_open_scene



###### Additional notes:
More details about Matterport 3D dataset can be found <a href="https://github.com/vlmaps/vlmaps#generate-dataset">here</a> and [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset).\
habitat-sim installation problems: https://github.com/facebookresearch/habitat-sim/issues/2147

## SemanticKITTI
* Download KITTI Odometry benchmark [color data 65 GB](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* Download KITTI Odometry benchmark [velodyne laser data 80 GB](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* Download KITTI Odometry benchmark [calibration files 1 MB](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* Download SemanticKITTI [label data 179 MB](https://semantic-kitti.org/dataset.html#download). It contains poses.txt as well.
* Place all donwloaded files into the `~/data` folder:

        ~ # user home directory ~
        ├── data
        │   ├── data_odometry_color.zip         # 65 GB
        │   ├── data_odometry_velodyne.zip      # 80 GB
        │   ├── data_odometry_labels.zip        # 179 MB
        │   ├── data_odometry_calib.zip         # 1 MB
        │   ├── ...

<p></p>

* run:

        pixi run unzip_sem_kitti
        pixi run download_xdecoder

