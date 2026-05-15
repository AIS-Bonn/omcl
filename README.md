<h3 align="left">
  <a href="https://arxiv.org/abs/2512.15557">Paper</a> |
  <a href="https://github.com/AIS-Bonn/omcl">Video (soon)</a>  
</h3>

# OMCL: Open-vocabulary Monte Carlo Localization

We present <b>OMCL</b> (Open-vocabulary Monte Carlo Localization), a localization framework that extends <b>Monte Carlo Localization</b> with <b>vision-language features</b>.
Our <b>Ocree Language Map</b> enables <b>OMCL</b> to perform <b>visual-only</b> localization in 3D environments while generalizing across different scales.
By grounding pose estimation in language features, <b>OMCL</b> accelerates global localization through <b>open-vocabulary prompts</b>.

<div style="margin-top: 10px;"><b>Cross-modal sensor support:</b></div>
<div style="margin-left: 12px;">
<div>Mapping:</div>
<ul style="margin-top: 0; margin-bottom: 0;">
  <li><i>RGB-D</i></li>
  <li><i>Point clouds</i></li>
</ul>
<div>Localization:</div>
<ul style="margin-top: 0; margin-bottom: 0;">
  <li><i>Visual (RGB)</i></li>
</ul>
</div>

## Approach
![](media/pipeline.jpg)

## Installation 
Build Docker image: 

    ./docker/build.sh 



## Datasets
We use <a href="https://pixi.prefix.dev/latest/"> pixi</a> for easier datasets management:
    
    curl -fsSL https://pixi.sh/install.sh | sh

All commands work from this `.` directory.

#### Matterport 3D

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

#### KITTI
    soon

# Matterport 3D
## Mapping

Extract Language Features (for mapping with Option 1 and Localization)

    ./docker/run.sh
    python3 data_scripts/matterport/extract_lang_features.py

Create Octree Language Map: 

#### (Option 1) from RGB-D images:

    ./docker/run.sh
    python3 data_scripts/matterport/create_map.py

#### (Option 2) from point cloud:

    ./docker/run.sh
    python3 data_scripts/matterport/create_map.py visual_model=open_scene


## Localization

Visualization is available at <a href="http://0.0.0.0:8080/">http://0.0.0.0:8080</a>

Matterport3D + LSeg:

    ./docker/run.sh
    python3 omcl/examples/localize_mp3d.py 


Matterport3D + OpenScene:

    ./docker/run.sh
    python3 omcl/examples/localize_mp3d.py visual_model=open_scene

![](media/mp3d_demo.gif)

## Prompt-augmented Initialization (Global Localization)
    soon

# KITTI
## Mapping
    soon
## Localization
    soon