# Extractes language features from RGB images

import numpy as np
import os
import shutil
from tqdm.auto import tqdm
import viser
import torch
import hydra
from omegaconf import DictConfig
import cv2
from PIL import Image
from torchvision import transforms
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from tools import LABELS
# import from X-Decoder
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_command
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
import time


def get_similarity_ids(image_features, features_db):
    """
    returns the image with feature indexes per pixel
    """
    similarity_ids = (image_features.half().cuda() @ features_db.T.cuda()).argmax(-1)
    return similarity_ids


def process_frames(viser_server: viser.ViserServer, scene_name: str, config: DictConfig, model):
    scene_dir = os.path.join(os.path.expanduser('~'), 'data', 'semantic_kitti', 'dataset', 'sequences', scene_name)
    rgb_dir = os.path.join(scene_dir, 'image_2')
    # for visualiztion
    # sem_colors = np.concatenate([d3_40_colors_rgb, generate_rgb_colors(config.vis.num_colors)], axis=0)
    ids = sorted((im_file[:-4] for im_file in os.listdir(rgb_dir)))
    all_features_db = torch.tensor([]).cuda()
    save_dir = os.path.join(scene_dir, f'{config.visual_model.name}_semantic')
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])
    
    
    for _, id_str in enumerate(tqdm(ids, desc=scene_name)):
        image_ori = Image.open(os.path.join(rgb_dir, f'{id_str}.png')).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
    
        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        sem_seg = outputs[0]['sem_seg'].argmax(0).to(torch.uint8).detach().cpu()
        torch.save(sem_seg, os.path.join(save_dir, f'{id_str}.pt'))
        
        
        if _ % 100 == 0:
            visual = Visualizer(image_ori, metadata=model.model.metadata)
            demo = visual.draw_sem_seg(sem_seg, alpha=0.5) # rgb Image
            d_img = demo.get_image()
            cv2.imwrite(os.path.join(save_dir, f'{id_str}.png'), d_img)
            _ = viser_server.scene.add_camera_frustum(name='x-decoder', fov=90, aspect=d_img.shape[1]/d_img.shape[0], scale=1., image=d_img)
            time.sleep(1)
        else:
            cv2.imwrite(os.path.join(save_dir, f'{id_str}.png'), image_ori)
            _ = viser_server.scene.add_camera_frustum(name='x-decoder', fov=90, aspect=d_img.shape[1]/d_img.shape[0], scale=1., image=image_ori)

    print(f"saving features database to {scene_dir}")
    all_features_db = model.model.sem_seg_head.predictor.lang_encoder.default_text_embeddings
    
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings( [v for v in LABELS.values()], is_eval=True)
    scene_features =  model.model.sem_seg_head.predictor.lang_encoder.default_text_embeddings
    assert scene_features.shape[0] == len(LABELS)
    torch.save({'features': all_features_db.cpu(),
                'labels': [*range(len(all_features_db))],
                'scene_features': scene_features.cpu()
                },
               os.path.join(scene_dir, f"{config.visual_model.name}_features_db.pt"))
    print("Done!")


@hydra.main(
    version_base=None,
    config_path="../../src/omcl/configs",
    config_name="sem_kitti_config",
)
def main(config: DictConfig):
    
    viser_server = viser.ViserServer()
    # viser_server.scene.add_grid('floor', width=50, height=50, position=(0,0, -1.5))
    if config.visual_model.name == "xdecoder":
        args = [
            "evaluate",
            "--conf_files", "third_party/X-Decoder/configs/xdecoder/xdecoder_focall_lang.yaml",
            "--overrides",
            "RESUME_FROM", os.path.join(os.path.expanduser('~'), 'data/models', "xdecoder_focall_last.pt")
        ]
        opt, cmdline_args = load_opt_command(args)
        opt = init_distributed(opt)
        model = BaseModel(opt, build_model(opt)).from_pretrained(os.path.join(os.path.expanduser('~'), 'data/models', "xdecoder_focall_last.pt")).eval().cuda()
        # For simplicity, open-set labels are used to extract the image features
        open_labels_set = ['sky',
                        "car", 
                        "bicycle", 
                        "bus", 
                        "motorcycle", 
                        "on-rails", 
                        "truck", 
                        # "other-vehicle", 
                        "person", 
                        "bicyclist", 
                        "motorcyclist", 
                        "road",
                        "parking", 
                        # "sidewalk",
                        # "other-ground", 
                        "building", 
                        "fence",
                        # "other-structure", 
                        "lane-marking", 
                        "vegetation",
                        "trunk", 
                        "terrain",
                        "pole",
                        "traffic-sign"
                        ]
        metadata = MetadataCatalog.get('demo')
        _ = MetadataCatalog.get("demo").set(
            stuff_colors=[random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(open_labels_set))],
            stuff_classes=open_labels_set,
            stuff_dataset_id_to_contiguous_id={x:x for x in range(len(open_labels_set))},
        )
    
    with torch.no_grad():    
        for scene in config.dataset.scenes:
            print("Scene: ", scene)
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(open_labels_set, is_eval=True)
            model.model.metadata = metadata
            model.model.sem_seg_head.num_classes = len(open_labels_set)
            process_frames(viser_server, scene_name=scene, config=config, model=model)


if __name__ == '__main__':
    main()
