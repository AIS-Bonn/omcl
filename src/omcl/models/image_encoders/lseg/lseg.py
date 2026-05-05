import math
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pathlib import Path

from omcl.models.image_encoders.lseg.modules.models.lseg_net import LSegEncNet
from omcl.models.image_encoders.lseg.additional_utils.models import resize_image, pad_image, crop_image
import gdown
import clip

def encode_text(encoder, text_list, device='cuda'):
    text = clip.tokenize(text_list)  
    text = text.to(device)
    text_features = encoder.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
    return text_features

class LSeg:
    def __init__(self, base_size, crop_size) -> None:
        self.crop_size = crop_size #kitti - 480 # 480 640
        self.base_size = base_size # kitti - 1241 # 520 680
        lseg_model, lseg_transform, norm_mean, norm_std = init_lseg(self.crop_size)
        self.lseg_model = lseg_model
        self.lseg_transform = lseg_transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def get_lseg_feat(self, rgb, labels):
        return get_lseg_feat(self.lseg_model, rgb, labels, self.lseg_transform, 
                             'cuda', self.crop_size, self.base_size, self.norm_mean, self.norm_std)
    
    def encode_text(self, labelset):
        text = clip.tokenize(labelset)  
        text = text.to('cuda')
        text_features = self.lseg_model.clip_pretrained.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
        return text_features.detach().cpu()
    

def get_lseg_feat(
    model: LSegEncNet,
    image: np.array,
    labels,
    transform,
    device,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5]
):
    # vis_image = image.copy()
    image = transform(image).unsqueeze(0).to(device)
    img = image[0].permute(1, 2, 0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            # outputs = model(pad_img)
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
        pred = None
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape  # .size()
        assert ph >= height and pw >= width
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().to(device)
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(device)
            count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(device)
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    # output = model(pad_crop_img)
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                count_norm[:, :, h0:h1, w0:w1] += 1
        assert (count_norm == 0).sum() == 0
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:, :, :height, :width]
        logits_outputs = logits_outputs[:, :, :height, :width]
        predicts = [torch.max(logit, 0)[1].cpu() for logit in logits_outputs]
        pred = predicts[0]
    # outputs = resize_image(outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
    # outputs = resize_image(outputs, image.shape[0], image.shape[1], **{'mode': 'bilinear', 'align_corners': True})
    # outputs = outputs.cpu()
    # outputs = outputs.numpy()  # B, D, H, W
    return outputs, pred


def init_lseg(crop_size):
    device = 'cuda'
    lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    model_state_dict = lseg_model.state_dict()
    checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
    checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_path.exists():
        print("Downloading LSeg checkpoint...")
        # the checkpoint is from official LSeg github repo
        # https://github.com/isl-org/lang-seg
        checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
        gdown.download(checkpoint_url, output=str(checkpoint_path))

    pretrained_state_dict = torch.load(checkpoint_path, map_location=device)
    pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    model_state_dict.update(pretrained_state_dict)
    lseg_model.load_state_dict(pretrained_state_dict)

    lseg_model.eval()
    lseg_model = lseg_model.to(device)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    lseg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    clip_feat_dim = lseg_model.out_c
    return lseg_model, lseg_transform, norm_mean, norm_std