import torch
from omcl.models.image_encoders.minkowski.disnet import DisNet, state_dict_remove_moudle
from omcl.models.image_encoders.minkowski.voxelizer import Voxelizer
from MinkowskiEngine import SparseTensor
import os

class OpenSceneModel:
    def __init__(self):
        # download weights from https://cvg-data.inf.ethz.ch/openscene/models/ and put it to ~/data/models
        checkpoint = torch.load(os.path.join(os.path.expanduser('~'), "data/models/matterport_lseg.pth.tar"), map_location=lambda storage, loc: storage.cuda(), weights_only=False)
        class DisnetConfig:
            feature_2d_extractor = 'lseg'
            arch_3d = 'MinkUNet18A'
        self.model = DisNet(DisnetConfig)
        self.model = self.model.cpu()#.cuda()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        # kernel = self.model.net3d.final.kernel
        # kernel.requires_grad_(False)

    def forward(self, point_cloud, voxel_size=0.02):
        feat = torch.ones(point_cloud.shape[0], 3, device=point_cloud.device)
        vox = Voxelizer(voxel_size=voxel_size)
        locs, feats, inds_reconstruct = vox.voxelize(point_cloud.cpu().numpy(), feat.cpu().numpy(), None)
        coords = torch.from_numpy(locs).int()
        feats = torch.ones(coords.shape[0], 3, device=coords.device)
        coords = torch.cat((torch.ones((coords.shape[0], 1), dtype=torch.int, device=coords.device), coords), dim=1)
        # sinput = SparseTensor(feats.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        sinput = SparseTensor(feats, coords)
        output = self.model(sinput).half()
        sem_features = output[inds_reconstruct]
        return sem_features
        