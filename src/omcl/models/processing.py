import torch
import time
import math


def get_sim_scores(feature, other_features, bsize=10000, device='cuda'):
    scores = torch.empty(other_features.shape[0], dtype=feature.dtype, device=device)
    for i in range(math.ceil(other_features.shape[0] / bsize)):
        st = i*bsize
        end = min((i+1)*bsize, other_features.shape[0])
        scores[st:end] = (1 - torch.nn.functional.cosine_similarity(feature, other_features[st:end], dim=-1)).to(device=scores.device)
    return scores
    
    
def get_unique_features(known_features, l_features, similarity_threshold=0.05, batch_size=5, mean=False):
    # LiLMaps: https://arxiv.org/abs/2501.03304
    with torch.no_grad():
        if len(known_features) > 0:
            for i in range(math.ceil(known_features.shape[0] / batch_size)):
                st = i*batch_size
                end = min((i+1)*batch_size, known_features.shape[0])
                scores = 1 - torch.nn.functional.cosine_similarity(known_features[st:end, None,...], l_features.half()[None], dim=-1)
                mask = torch.all(scores.abs() > similarity_threshold, dim=0)  # unknown features mask
                l_features = l_features[mask]
        unique_features = []
        while len(l_features) > 0:
            feature = l_features[0] # choose random feature
            # scores = 1 - torch.nn.functional.cosine_similarity(feature[None], l_features)
            scores = get_sim_scores(feature[None], l_features)
            mask = scores.abs() <= similarity_threshold  # find similar features
            if mean:
                # choose mean features
                mean_feature = l_features[mask].mean(0)
                mean_feature /= mean_feature.norm(2)
            else:
                # choose random feature
                mean_feature = l_features[mask][0]
            l_features = l_features[torch.logical_not(mask)]    # remove averaged features from the set
            unique_features.append(mean_feature)
        if len(unique_features) > 0:
            return torch.stack(unique_features)
        else:
            return torch.tensor([], device=known_features.device)