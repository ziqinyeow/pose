import os
import torch
import torch.nn as nn

from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .utils import keypoints_from_heatmaps
from ...utils.transform import resize

import numpy as np


__all__ = ["ViTPose"]


class ViTPose(nn.Module):
    def __init__(self, config) -> None:
        super(ViTPose, self).__init__()

        self.data_cfg = config.get("data_cfg")
        self.model_cfg = config.get("model_cfg")
        self.ckpt_path = config.get("model_path")
        self.H, self.W = self.data_cfg["image_size"]

        backbone_cfg = {
            k: v for k, v in self.model_cfg["backbone"].items() if k != "type"
        }
        head_cfg = {
            k: v for k, v in self.model_cfg["keypoint_head"].items() if k != "type"
        }

        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)

        self.load_pretrained()

    def load_pretrained(self):
        ckpt = torch.load(
            os.path.join("pose", "model", "vitpose", "weight", self.ckpt_path)
        )
        if "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"])
        else:
            self.load_state_dict(ckpt)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, frame):
        H, W, _ = frame.shape

        tensor = resize(frame, (self.W, self.H), "pt").unsqueeze(0)

        heatmaps = self.keypoint_head(self.backbone(tensor))
        heatmaps = heatmaps.detach().numpy()

        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[W // 2, H // 2]]),
            scale=np.array([[W, H]]),
            unbiased=True,
            use_udp=True,
        )
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)
        return points
