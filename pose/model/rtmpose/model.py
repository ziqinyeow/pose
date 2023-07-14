import numpy as np
import torch

from ..base import BaseModel

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline


class RTMPose(BaseModel):
    def __init__(self, config):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_estimator = init_pose_estimator(
            config.get("pose_config"), config.get("pose_ckpt"), device=device
        )
        self.detector = init_detector(
            config.get("det_config"), config.get("det_ckpt"), device=device
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def forward(self, frame: np.ndarray):
        try:
            detect_result = inference_detector(self.detector, frame)
            pred_instance = detect_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)
            ]
            bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

            # predict keypoints
            pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
            data_samples = merge_data_samples(pose_results)
            keypoints = data_samples.pred_instances.keypoints
            keypoints[:, :, [0, 1]] = keypoints[:, :, [1, 0]]
            return keypoints
        except:
            pass
