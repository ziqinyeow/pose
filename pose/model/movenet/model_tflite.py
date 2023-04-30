import os
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..base import BaseModel


class TFLiteMovenet(BaseModel):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.input_size = config.get("input_size")
        self.model = self.load_tflite_from_local(config.get("model_path"))

    @staticmethod
    def load_tflite_from_local(path: str) -> None:
        """
        Load a movenet model from local path
        """

        path = (
            path
            if path.startswith("pose/model/movenet/weight")
            else os.path.join("pose/model/movenet/weight", path)
        )

        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        dtype = (
            tf.uint8 if "fp16" in path or "int8" in path else tf.float32
        )

        def movenet(input_image):
            input_image = tf.cast(input_image, dtype=dtype)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
            return keypoints_with_scores

        return movenet

    def forward(self, frame, crop_region=None):
        """
        x: A single frame -> [height, width, channel]

        return tensor with shape [1, 1, 17, 3]
        """

        if len(frame.shape) == 4 and frame.shape[0] == 1:
            return self.model(frame)

        assert len(frame.shape) == 3, "Only 1 frame is allowed at a time"

        if crop_region is not None:
            H, W, _ = frame.shape
            cropped_frame = tf.image.crop_and_resize(
                tf.expand_dims(frame, axis=0),
                box_indices=[0],
                boxes=[
                    [
                        crop_region["y_min"],
                        crop_region["x_min"],
                        crop_region["y_max"],
                        crop_region["x_max"],
                    ]
                ],
                crop_size=[self.input_size, self.input_size],
            )

            logits = self.model(cropped_frame)  # [1, 1, 17, 3]

            for idx in range(17):
                logits[0, 0, idx, 0] = (
                    crop_region["y_min"] * H
                    + crop_region["height"] * H * logits[0, 0, idx, 0]
                ) / H
                logits[0, 0, idx, 1] = (
                    crop_region["x_min"] * W
                    + crop_region["width"] * W * logits[0, 0, idx, 1]
                ) / W

            return logits
        else:
            return self.model(
                tf.expand_dims(
                    tf.image.resize_with_pad(frame, self.input_size, self.input_size),
                    axis=0,
                )
            )
