import cv2
import torch
import numpy as np
import tensorflow as tf

from vidgear.gears import CamGear


def load(src: str, return_tensors="np"):
    """
    Load a video/image file async

    return_tensors:
        np: Numpy array
        pt: Pytorch tensor
        tf: TensorFlow tensor

    return a generator
    """
    stream = CamGear(source=src, colorspace="COLOR_BGR2RGB").start()

    while True:
        frame = stream.read()
        if frame is None:
            break

        yield (
            frame
            if return_tensors == "np"
            else torch.from_numpy(frame)
            if return_tensors == "pt"
            else tf.convert_to_tensor(frame)
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
