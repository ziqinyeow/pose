import numpy as np
import tensorflow as tf


def resize(frame, sz=(1280, 1280)):
    # return -> np.ndarray
    H, W = sz
    return tf.image.resize_with_pad(frame, H, W).numpy().astype(np.uint8)
