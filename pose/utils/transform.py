import numpy as np
from PIL import Image
import tensorflow as tf
from torchvision.transforms import transforms


def resize(frame, sz=(1280, 1280), backend="tf") -> np.ndarray:
    # return -> np.ndarray
    H, W = sz
    if backend == "tf":
        return tf.image.resize_with_pad(frame, H, W).numpy().astype(np.uint8)
    elif backend == "pt":
        img = Image.fromarray(frame)
        img = transforms.Compose(
            [transforms.Resize((256, 192)), transforms.ToTensor()]
        )(img)
        return img
