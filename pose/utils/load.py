import os
import sys
import mimetypes


import cv2
import torch
import numpy as np
import tensorflow as tf

mimetypes.init()


def check_file(path: str):
    mimestart = mimetypes.guess_type(path)[0]

    if mimestart != None:
        mimestart = mimestart.split("/")[0]

        if mimestart in ["video", "image"]:
            return mimestart

    return None


def load(path: str, return_tensors="tf"):
    """
    Load a video/image file

    return_tensors:
        np: Numpy array
        pt: Pytorch tensor
        tf: TensorFlow tensor
    """
    if not os.path.exists(path):
        print("Path not exists!")
        return None

    mime = check_file(path)

    if not mime:
        print("Path is not a video or image file")
        return None

    if mime == "video":
        fc = 0
        ret = True

        cap = cv2.VideoCapture(path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tensor = np.empty((count, height, width, 3), np.dtype("uint8"))

        while fc < count and ret:
            ret, frame = cap.read()
            tensor[fc] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fc += 1

        cap.release()
        print(f"Video size: {sys.getsizeof(tensor) / sys.maxsize * 100} % space used")

    else:  # image
        tensor = cv2.imread(path)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        print(f"Image size: {sys.getsizeof(tensor) / sys.maxsize * 100} % space used")

    return (
        tensor
        if return_tensors == "np"
        else torch.from_numpy(tensor)
        if return_tensors == "pt"
        else tf.convert_to_tensor(tensor)
    )
