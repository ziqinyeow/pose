import numpy as np

class BaseModel:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, frame: np.ndarray, crop_region=None):
        raise NotImplementedError
    
    def plot(self, frame: np.ndarray, keypoints=None):
        raise NotImplementedError
