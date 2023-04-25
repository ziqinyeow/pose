from enum import Enum
from dataclasses import dataclass


class BACKEND(Enum):
    TORCH = "pt"
    TF = "tf"
    TFLITE = "tflite"
