import importlib
from dataclasses import dataclass


@dataclass
class AutoConfigType:
    backend: str
    input_size: int
    model_path: str


class AutoConfig:
    def __init__(self, config=None):
        self.config = config or dict()

    @staticmethod
    def from_pretrained(model: str):
        module = importlib.import_module(f"pose.model.{model}.config")
        config = getattr(module, "CONFIG")(model)
        return AutoConfig(config)

    def __repr__(self) -> str:
        return str(self.config)
    
    def get(self, key, v=None):
        return self.config.get(key, v)

    def set(self, key, value):
        self.config[key] = value
        return self.config
