import os
import inspect
import importlib


class Wiki:
    def __init__(self):
        pass

    @staticmethod
    def list_dataset():
        pass

    @staticmethod
    def list_models():
        path = "pose/model"
        models = set(
            [
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]
        )
        models.discard("__pycache__")

        return list(models)

    @staticmethod
    def get_model_class_dict(model, backend):
        if backend == "pt":
            module = importlib.import_module(f"pose.model.{model}.model")
        elif backend == "tf":
            module = importlib.import_module(f"pose.model.{model}.model_tf")
        elif backend == "tflite":
            module = importlib.import_module(f"pose.model.{model}.model_tflite")

        cls = dict(map(lambda x: x, inspect.getmembers(module, inspect.isclass)))
        return cls

    @staticmethod
    def get_config(model: str):
        pass
