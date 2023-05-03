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

        res = []
        for m in list(models):
            module = importlib.import_module(f"pose.model.{m}.config")
            config = list(getattr(module, "MODEL").keys())
            res += config
        return res

    @staticmethod
    def get_model_parent(model: str):
        path = "pose/model"
        models = set(
            [
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]
        )
        models.discard("__pycache__")

        for m in list(models):
            module = importlib.import_module(f"pose.model.{m}.config")
            config = list(getattr(module, "MODEL").keys())
            if model in config:
                return m

        return None

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
