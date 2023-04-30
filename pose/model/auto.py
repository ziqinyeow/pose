import inspect
import importlib


class AutoModel:
    def __init__(self, model=None, config=None) -> None:
        self.model = model(config)
        self.config = config

    @staticmethod
    def from_pretrained(model: str, config=None):
        main_model = model.split(".")[0]
        module = importlib.import_module(f"pose.model.{main_model}.config")
        config = config or getattr(module, "CONFIG")(model)
        backend = config.get("backend")

        if backend == "pt":
            module = importlib.import_module(f"pose.model.{main_model}.model")
        elif backend == "tf":
            module = importlib.import_module(f"pose.model.{main_model}.model_tf")
        elif backend == "tflite":
            module = importlib.import_module(f"pose.model.{main_model}.model_tflite")

        # TODO: Remove the use of only the last class model
        model = list(map(lambda x: x, inspect.getmembers(module, inspect.isclass)))[-1][
            1
        ]

        return AutoModel(model=model, config=config)

    def get_config(self, key, v=None):
        return self.config.get(key, v)

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
