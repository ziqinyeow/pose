class TFMovenet:
    def __init__(self, config=None) -> None:
        self.model = None
        self.config = config or {}

    def load_tflite(self) -> None:
        pass
