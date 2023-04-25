import os
import tensorflow as tf


class TFLiteMovenet:
    def __init__(self, config) -> None:
        self.config = config
        self.input_size = config.get("input_size")
        self.model = self.load_tflite_from_local(config.get("model_path"))

    @staticmethod
    def load_tflite_from_local(path: str) -> None:
        path = (
            path
            if path.startswith("pose/model/movenet/weight")
            else os.path.join("pose/model/movenet/weight", path)
        )

        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        dtype = (
            tf.float16 if "fp16" in path else tf.int8 if "int8" in path else tf.float32
        )

        def movenet(input_image):
            input_image = tf.cast(input_image, dtype=dtype)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
            return keypoints_with_scores

        return movenet

    def __call__(self, x):
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=0)
        x = tf.image.resize_with_pad(x, self.input_size, self.input_size)
        return self.model(x)
