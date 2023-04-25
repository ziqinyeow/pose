# 🚀 Type of Models

1. `movenet`: default to `movenet.thunder.single`
2. `movenet.lightning.single`
3. `movenet.lightning.single.int8`
4. `movenet.lightning.single.fp16`
5. `movenet.lightning.multi.fp16`
6. `movenet.thunder.single`
7. `movenet.thunder.single.int8`
8. `movenet.thunder.single.fp16`

# 👇 Current Version

## Supported model backend

1. Only TFLite
2. PyTorch (TODO)
3. TensorFlow (TODO)

## Supported stage

Only inference is supported, no training, evaluation api yet

# 🤔 Usage

```python
from pose import AutoModel

model = AutoModel.from_pretrained('movenet')
```
