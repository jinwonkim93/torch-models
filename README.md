## Simple VGG implementation

## Usage
```python
from vgg.model import VGG16
from vgg.trainer import Trainer

configs = {"batch_size":16, "max_epoch":10}

model = VGG16(n_classes=10)
trainer = Trainer(model)

trainer.train(configs)

```