## Simple VGG implementation

## Usage
```python
from models.vgg import VGG16
from trainer.trainer import Trainer

configs = {"batch_size":16, "max_epoch":10}

model = VGG16(n_classes=10)
trainer = Trainer(model)

trainer.train(configs)

```