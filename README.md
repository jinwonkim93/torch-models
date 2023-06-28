## Simple VGG implementation

## Usage
```python
from models.vgg import VGG16
from trainer.cifar10_trainer import Cifar10Trainer

configs = {"batch_size":16, "max_epoch":10}

model = VGG16(n_classes=10)
trainer = Cifar10Trainer(model)

trainer.train(configs)

```