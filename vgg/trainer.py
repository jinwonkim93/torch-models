from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(self, n_classes=10, dataset_length=100):
        self.n_classes = n_classes
        self.dataset_length = dataset_length

    def __getitem__(self, index) -> Any:
        image = torch.rand((3, 224, 224), dtype=torch.float32)
        target = torch.rand(size=(self.n_classes, ))
        return image, target
    
    def __len__(self):
         return self.dataset_length

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def train(self, configs):

        batch_size = configs['batch_size']
        max_epoch = configs['max_epoch']

        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size)

        for epoch in range(max_epoch):
            for batch_image, batch_target in dataloader:
                    self.optimizer.zero_grad()
                    pred = self.model(batch_image)
                    loss = self.loss_fn(pred, batch_target)
                    print(loss)        
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()