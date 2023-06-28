from typing import Any

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


class Cifar10Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def train(self, configs):

        #TODO##################################
        batch_size = configs['batch_size']
        num_workers = configs['num_workers']
        max_epoch = configs['max_epoch']
        dataset_type = configs.get("dataset_type", None)
        
        if dataset_type == "cifar10":
            dataset_root = configs.get("dataset_root", None)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                            ])

            cifar_data = datasets.CIFAR10(dataset_root, download=True, transform=transform, train=True)

            data_loader = DataLoader(cifar_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

        for epoch in range(max_epoch):
            for batch_image, batch_target in data_loader:
                    self.optimizer.zero_grad()
                    pred = self.model(batch_image)
                    loss = self.loss_fn(pred, batch_target)     
                    loss.backward()
                    print(f"Epoch: {epoch}, Loss: {loss.item()}")
                    self.optimizer.step()
                    self.scheduler.step()
        ############################################