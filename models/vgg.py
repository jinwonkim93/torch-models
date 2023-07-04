import torch
import torch.nn as nn
class VGG16(torch.nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        
        relu = torch.nn.ReLU(inplace=True)
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        dropout = torch.nn.Dropout(p=0.5, inplace=False)

        conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        conv8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        conv9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        conv10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        conv11 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        conv13 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.features = torch.nn.Sequential(
            conv1,
            relu,
            conv2,
            relu,
            maxpool,
            conv3,
            relu,
            conv4,
            relu,
            maxpool,
            conv5,
            relu,
            conv6,
            relu,
            conv7,
            relu,
            maxpool,
            conv8,
            relu,
            conv9,
            relu,
            conv10,
            relu,
            maxpool,
            conv11,
            relu,
            conv12,
            relu,
            conv13,
            relu,
            maxpool,
            )
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))

        fc1 = torch.nn.Linear(in_features=25088, out_features=4096)
        fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        fc3 = torch.nn.Linear(in_features=4096, out_features=n_classes)
        self.classifier = torch.nn.Sequential(
            fc1,
            relu,
            dropout,
            fc2,
            relu,
            dropout,
            fc3,
            )

    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

