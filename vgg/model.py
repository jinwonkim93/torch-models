import torch

class VGG16(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = torch.nn.Softmax(1)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
            )

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.layer2 = torch.nn.Sequential(
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool
            )

        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.layer3 = torch.nn.Sequential(
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.conv7,
            self.relu,
            self.maxpool
            )

        self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer4 = torch.nn.Sequential(
            self.conv8,
            self.relu,
            self.conv9,
            self.relu,
            self.conv10,
            self.relu,
            self.maxpool
            )

        self.conv11 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer5 = torch.nn.Sequential(
            self.conv11,
            self.relu,
            self.conv12,
            self.relu,
            self.conv13,
            self.relu,
            self.maxpool
            )

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.LazyLinear(out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = torch.nn.Linear(in_features=4096, out_features=n_classes)
        self.layer6 = torch.nn.Sequential(
            self.flat,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.softmax
            )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

