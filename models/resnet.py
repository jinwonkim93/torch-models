import torch
from torch.nn.functional import relu


class ResBlock(torch.nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                increase_dim = False
    ):
        """
        Args: 
            in_channels : int 256
            out_channels : int 256
            kernel_size : int 3
            stride : int 1
            padding : int 1
            increase_dim = False
        """
        super().__init__()
        if increase_dim is True: 
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride+1, padding)
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, 1, 2, 0)
        else :
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):        
        h1 = relu(self.conv1(x))
        h2 = self.conv2(h1)
        x = self.conv1x1(x)
        return relu(h2+x) #elemental wise add
    
class ResNet(torch.nn.Module):
    def __init__(self, conv2_size, conv3_size, conv4_size, conv5_size): #3 4 6 3
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        modules_list = []
        for i in range(conv2_size) : #3
            modules_list.append(ResBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, increase_dim=False))
        self.conv_2_x= torch.nn.Sequential(*modules_list)

        modules_list = []
        for i in range(conv3_size) : #4
            if i == 0 :
                modules_list.append(ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, increase_dim=True))
            else :
                modules_list.append(ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, increase_dim=False))
        self.conv_3_x= torch.nn.Sequential(*modules_list)

        modules_list = []
        for i in range(conv4_size) : #6
            if i == 0 :
                modules_list.append(ResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, increase_dim=True))
            else : 
                modules_list.append(ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, increase_dim=False))
        self.conv_4_x= torch.nn.Sequential(*modules_list)

        modules_list = []
        for i in range(conv5_size) : #6
            if i == 0 :
                modules_list.append(ResBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, increase_dim=True))
            else :
                modules_list.append(ResBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, increase_dim=False))
        self.conv_5_x= torch.nn.Sequential(*modules_list)

    
        #self.avg_pool = partial(avg_pool2d(kernel_size=7, stride=None, padding=0))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.ffn = torch.nn.Linear(in_features=512, out_features=1000)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv_2_x(x)
        x = self.conv_3_x(x)
        x = self.conv_4_x(x)
        x = self.conv_5_x(x)
        x = self.avgpool(x) #14 14 -> 8 8
        x = torch.flatten(x, start_dim=1)
        x = self.ffn(x)
        x = self.softmax(x)
        return x



resnet = ResNet(conv2_size=3, conv3_size=4, conv4_size=6, conv5_size=3)
dummy = torch.zeros((4, 3, 224, 224))
print(resnet(dummy))

