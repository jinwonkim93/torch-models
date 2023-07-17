import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding )
        )
    def forward(self, x):
        return self.layer(x)

class DenseBlock(nn.Module):
    """
    H_l layer
    """
    def __init__(self, num_layers, k):
        super().__init__()
        #TODO denseblock 역할이 무엇인지

        self.module_list = [(ConvBlock(k+k*(i-1), k+k*(i-1), kernel_size=1, padding=0),
                            ConvBlock(k+k*(i-1), k+k*(i-1), kernel_size=3))
                            for i in range(1, num_layers+1)
                             ]
        

    def forward(self, x):
        hidden_dims = []
        
        for conv1x1, conv3x3 in self.module_list:
            if len(hidden_dims) > 1:
                x = torch.cat(hidden_dims, dim=1)
            
            x = conv1x1(x)
            x = conv3x3(x)
            hidden_dims.append(x)


class TransitionLayer(nn.Module):
    pass

class DenseNet(nn.Module):
    pass


if __name__ == "__main__":
    input_batch = torch.ones((1, 3, 224, 224))
    convblock = ConvBlock(3, 32, kernel_size=7, stride=2, padding=3)
    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    # -> 
    # input_hs = torch.ones((4, 512, 112, 112))

    dense_block = DenseBlock(6, 32)

    h1 = convblock(input_batch)
    print(h1.shape)

    h2 = pool(h1)
    print(h2.shape)

    h3 = dense_block(h2)
    print(h3.shape)