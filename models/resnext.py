import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality=32):
        super().__init__()
        self.cardinality = cardinality
        res_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, res_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(res_channels, res_channels, kernel_size=3, stride=stride, padding=1, groups=self.cardinality)
        self.conv3 = ConvBlock(res_channels, out_channels, kernel_size=1,stride=1, padding=0)
        self.projection_layer = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)  #for downprojection
        self.relu = nn.ReLU()

    def forward(self, x):
        print("x_out", x.shape) #64
        out = self.relu(self.conv1(x))
        print("conv1_out", out.shape) #128
        out = self.relu(self.conv2(out))
        print("conv2_out", out.shape) #128
        out = self.conv3(out)
        print("conv3_out", out.shape) #256
        x = self.projection_layer(x)
        print("projection_layer_out",x.shape)
        return self.relu(out+x) #256 + 64 x

class ResNeXt(nn.Module):
    def __init__(self, n_classes=1000, conv2_size=3, conv3_size=4, conv4_size=6, conv5_size=3, cardinality=32):
        super().__init__()
        self.n_classes = n_classes
        in_channels, out_channels = 3, 64
        
        bongsu_print("#conv1_size")
        print("in out : ", in_channels, out_channels)
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3)

        bongsu_print("#conv2_size")
        in_channels, out_channels = out_channels, out_channels*4
        print("in out : ", in_channels, out_channels)
        self.conv2 = nn.Sequential( 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #
            ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=1, cardinality=cardinality),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1, cardinality=cardinality),
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1, cardinality=cardinality)
        ) 
        
        bongsu_print("#conv3_size")
        in_channels, out_channels = out_channels, out_channels*2
        print("in out : ", in_channels, out_channels)
        in_channels_list = [in_channels] + [out_channels] * (conv3_size-1)
        print("in _ch_list : ",in_channels_list)
        stride_list = [2] + [1] * (conv3_size-1)
        print("stride list :", stride_list)
        self.conv3 = nn.Sequential(
            *[ResidualBlock(in_channels=in_ch,
                            out_channels=out_channels,
                            stride=stride,
                            cardinality=cardinality) for in_ch, stride,  _ in zip(in_channels_list, stride_list, range(conv3_size))]
        )

        bongsu_print("#conv4_size")
        in_channels, out_channels = out_channels, out_channels*2
        print("in out : ", in_channels, out_channels)
        in_channels_list = [in_channels] + [out_channels] * (conv4_size-1)
        print("in _ch_list : ",in_channels_list)
        stride_list = [2] + [1] * (conv4_size-1)
        print("stride list :", stride_list)
        self.conv4 = nn.Sequential(
            *[ResidualBlock(in_channels=in_ch,
                            out_channels=out_channels,
                            stride=stride,
                            cardinality=cardinality) for in_ch, stride,  _ in zip(in_channels_list, stride_list, range(conv4_size))]
        )

        bongsu_print("#conv5_size")
        in_channels, out_channels = out_channels, out_channels*2
        print("in out : ", in_channels, out_channels)
        in_channels_list = [in_channels] + [out_channels] * (conv5_size-1)
        print("in _ch_list : ",in_channels_list)
        stride_list = [2] + [1] * (conv5_size-1)
        print("stride list :", stride_list)
        self.conv5 = nn.Sequential(
            *[ResidualBlock(in_channels=in_ch,
                            out_channels=out_channels,
                            stride=stride,
                            cardinality=cardinality) for in_ch, stride,  _ in zip(in_channels_list, stride_list, range(conv5_size))]
        )

        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.ffn = nn.Linear(out_channels, self.n_classes)

    
    #cardinality
    def forward(self, x):
        
        bongsu_print("#conv1_x")
        print("x_out", x.shape)
        out = self.conv1(x)
        print("conv1_out", out.shape)
        
        bongsu_print("#conv2_x")
        out = self.conv2(out)
        print("conv2_out", out.shape)

        bongsu_print("#conv3_x")
        out = self.conv3(out)     
        print("conv3_out", out.shape)

        bongsu_print("#conv4_x")
        out = self.conv4(out)
        print("conv4_out", out.shape)
        
        bongsu_print("#conv5_x")
        out = self.conv5(out)
        print("conv5_out", out.shape)

        bongsu_print("#GAP")
        out = self.GAP(out)
        print("GAP", out.shape)

        
        bongsu_print("#flatten")
        out = torch.flatten(out, start_dim=1)
        print("flatten", out.shape)


        bongsu_print("#ffn")
        out = self.relu(out)
        out = self.ffn(out)
        print("ffn", out.shape)

        return out

def bongsu_print(string):
    print(f"""\n{'##'*50}\n{string}\n{'##'*50}\n""")

if __name__=='__main__':
    dummy_inputs = torch.rand((4, 3, 224, 224))
    # conv1 = nn.Conv2d(64, 32, kernel_size=1, groups=32)
    resnext = ResNeXt(1000, 3,4,6,3)
    print(resnext(dummy_inputs).shape)
    # print(conv1(dummy_inputs).shape)
    # print(conv1(dummy_inputs))

    # dummy_inputs = torch.rand((4, 6, 224, 224))

