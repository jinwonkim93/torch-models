import torch
import torch.nn as nn



# k = 32일때
# l = 6일때
# 64 + 32 + 32 + 32 + 32 + 32 = 224
#k0 + k*(l-1) 
#64 -> 32 -> [32] -> [32],[32] -> [32],[32,32] -> [32],[32,32,32] -> [32][32, 32, 32, 32] -?

class ConvBlock(nn.Module):
    def __init__(self, in_channels, bn_size, growth_rate):
        
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=bn_size*growth_rate, kernel_size=1, stride=1, padding=0 ),
            nn.BatchNorm2d(num_features=bn_size*growth_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=bn_size*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1)        
        )
    def forward(self, x, skip_connection=True):
        out = self.layer(x)
        if skip_connection is True:
            out = torch.cat([out, x], dim=1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0 ),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module) :
    def __init__(self, n_layer=121, bn_size=4, growth_rate=32) :
        #super(DenseNet, self).__init__() #python 2
        super().__init__()

        #dense1
        #conv1(3, 64) =  ConvBlock
        
        
        self.conv_init = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) ##112 112
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #DenseBlock 1
        self.conv1_1 =  ConvBlock(64, bn_size, growth_rate) #in_channel
        self.conv1_2 =  ConvBlock(32, bn_size, growth_rate)
        self.conv1_3 =  ConvBlock(64, bn_size, growth_rate)
        self.conv1_4 =  ConvBlock(96, bn_size, growth_rate)
        self.conv1_5 =  ConvBlock(128, bn_size, growth_rate)
        self.conv1_6 =  ConvBlock(160, bn_size, growth_rate)

        #TransitionLayer1
        self.tran1 = TransitionLayer(160+growth_rate, (160+growth_rate)//2) #in_channel192, out_channel 96

        #DenseBlock 2
        self.conv2_1 =  ConvBlock(96, bn_size, growth_rate) #in_channel
        self.conv2_2 =  ConvBlock(32, bn_size, growth_rate)
        self.conv2_3 =  ConvBlock(64, bn_size, growth_rate)
        self.conv2_4 =  ConvBlock(96, bn_size, growth_rate)
        self.conv2_5 =  ConvBlock(128, bn_size, growth_rate)
        self.conv2_6 =  ConvBlock(160, bn_size, growth_rate)
        self.conv2_7 =  ConvBlock(192, bn_size, growth_rate)
        self.conv2_8 =  ConvBlock(224, bn_size, growth_rate)
        self.conv2_9 =  ConvBlock(256, bn_size, growth_rate)
        self.conv2_10 =  ConvBlock(288, bn_size, growth_rate)
        self.conv2_11 =  ConvBlock(320, bn_size, growth_rate)
        self.conv2_12 =  ConvBlock(352, bn_size, growth_rate)

        #TransitionLayer2
        self.tran2 = TransitionLayer(352+growth_rate, (352+growth_rate)//2) #in_channel192, out_channel 96

        #Dense3
        self.conv3_1 = ConvBlock(192, bn_size, growth_rate)
        self.dense3 = nn.Sequential(
            *[
                ConvBlock(i*growth_rate, bn_size, growth_rate)
            for i in range(1, 24)]
            )

        #TransitionLayer
        self.tran3= TransitionLayer(736+growth_rate, (736+growth_rate)//2) #in_channel192, out_channel 96

        #Dense4
        self.conv4_1 = ConvBlock(384, bn_size, growth_rate)
        self.dense4 = nn.Sequential(
            *[
                ConvBlock(i*growth_rate, bn_size, growth_rate)
            for i in range(1, 16)]
            )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1000),
        )
 


    def forward(self, x) :
        out = self.conv_init(x)
        out = self.max_pool(out)

        out = self.conv1_1(out, skip_connection=False)
        print(out.shape)
        out = self.conv1_2(out)
        print(out.shape)
        out = self.conv1_3(out)
        print(out.shape)
        out = self.conv1_4(out)
        print(out.shape)
        out = self.conv1_5(out)
        print(out.shape)
        out = self.conv1_6(out)
        print(out.shape)
        out = self.tran1(out)
        print(out.shape)

        out = self.conv2_1(out, skip_connection=False)
        print(out.shape)
        out = self.conv2_2(out)
        print(out.shape)
        out = self.conv2_3(out)
        print(out.shape)
        out = self.conv2_4(out)
        print(out.shape)
        out = self.conv2_5(out)
        print(out.shape)
        out = self.conv2_6(out)
        print(out.shape)
        out = self.conv2_7(out)
        print(out.shape)
        out = self.conv2_8(out)
        print(out.shape)
        out = self.conv2_9(out)
        print(out.shape)
        out = self.conv2_10(out)
        print(out.shape)
        out = self.conv2_11(out)
        print(out.shape)
        out = self.conv2_12(out)
        print(out.shape)
        out = self.tran2(out)
        print(out.shape)


        out = self.conv3_1(out, skip_connection=False)
        out = self.dense3(out)
        print(out.shape)
        out = self.tran3(out)
        print(out.shape)

        out = self.conv4_1(out, skip_connection=False)
        out = self.dense4(out)
        print(out.shape)

        out = self.classifier(out)
        print(out.shape)

        return out


if __name__ == "__main__" :
    dummy_input = torch.ones(1, 3, 224, 224) #b c w h
    densenet = DenseNet()
    densenet(dummy_input)








"""






////////////////////////////////////////////////////////////////

그로스 레이트

<dense connectivity>
dense block 안에서 conv 끼리 channelwise concatenation

hl이란 batchnorm, pooling, conv
다이렉트 커넥션, 피처맵 연결

<풀링레이어 특징>
다운샘플링 레이어 : 피쳐맵 크기 조절하기 위함
다운샘플링을 수행하기 위해 어러댄스블락으로 쪼갠것
멀티플 댄스블락이라하는데

인접한 두 블락 사이의 컨볼루션과 풀링레이어를
트랜지션 레이어라고 한다
피처맵 사이즈 변경하게됨


<그로스 레이트>
 각 계층에서 생성되는 특성들이 다음 계층으로 직접 전달되며,
 정보와 그래디언트가 효율적으로 통과할 수 있게

하나의 댄스블락은 
hl이 k 개의 피쳐맵을 생성한다면 
k=4
k0 + k*(l-1) 개의 인풋 피쳐맵을 가지게 된다

k0는 시작부분의 피쳐맵크기
k는 각 레이어에서 추가되는 feturemap의 수
l은 레이어의 수

k = 32일때
l = 6일때
64 + 32 + 32 + 32 + 32 + 32 = 224


<보틀낵 레이어>
각 레이어는 k개의 피쳐맵을 생성하지만
입력 피쳐맵의 개수를 줄임
댄스넷의 피쳐맵 향상에 크게 기여 
densenet-b 가 보틀넥 도입

<컴포짓 펑션 >
에티베이션 배치와 성능에 대한 연구
컨볼루션 연산 앞에 배치



"""