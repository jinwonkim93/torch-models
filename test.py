import torch

class RESNET(torch.nn.Module):
    pass

dummy = torch.zeros((1, 3, 224, 224))
conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
modules_list = []
for _ in range(6):
    modules_list.append(torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
conv2_x = torch.nn.Sequential(*modules_list)


modules_list = []
for i in range(8):
    if i == 0:
        modules_list.append(torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
    else:
        modules_list.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
conv3_x = torch.nn.Sequential(*modules_list)

modules_list = []
for i in range(12):
    if i == 0:
        modules_list.append(torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
    else:
        modules_list.append(torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
conv4_x = torch.nn.Sequential(*modules_list)

modules_list = []
for i in range(6):
    if i == 0:
        modules_list.append(torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
    else:
        modules_list.append(torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
conv5_x = torch.nn.Sequential(*modules_list)

avg_pool = torch.nn.AvgPool2d(kernel_size=7,stride=1)
fc = torch.nn.Linear(in_features= 512, out_features=1000)

###
# import torch.nn as nn

# modules = []
# modules.append(nn.Linear(10, 10))
# modules.append(nn.Linear(10, 10))

# sequential = nn.Sequential(*modules)
####


h = conv1(dummy)
h2 = maxpool(h)
h3 = conv2_x(h2)
h4 = conv3_x(h3)
h5 = conv4_x(h4)
h6 = conv5_x(h5)
h7 = avg_pool(h6)
h8 = torch.nn.Flatten()(h7)
h9 = fc(h8)



print(h.shape)
print(h2.shape)
print(h3.shape)
print(h4.shape)
print(h5.shape)
print(h6.shape)
print(h7.shape)
print(h8.shape)
print(h9.shape)


#modulist 는 forward, sequncial 
#conv

#input data 1b, 3f, 224w, 224h
#in channel(rgb) 3 -> out_channel(feature) 64 #kernel_size 7 -> 이미지 가로 세로 7x7 #stride 2 #padding 4
#torch.Size([1, 64, 110, 110])

#fillter
#maxpooling

torch.Size([1, 64, 112, 112])
torch.Size([1, 64, 17, 17])


def forward(x, skip_connection):
    h = conv1(x)
    h += skip_connection
    h = conv2(h)
    return h


class ResBlock(torch.nn.Module):
    def __init__(self, n_layers=4):
        modules_list = []
        for _ in range(n_layers):
                modules_list.append(torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        for module in self.modules_list:
            #TODO do skip connection
            h = module(x)


class ResNet(torch.nn.Module):
    def __init__(self):
        
        self.conv1_1 = 1
        self.conv2_2 = 2

    def forward(self, x, skip_connection):
        h = self.conv1(x)
        h += skip_connection
        h = self.conv2(h)
        return h



"""
modules_list = []
for i in range(1):
    if i == 0:
        modules_list.append(torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
    else:
        modules_list.append(torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
conv5_x = torch.nn.Sequential(*modules_list)

for i in range(5):
    if i == 0:
        modules_list.append(torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
    else:
        modules_list.append(torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
conv5_x = torch.nn.Sequential(*modules_list)

"""