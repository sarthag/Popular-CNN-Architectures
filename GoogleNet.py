import torch
from torch import nn

#INPUT SIZE: 3 x 224 x 224 
#Contains normal conv blocks and inception blocks to test best filter size 
#INCEPTION BLOCK:
#1x1 conv block , 1x1 + 3x3, 1x1 + 5x5, pooling
#total filetrs = sum of filters of each layer
#NETWORK:
#224 x 224 x3 (kernel = 7, stride = 2 , padding = 3) -> 
#112 x 112 x 64 (maxpool , kernel = 3, stride = 2) -> 
# 56 x 56 x64 (kernel = 3, stride = 1 ,pad = 1) -> 
# 56 X 56 X 192 (inception 3a and 3b) -> 28 x 28 x256 -> 
# 28 x 28 x 480 (maxpool , kernel = 3, stride = 2) -> 
# 14 x 14 x 480 (inception 4 a, b, c ,d ,e) -> 
# 14 x 14 x 832 (maxpool , kernel = 3, stride = 2) -> 
# 7 x 7 x 832 (inception 5a, b) -> 
# 7 x 7 x 1024 (avgpool, kernel = 7, stride = 1) -> 
# 1 x 1 x 1024 -> dropout(0.4) -> linear (1000) -> 1x 1000 -> softmax    


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
    
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1, stride = 1)

        self.branch2 = nn.Sequential(conv_block(in_channels, red_3x3, kernel_size=1, stride = 1),
                                     conv_block(red_3x3, out_3x3, kernel_size=3, padding=1, stride = 1))

        self.branch3 = nn.Sequential(conv_block(in_channels, red_5x5, kernel_size=1),
                                     conv_block(red_5x5, out_5x5, kernel_size=5,stride = 1, padding=2))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     conv_block(in_channels, out_1x1pool, kernel_size=1, stride =1 ))

    
    def forward(self, x):
        #N x filters X 28 x 28
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x



class GoogleNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes=1000):
        super(GoogleNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv_block(in_channels=in_channels,out_channels = 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)

        self.inception3 = nn.Sequential(Inception_block(192, 64, 96, 128, 16, 32, 32),
                                        Inception_block(256, 128, 128, 192, 32, 96, 64))

        self.inception4 = nn.Sequential(Inception_block(480, 192, 96, 208, 16, 48, 64),
                                        Inception_block(512, 160, 112, 224, 24, 64, 64),
                                        Inception_block(512, 128, 128, 256, 24, 64, 64),
                                        Inception_block(512, 112, 144, 288, 32, 64, 64),
                                        Inception_block(528, 256, 160, 320, 32, 128, 128))

        self.inception5 = nn.Sequential(Inception_block(832, 256, 160, 320, 32, 128, 128),
                                        Inception_block(832, 384, 192, 384, 48, 128, 128))

        self.finalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.inception3(x)
        x = self.pool(x)

        x = self.inception4(x)
        x = self.pool(x)
        x = self.inception5(x)
        x = self.finalpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.output(x)
        
        return x


'''
#Basic Test
x = torch.randn(3, 3, 224, 224)
model = GoogleNet()
print(model(x).shape)
#output = (3,1000)
'''