import torch
import torch.nn as nn 

#Skip connections, never forgets prev connections 
#7x7 conv, 64 channels and stride of 2, padding of 3
#3x3 pool with stride 2 
#Resnet layers 

class ResNet_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride= 1):
        super(ResNet_block, self).__init__()
        
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1,  stride = 1, padding=0, bias= False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample 
        
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.bn2(self.conv3(x))
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x = x + identity
        x = self.relu(x)
        return x
    
    

class ResNet(nn.Module):
    def __init__(self, ResNet_block, layers, in_channels, num_classes): #layers is list of repeats
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding= 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #Resnet
        self.layer1 = self._make_layer(ResNet_block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(ResNet_block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(ResNet_block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(ResNet_block, layers[3], out_channels=512, stride=2)
        
        self.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.output = nn.Linear(512*4, num_classes) 
        
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        
        return x
    
        
    def _make_layer(self, ResNet_block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride = stride, bias= False),
                                                nn.BatchNorm2d(out_channels*4))
            
        layers.append(ResNet_block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(num_residual_blocks - 1):
            layers.append(ResNet_block(self.in_channels, out_channels))  #256 -> 64, 64*4 
            
        return nn.Sequential(*layers)
        
        
    


def ResNet50(image_channles = 3, num_classes = 1000):
    return ResNet(ResNet_block, [3,4,6,3], image_channles, num_classes)


def ResNet101(image_channles = 3, num_classes = 1000):
    return ResNet(ResNet_block, [3,4,23,3], image_channles, num_classes)


def ResNet152(image_channles = 3, num_classes = 1000):
    return ResNet(ResNet_block, [3,8,36,3], image_channles, num_classes)

'''
#Basic Test
x = torch.randn(3, 3, 224, 224)
model = ResNet50()
print(model(x).shape)

model = ResNet101()
print(model(x).shape)

model = ResNet152()
print(model(x).shape)

#output = (3,1000)
'''