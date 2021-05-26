import torch
import torch.nn as nn

#INPUT: 3 x 224 x 224 images 
#kernel size  = 3,  padding = 1, stride = 1, dimentions preserved
#maxpool over kernet_size = 2 and stride = 2
#VGG -> 11, 13, 16 ,19 
#General Arch: 
# 3 -> 64 -> pool -> 128 -> pool -> 256 -> pool -> 512 -> pool -> 512 -> pool -> 
# fc 4096 -> fc 4096 -> fc 1000 -> softmax


VGG11 = [64 , "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG13 = [64 , 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG16 = [64, 64, "M", 128 ,128, "M", 256, 256 ,256, "M" , 512, 512, 512, "M" , 512, 512, 512, "M"]
VGG19 = [64, 64, "M", 128 ,128, "M", 256, 256 ,256, 256, "M" , 512, 512, 512, 512, "M" , 512, 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv = self.create_conv_layers(VGG16)
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(4096, 4096),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(4096, num_classes))
    
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        
        return x
    
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x        
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=1,  padding= 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
        return nn.Sequential(*layers) 
        
        
''''        
#Basic Test
x = torch.randn(1, 3, 224, 224)
model = VGG()
print(model(x).shape)
#Output = (1,1000)
'''