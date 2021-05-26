import torch 
import torch.nn as nn

#32 x 32 (Kernel = 5, stride = 1, padding 0) ->
#6 x 28 x 28 (AvgPool = 2, stride = 2)->
#6 x 14 x 14 (Kernel = 5, stride = 1, padding 0)  ->
#16 x 10 x 10 (AvgPool = 2, stride = 2)->
#16 x 5 x 5 (Kernel = 5, stride = 1, padding 0) ->
#120 x 1 x 1 (fc) -> 84 (fc) -> classes


class LeNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 10):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = 5, stride = 1, padding = 0 )
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0 )
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1, padding = 0 )
        self.fc = nn.Linear(120, 84)
        self.output = nn.Linear(84, out_channels)
    
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc(x))
        x = self.output(x)
        
        return x
        

'''
#basic test        
x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)
#Output = (64,10)
'''