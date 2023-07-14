import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, branch1x1_channels, branch3x3_reduce_channels,
                 branch3x3_channels, branch5x5_reduce_channels, branch5x5_channels, branch_pool_channels):
        super(InceptionBlock, self).__init__()
        
        self.branch1x1 = ConvBlock(in_channels, branch1x1_channels, kernel_size=1, stride=1, padding=0)
        
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, branch3x3_reduce_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(branch3x3_reduce_channels, branch3x3_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, branch5x5_reduce_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(branch5x5_reduce_channels, branch5x5_channels, kernel_size=5, stride=1, padding=2)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, branch_pool_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)

class ReductionBlock(nn.Module):
    def __init__(self, in_channels, branch3x3_reduce_channels, branch3x3_channels,
                 branch7x7_reduce1_channels, branch7x7_reduce2_channels, branch7x7_channels):
        super(ReductionBlock, self).__init__()
        
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, branch3x3_reduce_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(branch3x3_reduce_channels, branch3x3_channels, kernel_size=3, stride=2, padding=1)
        )
        
        self.branch7x7 = nn.Sequential(
            ConvBlock(in_channels, branch7x7_reduce1_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(branch7x7_reduce1_channels, branch7x7_reduce2_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(branch7x7_reduce2_channels, branch7x7_channels, kernel_size=3, stride=2, padding=1)
        )
        
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch7x7, branch_pool]
        return torch.cat(outputs, dim=1)


class InceptionResnetV1(nn.Module):
    def __init__(self, pretrained=False):
        super(InceptionResnetV1, self).__init__()
        
        # Stem block
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 80, kernel_size=1, stride=1, padding=0),
            ConvBlock(80, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception blocks
        self.inception1 = InceptionBlock(192, 32, 32, 32, 32, 48, 64)
        self.inception2 = InceptionBlock(256, 64, 64, 64, 64, 96, 96)
        self.inception3 = InceptionBlock(384, 64, 64, 96, 64, 96, 96)
        
        # Reduction blocks
        self.reduction1 = ReductionBlock(576, 384, 256, 256, 384)
        self.reduction2 = ReductionBlock(1152, 192, 384, 384, 512)
        
        # Face embedding layer
        self.embedding_layer = nn.Linear(1792, 512)
        
        # Initialize weights
        self._init_weights()
        
        # Load pre-trained weights if specified
        if pretrained:
            self.load_state_dict(torch.load('inception_resnet_v1.pt'))
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.reduction1(x)
        x = self.reduction2(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)