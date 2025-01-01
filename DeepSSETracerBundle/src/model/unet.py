import torch
import torch.nn as nn


def conv_block_3d_down(in_dim, mid_dim, out_dim, activation):
    
    conv_down_1 = nn.Conv3d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_down_1.weight, std = 0.1)
    conv_down_2 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_down_2.weight, std = 0.1)
    return nn.Sequential(
        conv_down_1, 
        nn.BatchNorm3d(mid_dim),
        activation,
        conv_down_2,
        nn.BatchNorm3d(out_dim),
        activation,                
        )


def conv_trans_block_3d(in_dim):
    return nn.ConvTranspose3d(in_dim, in_dim, kernel_size=2, stride=2, padding=0, output_padding=0)


def max_pooling_3d():
    return nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

    
def conv_block_3d_up(in_dim, out_dim, activation):
    conv_up_1 = nn.Conv3d(in_dim+out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_up_1.weight, std = 0.1)
    conv_up_2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_up_2.weight, std = 0.1)
    return nn.Sequential(
        conv_up_1,
        nn.BatchNorm3d(out_dim),
        activation,
        conv_up_2,
        nn.BatchNorm3d(out_dim),
        activation,
        nn.Dropout(p=0.5, inplace=True))

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        activation = nn.ReLU()
        # Down sampling
        self.down_1 = conv_block_3d_down(1, 32, 64, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_3d_down(64, 64, 128, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_3d_down(128, 128, 256, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(256)
        self.up_1 = conv_block_3d_up(256, 128, activation)
        self.trans_2 = conv_trans_block_3d(128)
        self.up_2 = conv_block_3d_up(128, 64, activation)
#        
        # Output
        self.out = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)
        torch.nn.init.normal_(self.out.weight, std = 0.1)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1) 
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2) 

        # Up sampling
        trans_1 = self.trans_1(down_3) 
        concat_1 = torch.cat([trans_1, down_2], dim=1)
        up_1 = self.up_1(concat_1) 
        trans_2 = self.trans_2(up_1) 
        concat_2 = torch.cat([trans_2, down_1], dim=1) 
        
        up_2 = self.up_2(concat_2) 
        # Output
        out = self.out(up_2) 
        return out


class Gem_UNet(nn.Module):
    def __init__(self, args):
        super(Gem_UNet, self).__init__()
        self.unet = UNet()
        self.gpu = args.cuda

    def forward(self, x):
        output = self.unet(x)
        return output