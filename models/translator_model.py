import torch
import torch.nn as nn

#Based on UNet in: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNet(torch.nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, in_c = 3, out_c = 3, outer_nc=64, inner_nc=512, dropout=False):
        """[summary]

        Args:
            in_c (int, optional): [description]. Defaults to 3.
            out_c (int, optional): [description]. Defaults to 3.
            outer_nc (int, optional): [description]. Defaults to 64.
            inner_nc (int, optional): [description]. Defaults to 512.
            dropout (bool, optional): [description]. Defaults to False.
        """        
        # """Construct a Unet generator

        # Args:
        #     in_c (int): the number of channels in input image
        #     out_c (int): the number of channels in output image
        #     num_downs (int): the number of downsamplings in UNet. For example, # if |num_downs| == 7,
        # #                         image of size 128x128 will become of size 1x1 # at the bottleneck
        #     n_filters (int): the number of filters in the last conv layer
        #     norm_layer (module, optional): normalization layer. Defaults to nn.BatchNorm2d.
        #     dropout (bool, optional): use dropout or not. Defaults to False.
        # """        
        
        super(UNet, self).__init__()

        #* For left UNet
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.module1l = sub_left(in_nc, outer_nc, relu=None, norm=None)
        self.module2l = sub_left(outer_nc, outer_nc * 2, relu=self.leaky_relu, nn.BatchNorm2d(outer_nc * 2))
        self.module3l = sub_left(outer_nc * 2, outer_nc * 4, relu=self.leaky_relu, norm=nn.BatchNorm2d(outer_nc * 4))
        self.module4l = sub_left(outer_nc * 4, inner_nc, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module5l = sub_left(inner_nc, inner_nc, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module6l = sub_left(inner_nc, inner_nc, kernel_size = 3, relu=self.leaky_relu, norm=None)

        #* For right UNet
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.module6r = sub_right(inner_nc + 1, inner_nc, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module5r = sub_right(inner_nc, inner_nc, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module4r = sub_right(inner_nc, inner_nc / 2, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module3r = sub_right(inner_nc / 2, inner_nc / 4, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module2r = sub_right(inner_nc / 4, inner_nc / 8, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module1r = sub_right(inner_nc / 8, out_c, relu=self.relu, norm=None, tanh=self.tanh, dropout=dropout)



        # self.leftmodule = UNetLeft()
        # self.rightmodule = UNetRight(dropout)

        # (3 * 224 * 224) -> (64 * 112 * 112) -> (128 * 56 * 56) -> (256 * 28 * 28) -> (512 * 14 * 14) -> (512 * 7 * 7) -> (512 * 4 * 4)
        # (513 * 4 * 4) -> (512 * 7 * 7) -> (512 * 14 * 14) -> (256 * 28 * 28) -> (128 * 56 * 56)-> (64 * 112 * 112) -> (3 * 224 * 224) 


    def forward(self,x,embed):
        """forward pass of UNet

        Args:
            x (torch.tensor): input image to the network
            embed (torch.tensor): the expression embedding network associated with the input image

        Returns:
            torch.tensor: output of the network
        """        
        x_1l = self.module1l(x)
        x_2l = self.module2l(x)
        x_3l = self.module3l(x)
        x_4l = self.module4l(x)
        x_5l = self.module5l(x)
        x_6l = self.module6l(x)
        embed = embed.view(-1, 1, 4, 4)
        x = torch.cat([x_6l, embed], 1)

        x = torch.cat([self.module6r(x), x_5l], 1)
        x = torch.cat([self.module5r(x), x_4l], 1)
        x = torch.cat([self.module4r(x), x_3l], 1)
        x = torch.cat([self.module3r(x), x_2l], 1)
        x = torch.cat([self.module2r(x), x_1l], 1)
        x = self.module1r(x)
        return x

        # x = self.leftmodule(x,embed)
        # x = self.rightmodule(x)
        return x

class sub_left(nn.Module):
    # Input image size = 3 * 224 * 224
    # (3 * 224 * 224) -> (64 * 112 * 112) -> (128 * 56 * 56) -> (256 * 28 * 28) -> (512 * 14 * 14) -> (512 * 7 * 7) -> (512 * 4 * 4)
    # For (512 * 7 * 7) -> (512 * 4 * 4), use kernel size 3 with other parameters same? Or kernel size 4 with no stride?

    # Block 1l: (3 * 224 * 224) -> (64 * 224 * 224)
    # Block 2l: (64 * 112 * 112) -> (128 * 112 * 112)
    # Block 3l: (128 * 56 * 56) -> (256 * 56 * 56)
    # Block 4l: (256 * 28 * 28) -> (512 * 28 * 28)
    # Block 5l: (512 * 14 * 14)
    # Block 6l: (512 * 7 * 7)
    # Block 7 (ceil): (512 * 4 * 4)
    
    def __init__(self, in_nc, out_nc, kernel_size=4, padding=1, stride=2, **kwargs):
        """[summary]

        Args:
            in_nc ([type]): [description]
            out_nc ([type]): [description]
            kernel_size (int, optional): [description]. Defaults to 4.
            padding (int, optional): [description]. Defaults to 1.
            stride (int, optional): [description]. Defaults to 2.
        """        
        super(sub_left, self).__init__()

        self.layer = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, bias=False)
        model = []
        if kwargs["relu"] != None:
            model.append(kwargs["relu"])
        model.append(self.layer)
        if kwargs["norm"] != None:
            model.append(kwargs["norm"])
        self.model = nn.Sequential(*model)
    def forward(self,x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self.model(x)

class sub_right(nn.Module):
    # Starting image size = 512 * 4 * 4
    # (513 * 4 * 4) -> (512 * 7 * 7) -> (512 * 14 * 14) -> (256 * 28 * 28) -> (128 * 56 * 56)-> (64 * 112 * 112) -> (3 * 224 * 224) 
    # For (512 * 4 * 4) -> (512 * 7 * 7), use kernel size 3 with other parameters same?

    # Block 6r: (512 * 8 * 8) -> (256 * 7 * 7) ***** ONLY PAD ONCE *****
    # Block 5r: (256 * 14 * 14) -> (128 * 14 * 14)
    # Block 4r: (128 * 28 * 28) -> (64 * 28 * 28)
    # Block 3r: (64 * 56 * 56) -> (32 * 56 * 56)
    # Block 2r: (32 * 112 * 112) -> (16 * 112 * 112)
    # Block 1r: (16 * 224 * 224) -> (8 * 224 * 224)

    # Output segmentation map: (3 * 224 * 224)

    def __init__(self, in_nc, out_nc, kernel_size=4, padding=1, stride=2, **kwargs):
        """[summary]

        Args:
            in_nc ([type]): [description]
            out_nc ([type]): [description]
            kernel_size (int, optional): [description]. Defaults to 4.
            padding (int, optional): [description]. Defaults to 1.
            stride (int, optional): [description]. Defaults to 2.
        """        
        super(sub_right, self).__init__()

        self.layer = nn.ConvTranspose2d(in_nc, out_nc, kernel_size, stride, padding, bias=False)
        self.dropout = nn.Dropout(0.5)
        model = []
        if kwargs["relu"] != None:
            model.append(kwargs["relu"])
        model.append(self.layer)
        if kwargs["tanh"] != None:
            model.append(kwargs["tanh"])
        elif kwargs["norm"] != None:
            model.append(kwargs["norm"])
        if kwargs["dropout"]:
            model.append(self.dropout)
        self.model = nn.Sequential(*model)

    def forward(self,x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self.model(x)
