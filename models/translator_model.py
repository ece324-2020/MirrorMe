import torch
import torch.nn as nn

#Based on UNet in: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNet(torch.nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, in_c, out_c, num_downs, n_filters, norm_layer=nn.BatchNorm2d, dropout=False):
        """Construct a Unet generator

        Args:
            in_c (int): the number of channels in input image
            out_c (int): the number of channels in output image
            num_downs (int): the number of downsamplings in UNet. For example, # if |num_downs| == 7,
        #                         image of size 128x128 will become of size 1x1 # at the bottleneck
            n_filters (int): the number of filters in the last conv layer
            norm_layer (module, optional): normalization layer. Defaults to nn.BatchNorm2d.
            dropout (bool, optional): use dropout or not. Defaults to False.
        """        
        super(UNet, self).__init__()

        pass

    def forward(self,input,embed):
        """forward pass of UNet

        Args:
            input (torch.tensor): input image to the network
            embed (torch.tensor): the expression embedding network associated with the input image

        Returns:
            torch.tensor: output of the network
        """        

        return True

class UNetLeft(nn.Module):

    def __init__(self, inner_nc = 512, out_nc = 64, input_nc=3, embedding=None, dropout=False):
        """ Architecture of left/right in the UNet

        Args:
            inner_nc (int): the number of filters in the bottom section of the U
            outer_nc (int): the number of filters in the top section of the U
            input_nc (int, optional): the number of channels in input images/features. Defaults to 3.
            embedding (torch.tensor, optional): [description]. Defaults to None.
            submodule (nn.Module, optional): previously defined submodules. Defaults to None.
            outermost (bool, optional): if this module is the outermost block in Unet. Defaults to False.
            innermost (bool, optional): if this module is the innermost block in Unet. Defaults to False.
            norm_layer (module, optional): normalization layer. Defaults to nn.BatchNorm2d.
            dropout (bool, optional): use dropout or not. Defaults to False.
        """        
        super(UNetLeft,self).__init__()

        self.relu = nn.LeakyReLU(0.2)

        self.module1 = sub_left(input_nc, outer_nc, relu=None, norm=None)
        self.module2 = sub_left(outer_nc, outer_nc * 2, relu=self.relu, nn.BatchNorm2d(outer_nc * 2))
        self.module3 = sub_left(outer_nc * 2, outer_nc * 4, relu=self.relu, norm=nn.BatchNorm2d(outer_nc * 4))
        self.module4 = sub_left(outer_nc * 4, inner_nc, relu=self.relu, norm=nn.BatchNorm2d(inner_nc))
        self.module5 = sub_left(inner_nc, inner_nc, relu=self.relu, norm=nn.BatchNorm2d(inner_nc))
        self.module6 = sub_left(inner_nc, inner_nc, relu=self.relu, norm=None)

    def forward(self, x):
        return self.model(x)

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
    
    def __init__(self, in_nc, out_nc, kernel_size=4, padding=1, stride=2,**kwargs):
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
        return self.model(x)

class sub_right(nn.Module):
    # Starting image size = 512 * 4 * 4
    # (512 * 4 * 4) -> (512 * 7 * 7) -> (512 * 14 * 14) -> (256 * 28 * 28) -> (128 * 56 * 56)-> (64 * 112 * 112) -> (3 * 224 * 224) 
    # For (512 * 4 * 4) -> (512 * 7 * 7), use kernel size 3 with other parameters same?

    # Block 6r: (512 * 8 * 8) -> (256 * 7 * 7) ***** ONLY PAD ONCE *****
    # Block 5r: (256 * 14 * 14) -> (128 * 14 * 14)
    # Block 4r: (128 * 28 * 28) -> (64 * 28 * 28)
    # Block 3r: (64 * 56 * 56) -> (32 * 56 * 56)
    # Block 2r: (32 * 112 * 112) -> (16 * 112 * 112)
    # Block 1r: (16 * 224 * 224) -> (8 * 224 * 224)

    # Output segmentation map: (3 * 224 * 224)

    def __init__(self, in_nc, out_nc, kernel_size=4, padding=1, stride=2, is_embed=False, **kwargs):
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

    def forward(self,embed):
        pass
