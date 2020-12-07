# v3: increase depth of Unet: kernel size = 2, stride = 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from .model_utils import init_model

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

        self.module1l = sub_left(in_c, outer_nc, relu=None, norm=None)
        self.module2l = sub_left(outer_nc, outer_nc * 2, relu=self.leaky_relu, norm=nn.BatchNorm2d(outer_nc * 2))
        self.module3l = sub_left(outer_nc * 2, outer_nc * 4, relu=self.leaky_relu, norm=nn.BatchNorm2d(outer_nc * 4))
        self.module4l = sub_left(outer_nc * 4, inner_nc, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module5l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module6l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module7l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module8l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module9l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module10l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module11l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module12l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module13l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        self.module14l = sub_left(inner_nc, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.leaky_relu, norm=None)
        
        # (self, in_nc, out_nc, kernel_size=2, padding=0, stride=1, **kwargs):

        # self.module5l = sub_left(inner_nc, inner_nc, relu=self.leaky_relu, norm=nn.BatchNorm2d(inner_nc))
        # self.module6l = sub_left(inner_nc, inner_nc, kernel_size = 3, relu=self.leaky_relu, norm=None)

        #* For right UNet
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        #we take out inner_nc+1
        self.module14r = sub_right(inner_nc + 1, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module13r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module12r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module11r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module10r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module9r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module8r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module7r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module6r = sub_right(inner_nc*2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module5r = sub_right(inner_nc * 2, inner_nc, kernel_size = 2, padding = 0, stride = 1, relu=self.relu, norm=nn.BatchNorm2d(inner_nc), tanh=None, dropout=dropout)
        self.module4r = sub_right(inner_nc * 2, inner_nc // 2, relu=self.relu, norm=nn.BatchNorm2d(inner_nc // 2), tanh=None, dropout=dropout)
        self.module3r = sub_right(inner_nc, inner_nc // 4, relu=self.relu, norm=nn.BatchNorm2d(inner_nc // 4), tanh=None, dropout=dropout)
        self.module2r = sub_right(inner_nc // 2, inner_nc // 8, relu=self.relu, norm=nn.BatchNorm2d(inner_nc // 8), tanh=None, dropout=dropout)
        self.module1r = sub_right(inner_nc // 4, out_c, relu=self.relu, norm=None, tanh=self.tanh, dropout=dropout)

        # self.leftmodule = UNetLeft()
        # self.rightmodule = UNetRight(dropout)

        # (3 * 224 * 224) -> (64 * 112 * 112) -> (128 * 56 * 56) -> (256 * 28 * 28) -> (512 * 14 * 14) -> (512 * 7 * 7) -> (512 * 4 * 4)
        # (513 * 4 * 4) -> (512 * 7 * 7) -> (512 * 14 * 14) -> (256 * 28 * 28) -> (128 * 56 * 56)-> (64 * 112 * 112) -> (3 * 224 * 224) 


    def forward(self, x, embed):
        """forward pass of UNet

        Args:
            x (torch.tensor): input image to the network
            embed (torch.tensor): the expression embedding network associated with the input image

        Returns:
            torch.tensor: output of the network
        """        
        #* Concatenate source and target
        #x = torch.cat([source,target],1)
        x_1l = self.module1l(x) # Output: (64 * 112 * 112)
        x_2l = self.module2l(x_1l) # (128 * 56 * 56)
        x_3l = self.module3l(x_2l) # (256 * 28 * 28)
        x_4l = self.module4l(x_3l) # (512 * 14 * 14)
        x_5l = self.module5l(x_4l) # (512 * 13 * 13)
        x_6l = self.module6l(x_5l) # (512 * 12 * 12)
        x_7l = self.module7l(x_6l) # (512 * 11 * 11)
        x_8l = self.module8l(x_7l) # (512 * 10 * 10)
        x_9l = self.module9l(x_8l) # (512 * 9 * 9)
        x_10l = self.module10l(x_9l) # (512 * 8 * 8)
        x_11l = self.module11l(x_10l) # (512 * 7 * 7)
        x_12l = self.module12l(x_11l) # (512 * 6 * 6)
        x_13l = self.module13l(x_12l) # (512 * 5 * 5)
        x_14l = self.module14l(x_13l) # (512 * 4 * 4)

        embed = embed.view(-1, 1, 4, 4)
        x = torch.cat([x_14l, embed], 1)
        #x = x_14l

        r14 = self.module14r(x)
        x = torch.cat([r14, x_13l], 1)
        r13 = self.module13r(x)
        x = torch.cat([r13, x_12l], 1)
        r12 = self.module12r(x)
        x = torch.cat([r12, x_11l], 1)
        r11 = self.module11r(x)
        x = torch.cat([r11, x_10l], 1)
        r10 = self.module10r(x)
        x = torch.cat([r10, x_9l], 1)
        r9 = self.module9r(x)
        x = torch.cat([r9, x_8l], 1)
        r8 = self.module8r(x)
        x = torch.cat([r8, x_7l], 1)
        r7 = self.module7r(x)
        x = torch.cat([r7, x_6l], 1)
        r6 = self.module6r(x)
        x = torch.cat([r6, x_5l], 1)
        r5 = self.module5r(x)
        x = torch.cat([r5, x_4l], 1)
        r4 = self.module4r(x)
        x = torch.cat([r4, x_3l], 1)
        r3 = self.module3r(x)
        x = torch.cat([r3, x_2l], 1)
        r2 = self.module2r(x)
        x = torch.cat([r2, x_1l], 1)
        r1 = self.module1r(x)  
        x = self.module1r(x)
        # return x

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

        self.layer = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
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

def define_T(in_c = 3, out_c = 3, outer_nc=64, inner_nc=512, dropout=False, gpu_ids=[0], init_type='normal'):
    model = UNet(in_c=in_c, out_c=out_c, outer_nc=outer_nc, inner_nc=inner_nc, dropout=dropout)
    model = init_model(model, init_type=init_type, gpu_ids=gpu_ids)

    return model

def main():
    root = '/home/MirrorMe/project/test_img/'
    batch_size = 2
    epochs = 200
    num_workers = 6

    transform = transforms.Compose([#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                                   transforms.Resize((224, 224)),
                                   #transforms.RandomRotation(45),
                                   transforms.ToTensor()])
    
    dataset = dset.ImageFolder(root=root, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_emb = torch.rand((1, 1, 16))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.L1Loss()

    for epoch in range(epochs):
        print("************ Epoch {} *************\n".format(epoch+1))
        for i, batch in enumerate(dataloader, 0):
            optimizer.zero_grad()
            features, label = batch

            # print("Features: ", features.shape)
            features = features.to(device)
            out = model(features, test_emb.to(device))

            batch_loss = loss_function(input = out, target = features)
            batch_loss.backward()
            optimizer.step()

            print("Loss: {}".format(batch_loss.item()))
            # print("OUT: ", out.shape)
            if ((epoch+1) % 5 == 0):
                vutils.save_image(out[0], '/home/MirrorMe/project/test_img/{}.png'.format(epoch+1))


if __name__ == '__main__':
    main()