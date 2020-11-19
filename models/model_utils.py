import torch

#Utility functions common to all models

#Credits to CycleGAN
#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
def init_weights(model, init_type='normal', gain=0.02):
    def init_func(mod):
        classname = mod.__class__.__name__
        if hasattr(mod, 'weight') and (classname.find('Conv') != -1 \
            or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(mod.weight.data, 0.0, gain)

        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(mod.weight.data, 1.0, gain)
            torch.nn.init.constant_(mod.bias.data, 0.0)

    print('Initalizing model with %s' % init_type)
    net.apply(init_func)

#Credits to CycleGAN
#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
#I just called it init_model instead of network to make it consistent with the rest
#DataParallel for now maybe DistributedDataParallel later
def init_model(model, init_type='normal', gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = torch.nn.DataParallel(model, gpu_ids)

    init_weights(model, init_type, gain)

    return model