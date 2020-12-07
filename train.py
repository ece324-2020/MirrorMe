import time
import os
import argparse
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# from .reflection_gan_options import load_options_from_yaml
#from models. import *
from MirrorMe.models.reflection_gan_options import load_options_from_yaml
from MirrorMe.models.reflection_gan_model import ReflectionGAN
from MirrorMe.GAN_DataLoader2 import dataloader
from MirrorMe.models.discriminator_model import *
from MirrorMe.models.translator_model_v3 import *
from FECNet.models.FECNet import FECNet
#from torch.utils.data import DataLoader

#Take in small batch of images for testing, checking on progress
#eg. maybe 10-20 src-trg pairs
#save output to file every x epochs, we manually check
EVAL_IMG_PATH = 'eval_img'
EVAL_IMG_PREF = 'eval_img_epoch_'
FECNET_PATH = 'saved_models/model_epoch_1.pkl'
eval_freq = 5

def load_models(fecnet_path, options):
    #make sure to load the fecnet from other github branch
    translator = define_T(init_type=options.init_type)
    discriminator = define_D(init_type=options.init_type)
    fecnet = FECNet(pretrained=True)

    #fecnet.load_state_dict(torch.load(fecnet_path)['model_state_dict'])

    return translator, discriminator, fecnet

def eval(model, img_trg, img_src, path):

    eval_out = model(img_trg, img_src)

    for i, img in enumerate(eval_out):
        filename = '_{}.jpg'.format(i)
        trans = transforms.ToPILImage(mode='RGB')
        img = trans(img)
        img.save(path+filename)
        #vutils.save_image(255*img, path + filename)

    print('Saved eval images!')

def plot(x, y1, y2, y3, y4, eval_freq, mode, path):
    plt.clf()
    plt.plot(x, y1, x, y2, x, y3, x, y4)
    plt.xlabel('# of Iterations')
    plt.ylabel('{}'.format(mode))
    plt.title('{}'.format(mode))
    plt.legend(['Embedding loss', 'Adversarial loss', 'Consistency loss', 'Total loss'])
    plt.savefig(os.path.join(path,'{}.png'.format(mode)))
    return True

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--options', help='Set path to options yaml file')
    # args = parser.parse_args()

    #options = load_options_from_yaml(args.options)
    torch.manual_seed(10)

    options = load_options_from_yaml('options.yml')
    T, D, fecnet = load_models(FECNET_PATH, options)

    #? Load from checkpoint
    # checkpoint_file = 'final_trained_FEC.pkl'
    # checkpoint = torch.load(checkpoint_file)
    # fecnet.load_state_dict(checkpoint['model_state_dict'], strict=False)

    #Add data loading stuff here
    Dataloader_val = dataloader(root_source='final_eval/source', root_target='final_eval/target', batch_size=5, shuffle=False)
    Dataloader_train = dataloader(root_source='/home/MirrorMe/project/clean_dataset/train_data', root_target='/home/MirrorMe/project/clean_dataset/train_data',batch_size = 32)
    og_source_img, og_target_img = next(iter(Dataloader_val)) # 10-20 src-trg pairs
    og_source_img = og_source_img.cuda()
    og_target_img = og_target_img.cuda()

    #MODEL OBJECT INSTANTIATE
    model = ReflectionGAN(fecnet, T, D, options)

    epochs = options.epochs

    # Keep track of losses
    embedding_losses = np.array([])
    adversarial_losses = np.array([])
    consistency_losses = np.array([])
    total_losses = np.array([])
    num_iterations = 0
    for epoch in range(epochs):
        print('\n-----------------------------------------------')
        print("Epoch #: {}\n".format(epoch+1))
        for i, (source_img, target_img) in enumerate(Dataloader_train):
            num_iterations += 1
            source_img = source_img.cuda()
            target_img = target_img.cuda()

            embedding_loss, adversarial_loss, consistency_loss, total_loss = model.train_batch(target_img, source_img)

            embedding_losses = np.append(embedding_losses, embedding_loss)
            adversarial_losses = np.append(adversarial_losses, adversarial_loss)
            consistency_losses = np.append(consistency_losses, consistency_loss)
            total_losses = np.append(total_losses, total_loss)

            print("batch num:{}".format(i))
            print("embedding loss: {}".format(embedding_loss))
            print("adversarial_loss: {}".format(adversarial_loss))
            print("consistency loss: {}".format(consistency_loss))

        if (epoch % eval_freq == 0):
            dirname = EVAL_IMG_PREF + str(epoch)

            eval(model, og_target_img, og_source_img, os.path.join(EVAL_IMG_PATH, dirname)) #insert unchanging batch of src, trg images

    #? Plot the losses
    plot_path = './plots/'
    x = [i for i in range(1,num_iterations+1)]
    plot(x,embedding_losses,adversarial_losses,consistency_losses,total_losses,eval_freq,'loss',plot_path)
    print("PLOT SAVED!")

if __name__ == '__main__':
    main()