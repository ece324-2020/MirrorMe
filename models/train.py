import time
import os
import argparse
import torch
import torchvision.utils as vutils

# from .reflection_gan_options import load_options_from_yaml
from models. import *
from .reflection_gan_options import load_options_from_yaml
from reflection_gan_model import ReflectionGAN
from GAN_DataLoader import dataloader

#Take in small batch of images for testing, checking on progress
#eg. maybe 10-20 src-trg pairs
#save output to file every x epochs, we manually check
EVAL_IMG_PATH = 'eval_img'
EVAL_IMG_PREF = 'eval_img_epoch_'
eval_freq = 5

def eval(model, img_trg, img_src, path):
    eval_out = model(img_trg, img_src)

    for i, img in enumerate(eval_out):
        filename = '{}.jpg'.format(i)
        vutils.save_image(img, os.path.join(path, filename))

    print('Saved eval images!')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--options', help='Set path to options yaml file')
    # args = parser.parse_args()

    # options = load_options_from_yaml(args.options)

    #Add data loading stuff here
    dataloader = dataloader()
    og_source_img, og_target_img = next(iter(dataloader)) # 10-20 src-trg pairs

    #MODEL OBJECT INSTANTIATE
    model = ReflectionGAN(aioegpwihgpoawe)

    epochs = options.epochs

    # Keep track of losses
    embedding_losses = np.array([])
    adversarial_losses = np.array([])
    consistency_losses = np.array([])
    total_losses = np.array([])

    for epoch in range(epochs):
        for i, (source_img, target_img) in enumerate(dataloader):
            embedding_loss, adversarial_loss, consistency_loss, total_loss = model.train_batch(target_img, source_img)

            embedding_losses = np.append(embedding_loss)
            adversarial_losses = np.append(adversarial_loss)
            consistency_losses = np.append(consistency_loss)
            total_losses = np.append(total_loss)

        if (epoch % eval_freq == 0):
            dirname = EVAL_IMG_PREF + str(epoch)
            eval(model, og_target_img, og_source_img, os.path.join(EVAL_IMG_PATH, dirname)) #insert unchanging batch of src, trg images

        