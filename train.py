import time
import argparse

from .reflection_gan_options import load_options_from_yaml
from reflection_gan_model import ReflectionGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', help='Set path to options yaml file')
    args = parser.parse_args()

    options = load_options_from_yaml(args.options)

    #Add data loading stuff here

    epochs = options.epochs

    for epoch in range(epochs):