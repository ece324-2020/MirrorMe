import numpy as np
import pandas as pd
import cv2
import os

IMAGES_PATH = 'personai_icartoonface_rectrain/icartoonface_rectrain'

#Bounding boxes
BB_INFO_PATH = 'personai_icartoonface_rectrain/icartoonface_rectrain_det.txt'

OUT_PATH = 'clean_dataset/train_data'

def load_bb_info(path):
    return pd.read_csv(path, sep='\t', header=None).to_numpy()

def crop_to_bb(img_bb, inpath, outpath):
    for i in img_bb:
        f_name = i[0].split('/')[-1]
        img = cv2.imread(os.path.join(inpath, i[0]))
        img = img[i[2]:i[4], i[1]:i[3]]
        cv2.imwrite(os.path.join(outpath, f_name), img)

        print('Cropped image {}'.format(f_name))

def main():
    bb_info = load_bb_info(BB_INFO_PATH)
    crop_to_bb(bb_info, IMAGES_PATH, OUT_PATH)

if __name__ == '__main__':
    main()