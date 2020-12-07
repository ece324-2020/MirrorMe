import numpy as np
import pandas as pd
import re
import cv2
import os

from collect_images import load_dataset, url_strip_nonalphanum

TRAIN_DIR = 'dataset/train_data'
TRAIN_TRIPLETS = 'dataset/train_triplets.csv'

TEST_DIR = 'dataset/test_data'
TEST_TRIPLETS = 'dataset/test_triplets.csv'

CLEAN_TRAIN_OUT = 'clean_dataset/train_data'
CLEAN_TRAIN_TRIPLETS = 'clean_dataset/train_triplets.csv'

CLEAN_TEST_OUT = 'clean_dataset/test_data'
CLEAN_TEST_TRIPLETS = 'clean_dataset/test_triplets.csv'

URL_COLS = [0, 5, 10]

BB = [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]

#Generate dataframe connecting the downloaded images from collect_images to
#their respective bounding box
#Use the same naming convention based on url
#Assume that an image will only have one unique set of bb coordinates
#associated with it
def connect_img_to_bb(df):
    img_bb = []
    
    for i in range(len(URL_COLS)):
        img_bb += [df[[URL_COLS[i], *BB[i]]]]

    img_bb = np.concatenate((
        img_bb[0].to_numpy(),
        img_bb[1].to_numpy(),
        img_bb[2].to_numpy()
    ))

    urls, indices = np.unique(img_bb[:, 0], return_index=True)
    img_bb = img_bb[indices]

    return img_bb

def crop_to_bb(img_bb, inpath, outpath):
    for i in img_bb:
        f_name = url_strip_nonalphanum(i[0]) + '.jpg'
        img = cv2.imread(os.path.join(inpath, f_name))
        shape = img.shape
        img = img[int(i[3] * shape[0]):int(i[4] * shape[0]),
                  int(i[1] * shape[1]):int(i[2] * shape[1])]
        cv2.imwrite(os.path.join(outpath, f_name), img)

        print('Cropped image {}'.format(f_name))

    return True

#Some of the collected images are not referenced by the filtered triplet set
#We clean the image set by taking and cropping only the images used
def generate_clean_set(inpath, triplets_inpath, outpath, triplets_outpath):
    triplets = load_dataset(triplets_inpath)
    img_bb = connect_img_to_bb(triplets)

    crop_to_bb(img_bb, inpath, outpath)

def main():
    generate_clean_set(
        TRAIN_DIR,
        TRAIN_TRIPLETS,
        CLEAN_TRAIN_OUT,
        CLEAN_TRAIN_TRIPLETS
    )


if __name__ == '__main__':
    main()