
import numpy as np
import pandas as pd
import time
import re
import os

from urllib.request import Request, urlopen
from urllib.error import HTTPError

TRAIN_PATH = 'FEC_dataset/faceexp-comparison-data-train-public.csv'
TEST_PATH = 'FEC_dataset/faceexp-comparison-data-test-public.csv'

TRAIN_OUT = 'dataset/train_data'
TRAIN_TRIPLETS_OUT = 'dataset/train_triplets.csv'

TEST_OUT = 'dataset/test_data'
TEST_TRIPLETS_OUT = 'dataset/test_triplets.csv'

HEADER = ('User-Agent',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36'
    '(KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36')

LABEL_COLS = [17, 19, 21, 23, 25, 27]
URL_COLS = [0, 5, 10]

def load_dataset(path):
    return pd.read_csv(path, header=None, usecols=range(0, 28, 1))

#Utility function to see how many times a value occurs in a subset of a list
def sum_equal(list, indices, val):
    sum = 0

    for i in indices:
        sum += list[i] == val

    return sum

#In the paper, the authors only used triplets where the mode of the labels
#made up at least two thirds (4, since there are 6 annotators) of the total
#number of labels.
def filter_strong_agreement(df):
    #If there are multiple modes, only take the first
    label_modes = df.iloc[:, LABEL_COLS].mode(1).iloc[:, 0]

    #using sum_equal, determine how many people chose the majority option/label
    mode_frequency = [
        sum_equal(df.iloc[i], LABEL_COLS, label_modes.iloc[i])
        for i in range(0, len(df))
    ]

    concat = pd.concat([df, pd.DataFrame({'freq' : mode_frequency})], axis=1)

    filtered = concat[concat['freq'] >= 4].iloc[:, :-1]

    return filtered

def extract_urls(df):
    urls = list(set(df[URL_COLS].to_numpy().ravel()))

    return urls

#Easy way to create a unique name for each file
def url_strip_nonalphanum(url):
    #return re.sub(r'\W+', '', url[:-4])
    return re.sub(r'\W+', '', url[:-4]) + '.jpg'

def download_images(urls, path):
    #Need this later, we drop triplets that reference images that couldnt be
    #downloaded
    bad_urls = []
    counter = 0

    for i in urls:
        try:
            counter += 1
            print(
                'Downloading from '\
              + str(i)\
              + ' {}/{}'.format(counter, len(urls))
            )

            req = Request(url=i)
            req.add_header(*HEADER)
            content = urlopen(req)
            img_data = content.read()

            f_name = url_strip_nonalphanum(i)
            f = open(os.path.join(path, f_name + '.jpg'), 'wb')
            f.write(img_data)
            f.close()

        except HTTPError as e:
            print('Failed to download!')
            bad_urls += [i]
            continue

    return bad_urls

def generate_image_set(inpath, outpath, triplets_outpath):
    trainset = load_dataset(inpath)
    filtered = filter_strong_agreement(trainset)
    urls = extract_urls(filtered)
    bad_urls = download_images(urls, outpath)

    #If we failed to download an image, we want to get rid of any triplets
    #that reference that image
    final = filtered[~(filtered[URL_COLS[0]].isin(bad_urls)\
                     | filtered[URL_COLS[1]].isin(bad_urls)\
                     | filtered[URL_COLS[2]].isin(bad_urls))]
    final.to_csv(triplets_outpath, index=False, header=False)

    return True

def main():
    #generate_image_set(TEST_PATH, TEST_OUT, TEST_TRIPLETS_OUT)
    ds = load_dataset('clean_dataset/train_triplets.csv')[:45000]
    filt = filter_strong_agreement(ds)
    modes = filt.iloc[:, LABEL_COLS].mode(1).iloc[:, 0]
    out = pd.concat([filt[URL_COLS].applymap(url_strip_nonalphanum), modes], axis=1)
    out.to_csv('human_output_2.csv', index=False, header=False)

if __name__ == '__main__':
    main()