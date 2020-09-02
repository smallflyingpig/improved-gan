from model import get_inception_score
import os, tqdm
from PIL import Image
import numpy as np 
import scipy
import argparse

def get_parser():
    parser = argparse.ArgumentParser("inception score for coco dataset")
    parser.add_argument("--folder", type=str, default="/media/ubuntu/Elements1/dataset/COCO2014/val2014_j2k_QF50", help="")
    args = parser.parse_args()
    return args 

def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='lanczos')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    # img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return img 


def load_data(fullpath):
    print("load data from: {}".format(fullpath))
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in tqdm.tqdm(files):
            if os.path.splitext(name)[-1] in ['.jpg', '.jpeg', '.png', '.j2k']:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img_fp = Image.open(filename)
                    img = np.array(img_fp)
                    img = preprocess(img)
                    images.append(img)
                    img_fp.close()
    print('images', len(images), images[0].shape)
    return images


if __name__=="__main__":
    args = get_parser()
    images = load_data(args.folder)
    mean, std = get_inception_score(images)
    print("IS mean: {:5.2f}, std: {:5.2f}".format(mean, std))