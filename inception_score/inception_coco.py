from model import get_inception_features
import os, tqdm
from PIL import Image
import numpy as np 
import scipy
import argparse
import logging

def get_parser():
    parser = argparse.ArgumentParser("inception score for coco dataset")
    parser.add_argument("--folder", type=str, default="/media/ubuntu/Elements1/dataset/COCO2014/val2014_j2k_QF50", help="")
    parser.add_argument("--log_file", type=str, default="./inception_coco.log", help="")
    parser.add_argument("--max_data_len", type=int, default=2048, help="")
    parser.add_argument("--splits", type=int, default=10, help="")
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



def load_data(fullpath, max_data_len=2048):
    print("load data from: {}".format(fullpath))
    images = []
    total_data_len = 0
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
            if len(images)>=max_data_len:
                total_data_len += len(images)
                yield images
                images = []
    print('images number: ', total_data_len)
    return images

def get_score(preds, splits=10):
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def main(args):
    logger = get_logger(args.log_file)
    preds = []
    for images in load_data(args.folder, max_data_len=args.max_data_len):
        if len(images)>0:
            preds_this = get_inception_features(images)
            preds.append(preds_this)
    preds = np.concatenate(preds)
    score, std = get_score(preds, splits=args.splits)
    logger.info("for folder: {}\nIS: ({:6.3f}, {:6.3f})".format(args.folder, score, std))


def get_logger(log_file):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    return rootLogger

if __name__=="__main__":
    args = get_parser()
    main(args)