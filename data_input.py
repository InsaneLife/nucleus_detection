#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

import os
import random
import sys
import warnings
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.utils import Progbar
import scipy
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
from preparation import get_contour, split_overlay_mask_by_contour

# Setting seed for reproducability
seed = 42
random.seed = seed
np.random.seed = seed

# Data Path
data_root = 'E:/project_data/nucleus_detection/'
data_root = '/home/aaron/project_data/nucleus_detection/'
TRAIN_PATH = data_root + '/stage1_train/'
TEST_PATH = data_root + '/stage2_test/'
INPUT_PATH = data_root + '/input/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# Function read train images and mask return as nump array
def read_train_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    if os.path.isfile(INPUT_PATH + "train_img.npy") and os.path.isfile(INPUT_PATH + "train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load(INPUT_PATH + "train_img.npy")
        Y_train = np.load(INPUT_PATH + "train_mask.npy")
        return X_train, Y_train
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        masks, masks_counters = [], []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            masks.append(mask_)
            mask_contour = get_contour(mask_)
            masks_counters.append(mask_contour)
        masks = np.sum(np.array(masks), axis=0)
        masks_counters = np.sum(np.array(masks_counters), axis=0)
        split_masks = split_overlay_mask_by_contour(masks, masks_counters)
        Y_train[n] = np.expand_dims(resize(split_masks, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                                                            preserve_range=True), axis=-1)
        a.update(n)

    np.save(INPUT_PATH + "train_img", X_train)
    np.save(INPUT_PATH + "train_mask", Y_train)
    return X_train, Y_train


# Function to read test images and return as numpy array
def read_test_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()
    if os.path.isfile(INPUT_PATH + "test_img.npy") and os.path.isfile(INPUT_PATH + "test_size.npy"):
        print("Test file loaded from memory")
        X_test = np.load(INPUT_PATH + "test_img.npy")
        sizes_test = np.load(INPUT_PATH + "test_size.npy")
        return X_test, sizes_test
    b = Progbar(len(test_ids))
    for n, id_ in enumerate(test_ids):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        b.update(n)
    np.save(INPUT_PATH + "test_img", X_test)
    np.save(INPUT_PATH + "test_size", sizes_test)
    return X_test, sizes_test


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
def mask_to_rle(preds_test_upsampled):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids, rles


if __name__ == '__main__':
    x, y = read_train_data()
    x, y = read_test_data()
