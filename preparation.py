#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

import glob
import os

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from imageio import imwrite
from skimage.transform import resize
from sklearn.cluster import KMeans
from tqdm import tqdm


def train_valid_split(meta, validation_size, valid_category_ids=None):
    meta_train = meta[meta['is_train'] == 1]
    meta_train_split, meta_valid_split = split_on_column(meta_train,
                                                         column='vgg_features_clusters',
                                                         test_size=validation_size,
                                                         random_state=1234,
                                                         valid_category_ids=valid_category_ids
                                                         )
    return meta_train_split, meta_valid_split


def split_on_column(meta, column, test_size, random_state=1, valid_category_ids=None):
    if valid_category_ids is None:
        categories = meta[column].unique()
        np.random.seed(random_state)
        valid_category_ids = np.random.choice(categories,
                                              int(test_size * len(categories)))
    valid = meta[meta[column].isin(valid_category_ids)].sample(frac=1, random_state=random_state)
    train = meta[~(meta[column].isin(valid_category_ids))].sample(frac=1, random_state=random_state)
    return train, valid


def overlay_centers(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_center(image))
        overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)
    return img_contour


def get_center(img):
    img_center = np.zeros_like(img).astype(np.uint8)
    y, x = ndi.measurements.center_of_mass(img)
    cv2.circle(img_center, (int(x), int(y)), 4, (255, 255, 255), -1)
    return img_center


def split_overlay_mask_by_contour(masks, masks_counters):
    # retain overlay contour
    masks_counters[masks_counters <= 255] = 0
    splited_masks = np.sum([masks_counters, masks], axis=0)
    # drop overlay contour
    splited_masks[splited_masks > 255] = 0
    # to bool
    splited_masks[splited_masks > 0] = 1
    return splited_masks
