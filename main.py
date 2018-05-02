#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

from data_input import read_train_data, read_test_data, prob_to_rles, mask_to_rle, resize, np
from model import get_unet, dice_coef
import pandas as pd
from post_process import post_processing
from skimage.io import imshow
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import TensorBoard
epochs = 50
model_name = 'model-0416-test.h5'
# get train_data
train_img, train_mask = read_train_data()

# get test_data
test_img, test_img_sizes = read_test_data()

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\n Training...")
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
u_net.fit(train_img, train_mask, batch_size=16, epochs=epochs, callbacks=[tb])

print("\n Saving")
u_net.save(model_name)

print("\n load model")
u_net = load_model(model_name, custom_objects={'dice_coef': dice_coef})

print("\n Predicting and Saving predict")
# Predict on test data
test_mask = u_net.predict(test_img, verbose=1)
np.save("test_img_pred", test_mask)

test_mask = np.load("test_img_pred.npy")
# get test_data
# test_img, test_img_sizes = read_test_data()

# post processing
post_test_mask = post_processing(test_mask)

post_test_mask = np.expand_dims(post_test_mask, axis=-1)
# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(post_test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(post_test_mask[i]),
                                      (test_img_sizes[i][0], test_img_sizes[i][1]),
                                      mode='constant', preserve_range=True))
print('Done!')

test_ids, rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018_preprocess_post_process_0416_1.csv', index=False)

print("Data saved")