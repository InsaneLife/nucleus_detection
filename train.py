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
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

epochs = 50
model_name = 'model-0416-bn.h5'
# best_model_name = 'model-dsbowl2018-0416-best.h5'
# get train_data
train_img, train_mask = read_train_data()

# get test_data
test_img, test_img_sizes = read_test_data()

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\n Training...")
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# early_stopper = EarlyStopping(patience=5, verbose=1)
# check_pointer = ModelCheckpoint(best_model_name, verbose=1, save_best_only=True)
u_net.fit(train_img, train_mask, batch_size=16, epochs=epochs, callbacks=[tb])


print("\n Saving")
u_net.save(model_name)

print("\n load model")
u_net = load_model(model_name, custom_objects={'dice_coef': dice_coef})

print("\n Predicting and Saving predict")
# Predict on test data
test_mask = u_net.predict(test_img, verbose=1)
np.save("test_img_bn_pred", test_mask)
