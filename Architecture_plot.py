#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

from keras.utils import plot_model

from model import get_unet, dice_coef


# apply a 3x3 convolution with 64 output filters on a 256x256 image:
# get u_net model
model = get_unet()
print("We finish building the model")

plot_model(model, to_file='unet_model.png', show_shapes=True)