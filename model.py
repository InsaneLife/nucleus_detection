#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

smooth = 1.
padding_type_contract = 'same'
padding_type_expand = 'same'

# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)
    s = BatchNormalization()(s)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(s)
    c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_contract)(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding_type_expand)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding_type_expand)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(u7)
    c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding_type_expand)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(u8)
    c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding=padding_type_expand)(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding_type_expand)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model
