import keras
import cv2
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
import os
import pickle
import numpy as np


def build_autoencoder():
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Flatten()(x)
    x = Dense(3072)(x)
    x = BatchNormalization()(x)
    encoded = Activation('sigmoid')(x)
    x = Dense(3072)(encoded)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Reshape((32, 32, 3))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    model = Model(input_img, decoded)
    print(model.summary())
    model.compile(optimizer='adam', loss='mean_squared_error')


def resize_image(img_path):
    image=cv2.imread(img_path)
    cv2.imshow('real image',image)
    cv2.waitKey(0)
    image=cv2.resize(image,(32,32),interpolation = cv2.INTER_NEAREST)
    cv2.imshow('scaled image', image)
    cv2.waitKey(0)
    return image


if __name__=='__main__':
    build_autoencoder()
    resize_image('frame0.jpg')
