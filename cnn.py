# Author:   Jay Huang <askjayhuang@gmail.com>
# Created:  2017-11-19T23:41:18.706Z

"""A module for running a convolution neural network for image recognition on
   the CIFAR-10 dataset. The consists of 60,000 32x32 color images containing
   one of 10 object classes, with 6000 images per class. The training set
   contains 50,000 images while the test set contains 10,000 images.
"""

################################################################################
# Imports
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools

from sklearn.metrics import classification_report, classification, accuracy_score

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

################################################################################
# Global Constants
################################################################################

LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
IMG_DIM = 32
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 2

################################################################################
# Functions
################################################################################


def plot_random_images():
    fig = plt.figure(figsize=(5, 14))
    for i in range(NUM_CLASSES):
        idx = np.where(y_train == i)[0]
        features_idx = X_train[idx, ::]

        img_num1 = np.random.randint(features_idx.shape[0])
        im1 = np.transpose(features_idx[img_num1, ::], (1, 2, 0))
        img_num2 = np.random.randint(features_idx.shape[0])
        im2 = np.transpose(features_idx[img_num2, ::], (1, 2, 0))
        img_num3 = np.random.randint(features_idx.shape[0])
        im3 = np.transpose(features_idx[img_num3, ::], (1, 2, 0))
        img_num4 = np.random.randint(features_idx.shape[0])
        im4 = np.transpose(features_idx[img_num4, ::], (1, 2, 0))

        ax1 = fig.add_subplot(10, 4, (i * 4) + 1, xticks=[], yticks=[])
        ax2 = fig.add_subplot(10, 4, (i * 4) + 2, xticks=[], yticks=[])
        ax3 = fig.add_subplot(10, 4, (i * 4) + 3, xticks=[], yticks=[])
        ax4 = fig.add_subplot(10, 4, (i * 4) + 4, xticks=[], yticks=[])
        ax1.set_title(LABEL_NAMES[i])
        ax2.set_title(LABEL_NAMES[i])
        ax3.set_title(LABEL_NAMES[i])
        ax4.set_title(LABEL_NAMES[i])
        ax1.imshow(im1)
        ax2.imshow(im2)
        ax3.imshow(im3)
        ax4.imshow(im4)
    plt.show()


def plot_loss_acc(mh):
    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(mh.history['loss'], 'b-', mh.history['val_loss'], 'r-')
    ax1.legend(['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax1.set_title('Loss')

    ax2.plot(mh.history['acc'], 'b-', mh.history['val_acc'], 'r-')
    ax2.legend(['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax2.set_title('Accuracy')
    plt.show()

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    plot_random_images()

    # y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    # y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    #
    # model = Sequential()
    # model.add(Conv2D(IMG_DIM, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(Conv2D(IMG_DIM, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(IMG_DIM * 2, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(IMG_DIM * 2, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(IMG_DIM * 8))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(NUM_CLASSES))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy']
    #               )
    #
    # model.summary()
    #
    # early_stopping_monitor = EarlyStopping(patience=PATIENCE)
    #
    # mh = model.fit(X_train, y_train, validation_data=(X_test, y_test),
    #                batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping_monitor], shuffle=True)
