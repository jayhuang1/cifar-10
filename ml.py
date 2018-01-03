# Author:   Jay Huang
# E-mail: askjayhuang at gmail dot com
# GitHub: https://github.com/jayh1285
# Created:  2018-01-01T20:20:13.766Z

"""A module for running supervised learning algorithms for image recognition on
   the CIFAR-10 dataset. The consists of 60,000 32x32 color images containing
   one of 10 object classes, with 6000 images per class. The training set
   contains 50,000 images while the test set contains 10,000 images.
"""

################################################################################
# Imports
################################################################################

import pandas as pd
import pickle
import time
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.metrics import classification_report

from keras.datasets import cifar10

################################################################################
# Global Constants
################################################################################

TRAIN_BATCHES = ['data/data_batch_1', 'data/data_batch_2',
                 'data/data_batch_3', 'data/data_batch_4', 'data/data_batch_5']
TEST_BATCH = 'data/test_batch'
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

################################################################################
# Functions
################################################################################


def read_data():
    """Prepare batch data into machine learning training and test sets."""
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Open the 5 training batches
    for batch in TRAIN_BATCHES:
        with open(batch, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')

            # Append the image data to X_train
            data = dict[b'data']
            for image in data:
                X_train.append(image)

            # Append the label data to y_train
            labels = dict[b'labels']
            for label in labels:
                y_train.append(label)

    # Open the test batch
    with open(TEST_BATCH, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

        # Set test data
        X_test = dict[b'data']
        y_test = dict[b'labels']

    return X_train, y_train, X_test, y_test


def fit_model():
    # Train model
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start

    print("{:25} fit in: {:0.2f} seconds".format(model.__class__.__name__, duration))

    # Test model
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    nsamples, nx, ny, nz = X_train.shape
    X_train = X_train.reshape((nsamples, nx * ny * nz))
    nsamples, nx, ny, nz = X_test.shape
    X_test = X_test.reshape((nsamples, nx * ny * nz))
    # X_train, y_train, X_test, y_test = read_data()

    # Machine learning models to use
    models = (
        # LogisticRegression(),
        # SVC(),
        RandomForestClassifier(),
        # Perceptron(),
        # KNeighborsClassifier(),
        # KNeighborsClassifier(n_neighbors=15),
        # KNeighborsClassifier(n_neighbors=2),
        # GaussianNB(),
        # MultinomialNB()
    )

    # Iterate through each model
    for model in models:
        fit_model()
