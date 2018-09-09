# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from assignment1.constants import NUM_TEST, NUM_TRAIN, NUM_DEV, NUM_CLASSES, NUM_VALIDATION
from pathlib import Path
from assignment1.cs231n.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
import time
from assignment1.cs231n.gradient_check import grad_check_sparse


def visualize_examples(X_train):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()




def subsample_and_reshape_data(x_train, y_train, x_test, y_test):
    # Subsample the data for more efficient code execution in this exercise
    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(NUM_TRAIN, NUM_TRAIN + NUM_VALIDATION)
    x_val = x_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(NUM_TRAIN)
    x_train = x_train[mask]
    y_train = y_train[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(NUM_TRAIN, NUM_DEV, replace=False)
    x_dev = x_train[mask]
    y_dev = y_train[mask]

    mask = list(range(NUM_TRAIN))
    x_train = x_train[mask]
    y_train = y_train[mask]


    mask = list(range(NUM_TEST))
    x_test = x_test[mask]
    y_test = x_test[mask]

    # Preprocessing: reshape the image data into rows
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_dev = np.reshape(x_dev, (x_dev.shape[0], -1))

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(x_train, axis=0, dtype=np.uint8)

    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))  # visualize the mean image

    # second: subtract the mean image from train and test data
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image
    x_dev -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    x_dev = np.hstack([x_dev, np.ones((x_dev.shape[0], 1))])

    return x_train, y_train, x_test, y_test, x_dev, y_dev, x_val, y_val


def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #visualize_examples(X_train)
    x_train, y_train, x_test, y_test, x_dev, y_dev, x_val, y_val = subsample_and_reshape_data(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test, x_dev, y_dev, x_val, y_val


def loss_checker(W, x_dev, y_dev):
    # Next implement the function svm_loss_vectorized; for now only compute the loss;
    # we will implement the gradient in a moment.
    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, x_dev, y_dev, 0.000005)
    toc = time.time()
    print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, _ = svm_loss_vectorized(W, x_dev, y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.

    #there is the small diff in loss_naive and loss_vectorized because I include margin 1 when x[i] == y[i]
    # but in loss_naive i do not include this
    print('difference: %f' % (loss_naive - loss_vectorized))



def gradient_checker(W, X_dev, y_dev):


    # do the gradient check once again with regularization turned on
    # you didn't forget the regularization gradient did you?
    loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 0.05)
    f = lambda w: svm_loss_vectorized(w, X_dev, y_dev, 0.05)[0]
    grad_numerical = grad_check_sparse(f, W, grad)
    print(grad_numerical)


if __name__ == '__main__':
    # This is a bit of magic to make matplotlib figures appear inline in the notebook
    # rather than in a new window.
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    x_train, y_train, x_test, y_test, x_dev, y_dev, x_val, y_val = get_data()
    W = np.random.randn(3073, 10) * 0.0001
