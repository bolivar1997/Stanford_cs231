# -*- coding: utf-8 -*-

import numpy as np
import cv2
from cs231n.classifiers import KNearestNeighbor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from pathlib import Path
from constants import NUM_TEST, NUM_TRAIN


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


def subsample_and_reshape_data(X_train, y_train, X_test, y_test):
    # Subsample the data for more efficient code execution in this exercise
    mask = list(range(NUM_TRAIN))
    X_train = X_train[mask]
    y_train = y_train[mask]


    mask = list(range(NUM_TEST))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    return X_train, y_train, X_test, y_test


def get_distance(classifier, x_test):
    dists = classifier.compute_distances_no_loops(x_test)
    return dists


def use_classifier(x_train, y_train ,x_test, y_test, k):
    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)
    dists = get_distance(classifier, x_test)

    y_pred = classifier.predict_labels(dists, k)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def get_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #visualize_examples(X_train)

    x_train, y_train, x_test, y_test = subsample_and_reshape_data(X_train, y_train, X_test, y_test)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def cross_validation(x_train, y_train):
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]


    x_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.

    my_file = Path("temp/k_to_accuracies.npy")
    if not my_file.is_file():
        k_to_accuracies = {k: [] for k in k_choices}

        for current_k in k_choices:
            for current_folds in range(num_folds):

                print(current_k, current_folds)
                current_train_x =  np.concatenate([x_train_folds[i] for i in range(num_folds) if i != current_folds])
                current_train_y = np.concatenate([y_train_folds[i] for i in range(num_folds) if i != current_folds])

                current_test_x = x_train_folds[current_folds]
                current_test_y = y_train_folds[current_folds]

                accuracy = use_classifier(current_train_x, current_train_y, current_test_x, current_test_y, current_k)
                k_to_accuracies[current_k].append(accuracy)

        np.save('temp/k_to_accuracies.npy', k_to_accuracies)

    else:
        k_to_accuracies = np.load('temp/k_to_accuracies.npy').item()

    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    best_k = k_choices[np.argmax(accuracies_mean)]
    return best_k


if __name__ == '__main__':
    # This is a bit of magic to make matplotlib figures appear inline in the notebook
    # rather than in a new window.
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    x_train, y_train, x_test, y_test = get_data()

    best_k = cross_validation(x_train, y_train)

    print('Best accuracy: ', use_classifier(x_train, y_train, x_test, y_test, best_k))




