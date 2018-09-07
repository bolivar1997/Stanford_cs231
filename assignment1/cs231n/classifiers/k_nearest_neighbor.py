import numpy as np
from constants import NUM_CLASSES
from scipy.spatial import distance

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """

        #reshape images to a vector
        X = np.reshape(X, (X.shape[0], -1))

        self.X_train = X
        self.y_train = y


    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)


    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training
        point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            for j in range(num_train):
                test_image = X[i]
                train_image = self.X_train[j]
                dist = np.sum((train_image - test_image) ** 2)

                dists[i, j] = dist

        return dists


    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sum(np.square(self.X_train - X[i, :]), axis=1)
        return dists


    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """

        # our L2 distance should look like (a - b) ^ 2 or a^2 + b^2 - 2ab
        # when we multiply num_test and num_train^t in each cell of result matrix we get ab

        ab = np.dot(X, np.transpose(self.X_train))

        result = ab

        a_squared = np.sum(X ** 2, axis=1)
        b_squared = np.sum(self.X_train ** 2, axis=1)

        result = -2 * result + b_squared + a_squared.reshape(-1, 1)

        return result

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        num_test = dists.shape[0]
        y_pred = []

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            sorted_arrays_elements_indexes = np.argsort(dists[i])
            class_counter = np.zeros(NUM_CLASSES)
            for j in range(k) :
                sorted_element_index = sorted_arrays_elements_indexes[j]

                nearest_class = self.y_train[sorted_element_index, 0]
                class_counter[nearest_class] += 1

            predicted_class = np.argmax(class_counter)
            y_pred.append(predicted_class)

        return y_pred



