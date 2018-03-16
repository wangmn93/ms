import data_mnist as data
import utils
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def getMNISTDatapool(batch_size, keep=None):
    if keep is None:
        imgs, _, num_train_data = data.mnist_load('MNIST_data')
    else:
        imgs, _, num_train_data = data.mnist_load('MNIST_data', keep=keep)
    print "Total number of training data: " + str(num_train_data)
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs}, batch_size)
    return data_pool

def getOnehot(labels, depth):
    onehot_labels = tf.one_hot(indices=labels, depth=depth)
    return onehot_labels

def getToyDataset(mus, cov, numberPerCluster):
    X = np.array([mus[0]])
    for mu in mus:
        x = np.random.multivariate_normal(mu, cov, numberPerCluster)
        # plt.plot(x[:, 0], x[:, 1], 'rx')
        # plt.plot(x2, y2, 'bx')

        X = np.concatenate((X,x))
    # plt.axis('equal')
    # plt.show()
    return X[1:]

def getToyDatapool(batch_size, mus, cov, numberPerCluster):
    X = getToyDataset(mus, cov, numberPerCluster)
    print "Total number of training data: " + str(len(X))
    data_pool = utils.MemoryData({'point':X}, batch_size)
    return data_pool

def saveSampleImgs(imgs, full_path, row, column):
    utils.imwrite(utils.immerge(imgs, row, column),full_path)


