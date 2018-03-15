#test code snippet here
import data_mnist as data
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import my_utils

#extracting subset of mnist
if 0:
    imgs, labels, _ = data.mnist_load('MNIST_data')#one-hot = False

    for l in range(10):
        print labels[l]

    keep = [0,9]
    X,Y = [],[]
    for x,y in zip(imgs,labels):
        if y in keep:
            X.append(x)
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    for l in range(10):
        img = np.reshape(X[l],[28,28])
        plt.imshow(img, cmap='gray')
        print Y[l]
        plt.show()

#convert labels to one-hot
if 0:
    sess = tf.InteractiveSession()
    labels = [1,2,3]
    depth = 10
    onehot_labels = tf.one_hot(indices=labels, depth=depth)
    print sess.run(onehot_labels)

#generate toy data set
if 1:
    mus = [[1,1],[5,5]]
    cov = [[1,0],[0,1]]
    # x,y = np.random.multivariate_normal(mu, sigma, 1000).T
    #
    # mu2 = [5,5]
    # x2, y2 = np.random.multivariate_normal(mu2, sigma, 1000).T
    x = my_utils.getToyDataset(mus,cov,100)
    plt.plot(x[:,0], x[:,1], 'rx')
    # plt.plot(x2, y2, 'bx')
    plt.axis('equal')
    plt.show()
