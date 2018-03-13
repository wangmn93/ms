#test code snippet here
import data_mnist as data
import numpy as np
from matplotlib import pyplot as plt

#test extracting subset of mnist
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

