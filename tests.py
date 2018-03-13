#test code snippet here
import data_mnist as data
if 0:
    imgs, labels, _ = data.mnist_load('MNIST_data')#one-hot = False

    for l in range(10):
        print labels[l]