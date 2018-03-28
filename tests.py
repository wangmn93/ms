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
    labels = tf.scalar_mul(2,tf.ones([3], tf.int32))
    depth = 3
    onehot_labels = tf.one_hot(indices=labels, depth=depth)
    print sess.run(onehot_labels)

#generate toy data set
if 0:
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

#toy data_pool
if 0:
    mus = [[1, 1]]
    cov = [[1, 0], [0, 1]]
    data_pool = my_utils.getToyDatapool(10,mus, cov, 1000)
    for _ in range(1):
        x = data_pool.batch('point')
        plt.plot(x[:, 0], x[:, 1], 'rx')
        # plt.plot(x2, y2, 'bx')
        plt.axis('equal')
        plt.show()

if 0:
    var = raw_input("Continue training?")
    print var
    if var.lower() == 'y':
        print 'y'
    else:
        print 'n'

#custom init
if 0:
    sess = tf.Session()
    with tf.variable_scope("gmm", reuse=False):
        mu_1 = tf.get_variable("mean1", [2], initializer=tf.constant_initializer(0))
        mu_2 = tf.get_variable("mean2", [2], initializer=tf.constant_initializer(1))
        log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [2], initializer=tf.constant_initializer(3))
        log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [2], initializer=tf.constant_initializer(4))
    init_gmm = tf.initialize_variables([mu_1, mu_2, log_sigma_sq1, log_sigma_sq2])
    sess.run(init_gmm)  # or `assign_op.op.run()`
    print(sess.run([mu_1, mu_2, log_sigma_sq1, log_sigma_sq2]))

#sample from categorical dist
if 1:
    one_hot_labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    p = [1/3.]*3

    def sample_from_categorical(size, k=3):
        return [one_hot_labels[index] for index in np.random.choice(range(k), size=size, p=p)]

    print sample_from_categorical(10)