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
if 0:
    one_hot_labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    p = [1/3.]*3

    def sample_from_categorical(size, k=3):
        return [one_hot_labels[index] for index in np.random.choice(range(k), size=size, p=p)]

    print sample_from_categorical(10)

#test einsum
if 0:
    k = 4
    z_dim = 2
    batch_size = 5
    size =batch_size
    log_sigma_sq = -2
    with tf.variable_scope("gmm", reuse=False):
        # mus = tf.get_variable("mus", [k, z_dim], initializer=tf.constant_initializer(0))
        # log_sigma_sqs = tf.get_variable("log_sigma_sqs", [k, z_dim], initializer=tf.constant_initializer(0.001))
        mus =  tf.constant([[0.5, 0.5],[-0.5, 0.5],[-0.5,-0.5],[0.5, -0.5]],shape=[4,2],dtype=tf.float32)
        log_sigma_sqs = tf.constant([[log_sigma_sq, log_sigma_sq]]*4,shape=[4,2], dtype=tf.float32)


    def get_gmm_sample(size):
    # shape = (size,)+tf.shape(log_sigma_sqs)
    # print shape
        eps = tf.random_normal(shape=(size, k, z_dim),mean=0, stddev=1, dtype=tf.float32)
        zs = tf.tile(tf.expand_dims(mus, 0),[size,1,1]) \
         +  tf.tile(tf.expand_dims(tf.sqrt(tf.exp(log_sigma_sqs)),0),[size,1,1])*eps
        return zs


    softmaxs = tf.constant([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]], shape=[batch_size, k],dtype=tf.float32)
    zs = get_gmm_sample(batch_size) # batch_size x 4x 2
    # temp = tf.expand_dims(tf.sqrt(tf.exp(softmaxs)), -1) # batch_size x 4 x 1
    r = tf.einsum('ib,ibk->ik', softmaxs, zs)
    sess = tf.Session()
    # t = tf.tile(tf.expand_dims(mus, 0),[10,1,1])
    print zs.shape
    print softmaxs.shape
    print r.shape
    # r = tf.reshape(r,[batch_size,z_dim])
    # print sess.run([t])

    print sess.run([mus,r])
    sess.close()

if 0:
    batch_size = 10
    data_pool_1 = my_utils.getMNISTDatapool(batch_size,keep=[0])  # range -1 ~ 1
    data_pool_2 = my_utils.getMNISTDatapool(batch_size, keep=[1])
    imgs_1 = data_pool_1.batch('img')
    imgs_2 = data_pool_2.batch('img')


    for i,j in zip(imgs_1, imgs_2):
        fig = plt.figure()

        fig.add_subplot(1, 2, 0)
        img = np.reshape(i, [28, 28])
        plt.imshow(img, cmap='gray')

        img_2 = np.reshape(j, [28, 28])
        fig.add_subplot(1, 2, 1)
        plt.imshow(img_2, cmap='gray')

        plt.show()

if 1:
    p = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                           shape=[5, 4], dtype=tf.float32)
    def cond_entropy(y):
        # y1 = -y * F.log(y)
        # y2 = F.sum(y1) / batchsize
        # return y2
        y1 = -y * tf.log(y)
        print y1.shape
        y2 = tf.reduce_mean(y1, axis=0)
        print y2.shape
        return y2

    cond_entropy(p)
