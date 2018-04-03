from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
elu = tf.nn.elu
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
lrelu_2 = partial(ops.leak_relu, leak=0.1)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None,)
ln = slim.layer_norm


def generator(z, dim=64, reuse=True, training=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        return img

def generator_m(z, heads=10, dim=64, reuse=True, training=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img_sets = []
        for _ in range(heads):
            img_sets.append(tf.sigmoid(dconv(y, 1, 5, 2)))

        return img_sets

def mad_generator2(z, dim=64, reuse=True, training=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        img2 = tf.tanh(dconv(y, 1, 5, 2))
        img3 = tf.tanh(dconv(y, 1, 5, 2))
        return img, img2, img3

def cnn_classifier(x, name="classifier", reuse=True, keep_prob=1.):
    conv_relu = partial(conv, normalizer_fn=None, activation_fn=relu, biases_initializer=None)
    max_pool = partial(tf.layers.max_pooling2d, pool_size=[2, 2], strides=2)
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu, biases_initializer=None)
    # fc(y, out_dim)
    with tf.variable_scope(name, reuse=reuse):
        y = conv_relu(x,32,5,2)
        y = max_pool(y)
        y = conv_relu(y, 64, 5, 2)
        y = max_pool(y)
        y = fc_relu(y,1024)
        y = tf.nn.dropout(y, keep_prob)
        y = tf.nn.softmax(fc(y,10))
        return y

def cnn_classifier_2(x,keep_prob, out_dim=10,name="classifier", reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=1-keep_prob)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=out_dim)
        y = tf.nn.softmax(logits)
        return y,logits

#mimic
def generator2(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = ops.linear(z,1024,'gl1')
        y = tf.nn.relu(tf.layers.batch_normalization(y, training=training,momentum=0.9,epsilon=1e-5,scale=True))
        y = ops.linear(y, 7 * 7 * dim * 2,'gl2')
        y = tf.nn.relu(tf.layers.batch_normalization(y, training=training,momentum=0.9,epsilon=1e-5,scale=True))
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        return img

#DCGAN generator
def generator3(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = ops.linear(z,2 * 2 * dim * 8,'gl1')#with bias
        y = tf.reshape(y, [-1, 2, 2, dim * 8])
        y = tf.nn.relu(tf.layers.batch_normalization(y, training=training,momentum=0.9,epsilon=1e-5,scale=True))
        y = dconv_bn_relu(y, dim * 4, 5, 2)#without bias
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))

        return img

def discriminator(img, dim=64, reuse=True, training=True, name= 'discriminator'):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit

def discriminator_m(img, out_dim=10, dim=64, reuse=True, training=True, name= 'discriminator', stddev=0.05):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = img
        # y = img + tf.random_normal(shape=tf.shape(img), mean=0, stddev=stddev, dtype=tf.float32)
        y = lrelu(conv(y, 1, 5, 2))
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = conv_bn_lrelu(y, dim, 5, 2)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 1024)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        logits = fc(y, out_dim)
        softmax = tf.nn.softmax(logits)
        return softmax, logits

def discriminator_wgan_gp(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
    fc_ln_lrelu = partial(fc, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_ln_lrelu(y, dim, 5, 2)
        y = fc_ln_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit

#weight sharing arch



def g_shared_part(y, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('g_shared_part', reuse=reuse):
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        return img

def shared_generator(z, reuse=True, training = True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        return y

def d_shared_part(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('d_shared_part', reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        return y

def shared_classifier(y, reuse=True):
    with tf.variable_scope('shared_classifier', reuse=reuse):
        return fc(y, 1)

def shared_discriminator(y, reuse=True):
    with tf.variable_scope('shared_discriminator', reuse=reuse):
        return fc(y, 1)


#toy GAN
def toy_generator(z, reuse=True, name = "generator"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 1024)
        y = fc_relu(y, 1024)
        y = fc_relu(y, 512)
        return fc(y, 2)

def toy_discriminator(x, reuse=True, name = "discriminator"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 128)
        # y = fc_relu(y, 128)
        return fc(y, 1)

def toy_transform(z, reuse=True, name = "transform"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 128)
        y = fc_relu(y, 128)
        return fc(y, 256)

def toy_shared_generator(z, reuse=True, name = "generator"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 128)
        y = fc_relu(y, 128)
        # y = fc_relu(y, )
        return fc(y, 2),fc(y, 2)

def toy_shared_generator2(z, reuse=True, name = "generator"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 128)
        y = fc_relu(y, 128)
        # y = fc_relu(y, )
        return fc(y, 2)

#selective sampling
def ss_generator(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = tf.tanh(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y

def ss_generator_m(z, heads=10,reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 1024)
        img_sets = []
        for _ in range(heads):
            out = tf.sigmoid(fc(y, 784))
            out = tf.reshape(out, [-1, 28, 28, 1])
            img_sets.append(out)
        return img_sets

def ss_generator_2(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = tf.sigmoid(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y

def ss_generator_2_2heads(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y1 = tf.sigmoid(fc(y, 784))
        y1 = tf.reshape(y1, [-1, 28, 28, 1])
        y2 = tf.sigmoid(fc(y, 784))
        y2 = tf.reshape(y2, [-1, 28, 28, 1])
        return y1,y2

def cat_generator(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_lrelu(z, 500)
        y = fc_bn_lrelu(y, 500)
        y = fc_bn_lrelu(y, 1000)
        y = tf.sigmoid(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y

# def cat_conv_generator(z, reuse=True, name = "generator", training = True):
#     bn = partial(batch_norm, is_training=training)
#     dconv_bn_relu = partial(dconv, normalizer_fn=None, activation_fn=None)
#     fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
#
#
#     fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu_2)
#     # fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
#     with tf.variable_scope(name, reuse=reuse):
#         y = fc_lrelu(z, 8 * 8 * 96)
#         tf.layers.conv2d_transpose(
#             y,
#             filters,
#             kernel_size=[2,2])
#         y = fc_bn_lrelu(y, 500)
#         y = fc_bn_lrelu(y, 1000)
#         y = tf.sigmoid(fc(y, 784))
#         y = tf.reshape(y, [-1, 28, 28, 1])
#         return y

def cat_generator_3heads(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_lrelu(z, 500)
        y = fc_bn_lrelu(y, 500)
        y = fc_bn_lrelu_2(y, 1000)
        y_1 = tf.sigmoid(fc(y, 784))
        y_2 = tf.sigmoid(fc(y, 784))
        y_3 = tf.sigmoid(fc(y, 784))
        y_1 = tf.reshape(y_1, [-1, 28, 28, 1])
        y_2 = tf.reshape(y_2, [-1, 28, 28, 1])
        y_3 = tf.reshape(y_3, [-1, 28, 28, 1])
        return [y_1, y_2, y_3]

def cat_conv_discriminator(x, out_dim=10, reuse=True, name = "discriminator", stddev=0.05):
    with tf.variable_scope(name, reuse=reuse):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=lrelu_2)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=lrelu_2)

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=lrelu_2)

        #and Pooling Layer  #2
        pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

        # Convolutional Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=lrelu_2)

        # Convolutional Layer #5
        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=10,
            kernel_size=[1, 1],
            padding="same",
            activation=lrelu_2)

        # Dense Layer
        dense = lrelu_2(fc(conv5,128))
        # conv5_flat = tf.reshape(conv5, [-1, 7 * 7 * 64])
        # dense = tf.layers.dense(inputs=conv5_flat, units=128, activation=lrelu_2)

        # Logits Layer
        logits = tf.layers.dense(inputs=dense, units=out_dim)
        y = tf.nn.softmax(logits)
        return y, logits

def cat_generator_m(z, heads=10,reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_lrelu(z, 500)
        y = fc_bn_lrelu(y, 500)
        y = fc_bn_lrelu_2(y, 1000)
        out_put_sets = []
        for _ in range(heads):
            y_1 = tf.sigmoid(fc(y, 784))
            y_1 = tf.reshape(y_1, [-1, 28, 28, 1])
            out_put_sets.append(y_1)
        return out_put_sets

def cat_discriminator(z, out_dim=10, reuse=True, name = "discriminator", training = True, stddev=0.3):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    # fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = z + tf.random_normal(shape=tf.shape(z),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 1000)
        y = y + tf.random_normal(shape=tf.shape(y),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 500)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        logits = fc(y, out_dim)
        y_out = tf.nn.softmax(logits)
        return y_out,logits

def simple_mad_generator(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 256)
        y = fc_bn_relu(y, 512)
        y_1 = tf.tanh(fc(y, 2))
        y_2 = tf.tanh(fc(y, 2))
        y_3 = tf.tanh(fc(y, 2))
        return y_1, y_2, y_3

def simple_mad_discriminator(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_lrelu(x, 1024)
        y = fc_lrelu(y, 1024)
        return fc(y, 4)

def mad_generator(z, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 256)
        y = fc_bn_relu(y, 512)
        y = fc_bn_relu(y, 1024)

        y_1 = tf.tanh(fc(y, 784))
        y_2 = tf.tanh(fc(y, 784))
        y_3 = tf.tanh(fc(y, 784))

        y_1 = tf.reshape(y_1, [-1, 28, 28, 1])
        y_2 = tf.reshape(y_2, [-1, 28, 28, 1])
        y_3 = tf.reshape(y_3, [-1, 28, 28, 1])
        return y_1, y_2, y_3

def ss_discriminator(x, label = None, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y =  tf.reshape(x, [-1, 784])
        if label is not None:
            y = tf.concat([y, label], 1)
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

def discriminator_for_latent(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_lrelu(x, 512)
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

def transform_(z, reuse=True, name = "generator"):
    # bn = partial(batch_norm, is_training=training)
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 256)
        y = fc_relu(y, 100)
        return y

#### need to enlarge capacity??
def c_generator(z, label, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.concat([z, label], 1)
        y = fc_bn_relu(y, 1024)
        y = fc_bn_relu(y, 1024)
        y = tf.tanh(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y


# def c_discriminator(x, label,reuse=True, name = "discriminator"):
#     fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
#     with tf.variable_scope(name, reuse=reuse):
#         y =  tf.reshape(x, [-1, 784])
#         y = fc_lrelu(y, 1024)
#         return fc(y, 1)

def multi_c_discriminator(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y =  tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        return fc(y, 3)

#enlarge capacity of discriminator
def multi_c_discriminator2(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        y = fc_lrelu(y, 1024)
        return fc(y, 3)

def multi_c_discriminator3(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        y = fc_lrelu(y, 1024)
        return fc(y, 4)

def s_generator(z, reuse=True, name = "generator", part1 = "p1", part2 = "p2", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = tf.tanh(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y

#vae
def encoder(x, z_dim = 2, reuse=True, name = "encoder"):
    fc_elu = partial(fc, normalizer_fn=None, activation_fn=elu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_elu(x, 512)
        y = fc_elu(y, 384)
        y = fc_elu(y, 256)
        z_mu = fc(y, z_dim)
        z_log_sigma_sq = fc(y, z_dim)
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),
                           mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        return z, z_mu, z_log_sigma_sq

def relu_encoder(x, z_dim = 2, reuse=True, name = "encoder"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 512)
        y = fc_relu(y, 384)
        y = fc_relu(y, 256)
        z_mu = fc(y, z_dim)
        z_log_sigma_sq = fc(y, z_dim)
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),
                           mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        return z, z_mu, z_log_sigma_sq


def decoder(z, x_dim = 784, reuse=True, name = "decoder"):
    fc_elu = partial(fc, normalizer_fn=None, activation_fn=elu)
    with tf.variable_scope(name, reuse=reuse):
        x = fc_elu(z, 256)
        x = fc_elu(x, 384)
        x = fc_elu(x, 512)
        x = tf.sigmoid(fc(x, x_dim))
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x

def relu_decoder(z, x_dim=784, reuse=True, name="decoder"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        x = fc_relu(z, 256)
        x = fc_relu(x, 384)
        x = fc_relu(x, 512)
        x = tf.sigmoid(fc(x, x_dim))
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x

#output softmax
def encoder2(x, out_dim = 4, reuse=True, name = "encoder"):
    fc_elu = partial(fc, normalizer_fn=None, activation_fn=elu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_elu(x, 512)
        y = fc_elu(y, 384)
        y = fc_elu(y, 256)
        y = tf.nn.softmax(fc(y, out_dim))
        return y

#encode image into label and style
def encoder3(x, reuse=True, name = "encoder"):
    fc_elu = partial(fc, normalizer_fn=None, activation_fn=elu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_elu(x, 512)
        y = fc_elu(y, 384)
        y = fc_elu(y, 256)
        label = tf.nn.softmax(fc(y, 3))
        style = fc(y, 2)
        return label, style

#decode label and style to image
def decoder3(label, style, reuse=True, name = "decoder"):
    fc_elu = partial(fc, normalizer_fn=None, activation_fn=elu)
    with tf.variable_scope(name, reuse=reuse):
        x = tf.concat([label, style], 1)
        x = fc_elu(x, 256)
        x = fc_elu(x, 384)
        x = fc_elu(x, 512)
        x = tf.sigmoid(fc(x, 784))
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x