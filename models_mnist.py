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
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
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

#mimic
def generator2(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

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
        y = fc_relu(z, 128)
        y = fc_relu(y, 128)
        return fc(y, 2)

def toy_discriminator(x, reuse=True, name = "discriminator"):
    fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 128)
        return fc(y, 1)

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

def ss_discriminator(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y =  tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

####
def c_generator(z, label, reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.concat([z, label], 1)
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

#enlarge capacity of discriminator??
def multi_c_discriminator2(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        y = fc_lrelu(y, 1024)
        return fc(y, 3)
