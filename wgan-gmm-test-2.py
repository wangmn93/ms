from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import utils
import data_mnist as data
import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from tensorflow.examples.tutorials.mnist import input_data


""" param """
epoch = 400
batch_size = 64
lr = 0.0002
z_dim = 100
# clip = 0.01
n_critic = 5

gan_type="wgan-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0, 9])

""" graphs """

generator = models.generator
discriminator = models.discriminator


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

with tf.variable_scope("gmm", reuse=False):
    mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(0))
    mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(0))
    log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0.001))
    log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0.001))
init_gmm = tf.initialize_variables([mu_1, mu_2, log_sigma_sq1, log_sigma_sq2])
    # clusters
z1 = mu_1 + z * tf.sqrt(tf.exp(log_sigma_sq1))
z2 = mu_2 + z * tf.sqrt(tf.exp(log_sigma_sq2))


# generator
fake_1 = generator(z1, reuse=False, training=False)
fake_2 = generator(z2, training=False)
# fake_3 = generator(z, training=False)

# discriminator
f_logit_1 = discriminator(fake_1, reuse=False, training=False)
f_logit_2 = discriminator(fake_2, training=False)

#suplement discriminator
c_logit_1 = discriminator(fake_1,name="classifier" ,reuse=False)
c_logit_2 = discriminator(fake_2,name="classifier")

#KL
latent_loss_1 = -0.5 * tf.reduce_sum(1 + log_sigma_sq1 - tf.square(mu_1) - tf.exp(log_sigma_sq1))
latent_loss_2 = -0.5 * tf.reduce_sum(1 + log_sigma_sq2 - tf.square(mu_2) - tf.exp(log_sigma_sq2))

# losses
c_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_logit_1, labels=tf.zeros_like(c_logit_1)))
c_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_logit_2, labels=tf.ones_like(c_logit_2)))
c_loss = c_loss_1 + c_loss_2

#
gmm_loss_1 = -tf.reduce_mean(f_logit_1) + c_loss_1
gmm_loss_2 = -tf.reduce_mean(f_logit_2) + c_loss_2
gmm_loss = gmm_loss_1 + gmm_loss_2

# trainable variables for each network
G_vars = tf.global_variables()
T_vars = tf.trainable_variables()

gmm_var = [var for var in T_vars if var.name.startswith('gmm')]

c_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "classifier")
init_classifier = tf.initialize_variables(c_var)

#wgan param
d_var =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")
g_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator")
wgan_var = d_var + g_var
# wgan_var = [var for var in T_vars if var not in gmm_var]
# otpims
# d_step_ = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
# with tf.control_dependencies([d_step_]):
#     d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip, clip)) for var in d_var))
# g_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)
c_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(c_loss, var_list=c_var)
gmm_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(gmm_loss, var_list=gmm_var)

""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
wgan_saver = tf.train.Saver(var_list=wgan_var)
saver = tf.train.Saver()
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('C1_loss', gmm_loss_1)
tf.summary.scalar('C2_loss', gmm_loss_2)
tf.summary.scalar('Classifier_loss', c_loss)
images_for_tensorboard = generator(z, training=False)
images_for_1 = generator(z1, training=False)
images_for_2 = generator(z2, training=False)
tf.summary.image('Generated_images', images_for_tensorboard, 12)
tf.summary.image('C1_images', images_for_1, 12)
tf.summary.image('C2_images', images_for_2, 12)
tf.summary.histogram('mu_1', tf.reduce_mean(mu_1))
tf.summary.histogram('mu_2', tf.reduce_mean(mu_2))
merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('tensorboard dir: '+logdir)

''' initialization '''
# ckpt_dir = './checkpoints/mnist_wgan'
# utils.mkdir(ckpt_dir + '/')
# if not utils.load_checkpoint(ckpt_dir, sess):
sess.run(tf.global_variables_initializer())

wgan_saver.restore(sess, "results/wgan-param-0-9/checkpoint/WGAN-model.ckpt")
# sess.run(init_gmm)
# sess.run(init_classifier)




# batch_epoch = mnist.train.num_examples // (batch_size * n_critic)
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

def test():
    real_ipt = data_pool.batch('img')
    z_ipt = np.random.normal(size=[batch_size, z_dim])
    summary = sess.run(merged, feed_dict={real: real_ipt, z: z_ipt})
    writer.add_summary(summary, 1)

def save_img(list_of_generators, list_of_names):
    # list_of_generators = [images_for_1, images_for_2, images_for_tensorboard]  # used for sampling images
    # list_of_names = ['g1-it%d.jpg' % 1, 'g2-it%d.jpg' % 1, 'wgan-jpg']
    rows = 10
    columns = 10
    sample_imgs = sess.run(list_of_generators, feed_dict={z: np.random.normal(size=[rows * columns, z_dim])})
    save_dir = dir + "/sample_imgs"
    utils.mkdir(save_dir + '/')
    for imgs, name in zip(sample_imgs, list_of_names):
        my_utils.saveSampleImgs(imgs=imgs, full_path=save_dir + "/" + name, row=rows, column=columns)

def train_gmm():
    for it in range(max_it):
        #update classifier
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([c_step], feed_dict={z: z_ipt})

        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([gmm_step], feed_dict={z: z_ipt})

        if it % 10 == 0:
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={z: z_ipt})
            writer.add_summary(summary, it)

def train(max_it):
    for it in range(max_it):

        for i in range(n_critic):
            # batch data
            # real_ipt,_ = mnist.train.next_batch(batch_size)
            # real_ipt = tf.image.resize_images(real_ipt, [32, 32]).eval()
            # real_ipt = (real_ipt-0.5)/0.5
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})


        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        # _ = sess.run([g_step], feed_dict={z: z_ipt})


        if it%10 == 0 :
            # real_ipt,_ = mnist.train.next_batch(batch_size)
            # real_ipt = tf.image.resize_images(real_ipt, [32,32]).eval()
            # real_ipt = (real_ipt - 0.5) / 0.5
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt})
            writer.add_summary(summary, it)
''' train '''
try:
    train_gmm()
except Exception, e:
    traceback.print_exc()
finally:
    save_img([images_for_1, images_for_2, images_for_tensorboard], ['g1-it%d.jpg' % 1, 'g2-it%d.jpg' % 1, 'wgan.jpg'])
    # save checkpoint
    # save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    # print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
