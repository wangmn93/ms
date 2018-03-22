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
batch_size = 128
# batch_size2 = 64
lr = 0.001
z_dim = 256
beta = 0.125 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="toy_wgan_gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

# restore = False
# ckpt_dir =

''' data '''
mus = [[1, 10],[50, 60]]
cov = [[0.001, 0], [0, 0.001]]
data_pool = my_utils.getToyDatapool(batch_size,mus, cov, 30000)

""" graphs """
generator = models.toy_generator
discriminator = models.toy_discriminator
optimizer = tf.train.GradientDescentOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 2])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# z1 = models.transform_(z, reuse=False, name="cluster1")
# z2 = models.transform_(z, reuse=False, name="cluster2")

# generator
fake1 = generator(z, reuse=False, name="g1")
# fake2 = generator(z2, name="g1")

# discriminator
r_logit = discriminator(real, reuse=False, name="d1")
f1_logit = discriminator(fake1, name="d1")
# f2_logit = discriminator(fake2, name="d1")


#supplement discriminator
# f1_c = discriminator(fake1, reuse=False, name="d2")
# f2_c = discriminator(fake2, name="d2")

#discriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.zeros_like(f1_logit)))
# D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
# d_loss = D_loss_real + D_loss_fake + D_loss_fake2
d_loss = D_loss_real + D_loss_fake
#supplement discriminator loss
# D2_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
# D2_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
# d2_loss = D2_loss_f1 + D2_loss_f2

#generator loss
g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.ones_like(f1_logit)))
# g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))
# g_loss1 += D2_loss_f1
# g_loss2 += D2_loss_f2
# g_loss = g_loss1 + g_loss2

# trainable variables for each network
T_vars = tf.trainable_variables()

d_var = [var for var in T_vars if var.name.startswith('d1')]

# d2_var = [var for var in T_vars if var.name.startswith('d2')]

g_var = [var for var in T_vars if var.name.startswith('g1')]

# g2_var = [var for var in T_vars if var.name.startswith('g2')]

# gmm_var = [var for var in T_vars if var.name.startswith('cluster')]
# c1_var = [var for var in T_vars if var.name.startswith('cluster1')]

# c2_var = [var for var in T_vars if var.name.startswith('cluster2')]

# otpims
d_step = optimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
# d2_step = optimizer(learning_rate=lr).minimize(d2_loss, var_list=d2_var)
g_step1 = optimizer(learning_rate=lr).minimize(g_loss1, var_list=g_var)
# g_step2 = optimizer(learning_rate=lr).minimize(g_loss2, var_list=g_var+c2_var)
# g_step2 = optimizer(learning_rate=lr).minimize(g_loss2, var_list=g2_var)
# g_step = optimizer(learning_rate=lr,beta1=0.5).minimize(g_loss, var_list=g1_var+g2_var)
# g1_step = optimizer(learning_rate=lr,beta1=0.5).minimize(g1_loss, var_list=g1_var)
# g2_step = optimizer(learning_rate=lr,beta1=0.5).minimize(g2_loss, var_list=g2_var)
# d_step_ = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
# g_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)


""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('G1_loss', g_loss1)
# tf.summary.scalar('G2_loss', g_loss2)
# tf.summary.scalar('G2_loss', g2_loss)
# tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
# tf.summary.scalar('D2_loss', d2_loss)

# t1 = models.transform_(z, name="cluster1")
# t2 = models.transform_(z, name="cluster2")
points_from_g = generator(z, name="g1")
# points_from_g2 = generator(t2, name="g1")
# images_form_g1 = generator(z, name="g1", training=False)
# images_form_g2 = generator(z, name="g2", training=False)
tf.summary.histogram('g1 points', points_from_g)
# tf.summary.histogram('g2 points', points_from_g2)
# tf.summary.image('G2_images', images_form_g2, 12)
merged = tf.summary.merge_all()
logdir = dir+"tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('tensorboard dir: '+logdir)

''' initialization '''
# ckpt_dir = './checkpoints/mnist_wgan'
# utils.mkdir(ckpt_dir + '/')
# if not utils.load_checkpoint(ckpt_dir, sess):
sess.run(tf.global_variables_initializer())

''' train '''
try:

    # batch_epoch = mnist.train.num_examples // (batch_size * n_critic)
    batch_epoch = len(data_pool) // (batch_size * n_critic)
    max_it = epoch * batch_epoch
    print("Max it: " + str(max_it))


    for it in range(max_it):

        for i in range(n_critic):
            real_ipt = data_pool.batch('point')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
            # _, _ = sess.run([d_step,d2_step], feed_dict={real: real_ipt, z: z_ipt})


        # train G
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([g_step1], feed_dict={z: z_ipt})
            # _, _ = sess.run([g_step, g_step2], feed_dict={z: z_ipt})


        if it%10 == 0 :
            real_ipt = data_pool.batch('point')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt})
            writer.add_summary(summary, it)

except Exception, e:
    traceback.print_exc()
finally:
    # save checkpoint
    save_path = saver.save(sess, dir+"checkpoint" + "/" + "model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
