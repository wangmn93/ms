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
epoch = 200
batch_size = 64
lr = 0.0002
z_dim = 100


gan_type="wgan-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0, 9])

""" graphs """

generator = models.ss_generator
discriminator = models.ss_discriminator

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

with tf.variable_scope("gmm", reuse=False):
    mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(0.1))
    mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(-0.1))
    log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(1))
    log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(1))

#cluster
z1 = mu_1 + z * tf.sqrt(tf.exp(log_sigma_sq1))
z2 = mu_2 + z * tf.sqrt(tf.exp(log_sigma_sq2))

# generator
fake = generator(z, reuse=False)
fake_1 = generator(z1)
fake_2 = generator(z2)

# discriminator with gradient clip
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)
f_logit_1 = discriminator(fake_1)
f_logit_2 = discriminator(fake_2)

# supplement discriminator
f1_c = discriminator(fake_1, reuse=False, name="discriminator_2")
f2_c = discriminator(fake_2, name="discriminator_2")

# wasserstein losses
wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
d_loss = -wd
g_loss = -tf.reduce_mean(f_logit)

# supplement discriminator loss
D2_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
D2_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
d_loss_2 = D2_loss_f1 + D2_loss_f2

#KL divergence
latent_loss_1 = -0.5 * tf.reduce_sum(1 + log_sigma_sq1 - tf.square(mu_1) - tf.exp(log_sigma_sq1))
latent_loss_2 = -0.5 * tf.reduce_sum(1 + log_sigma_sq2 - tf.square(mu_2) - tf.exp(log_sigma_sq2))

#gmm loss
c_loss_1 = -tf.reduce_mean(f_logit_1) + D2_loss_f1 + latent_loss_1
c_loss_2 = -tf.reduce_mean(f_logit_2) + D2_loss_f2 + latent_loss_2
gmm_loss = c_loss_1 + c_loss_2

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
d_var_2 = [var for var in T_vars if var.name.startswith('discriminator_2')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]

# otpims
d_step_ = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
with tf.control_dependencies([d_step_]):
    d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip, clip)) for var in d_var))
d_step_2 = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(d_loss_2, var_list=d_var_2)
g_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)
c_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(gmm_loss, var_list=gmm_var)


""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('C1_loss', c_loss_1)
tf.summary.scalar('C2_loss', c_loss_2)
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('Supplement_Discriminator_loss', d_loss_2)
images_from_generator = generator(z, training=False)
images_from_cluster_1 = generator(z1, training=False)
images_from_cluster_2 = generator(z2, training=False)
tf.summary.image('Generated_images', images_from_generator, 12)
tf.summary.image('Generated_images_cluster_1', images_from_cluster_1, 12)
tf.summary.image('Generated_images_cluster_2', images_from_cluster_2, 12)
tf.summary.histogram('mu_1', tf.reduce_mean(mu_1))
tf.summary.histogram('mu_2', tf.reduce_mean(mu_2))
# tf.summary.histogram('log_sigma_sq_1', log_sigma_sq1)
# tf.summary.histogram('log_sigma_sq_2', log_sigma_sq2)

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


    for it in range(max_it):

        for i in range(n_critic):
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
            if i == 0:
                #train supplement discriminator
                _ = sess.run([d_step_2], feed_dict={real: real_ipt, z: z_ipt})

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _, _ = sess.run([g_step, c_step], feed_dict={z: z_ipt})


        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
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
