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
epoch = 100
batch_size = 64
lr = 0.0002
z_dim = 256
clip = 0.01
n_critic = 5

gan_type="toy_wgan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

''' data '''
mus = [[1, 10],[50, 60]]
cov = [[0.001, 0], [0, 0.001]]
data_pool = my_utils.getToyDatapool(batch_size,mus, cov, 30000)

""" graphs """

generator = models.toy_generator
discriminator = models.toy_discriminator


# inputs
real = tf.placeholder(tf.float32, shape=[None, 2])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# generator
fake = generator(z, reuse=False)

# dicriminator
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)

# losses
wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
d_loss = -wd
g_loss = -tf.reduce_mean(f_logit)

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]

# otpims
d_step_ = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
with tf.control_dependencies([d_step_]):
    d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip, clip)) for var in d_var))
g_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)


""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
points_from_generator = generator(z, name="generator")
tf.summary.histogram('generator points', points_from_generator)
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
    batch_epoch = len(data_pool) // (batch_size * n_critic)
    max_it = epoch * batch_epoch


    for it in range(max_it):

        for i in range(n_critic):
            real_ipt = data_pool.batch('point')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})


        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([g_step], feed_dict={z: z_ipt})

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
