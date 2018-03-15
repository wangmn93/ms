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
epoch = 75
batch_size = 64
batch_size2 = 32
lr = 0.0002
z_dim = 100
beta = 0.3#diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="mgan_modified"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

# restore = False
# ckpt_dir =

''' data '''
data_pool = my_utils.getDatapool(batch_size, keep=[0,9])

""" graphs """
generator = models.generator
d_shared_part = models.d_shared_part
shared_discriminator = models.shared_discriminator
shared_classifier = models.shared_classifier
optimizer = tf.train.AdamOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])


# generator
fake = generator(z, reuse=False, name="g1")
fake2 = generator(z, reuse=False, name="g2")

#results from shared part d
r_temp = d_shared_part(real, reuse=False)
f1_temp = d_shared_part(fake)
f2_temp = d_shared_part(fake2)

# dicriminator
r_logit = shared_discriminator(r_temp, reuse=False)
f1_logit = shared_discriminator(f1_temp)
f2_logit = shared_discriminator(f2_temp)

#classifier
f1_c = shared_classifier(f1_temp, reuse=False)
f2_c = shared_classifier(f2_temp)

#losses
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
# d_loss = D_loss_real + D_loss_fake
#
# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))
#discriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
D_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.zeros_like(f1_logit)))
D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2

#classifier loss
C_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
C_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
c_loss = C_loss_f1 + C_loss_f2
# # D C loss
d_c_loss = d_loss + C_loss_f1 + C_loss_f2

#generator loss
g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.ones_like(f1_logit)))
g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))
g1_loss += + beta*C_loss_f1
g2_loss += + beta*C_loss_f2

# trainable variables for each network
T_vars = tf.trainable_variables()

d_shared_var = [var for var in T_vars if var.name.startswith('d_shared_part')]
d_var = [var for var in T_vars if var.name.startswith('shared_discriminator')]
c_var = [var for var in T_vars if var.name.startswith('shared_classifier')]

g1_var = [var for var in T_vars if var.name.startswith('g1')]
g2_var = [var for var in T_vars if var.name.startswith('g2')]

# otpims
d_c_step = optimizer(learning_rate=lr,beta1=0.5).minimize(d_c_loss, var_list=d_shared_var+d_var+c_var)
g1_step = optimizer(learning_rate=lr,beta1=0.5).minimize(g1_loss, var_list=g1_var)
g2_step = optimizer(learning_rate=lr,beta1=0.5).minimize(g2_loss, var_list=g2_var)
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
tf.summary.scalar('G1_loss', g1_loss)
tf.summary.scalar('G2_loss', g2_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('Classifier_loss', c_loss)
tf.summary.scalar('d_c_loss', d_c_loss)
images_form_g1 = generator(z, name="g1", training=False)
images_form_g2 = generator(z, name="g2", training=False)
tf.summary.image('G1_images', images_form_g1, 10)
tf.summary.image('G2_images', images_form_g2, 10)
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
            z_ipt = np.random.normal(size=[batch_size2, z_dim])
            _ = sess.run([d_c_step], feed_dict={real: real_ipt, z: z_ipt})


        # train G
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([g1_step], feed_dict={z: z_ipt})
            _ = sess.run([g2_step], feed_dict={z: z_ipt})


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
