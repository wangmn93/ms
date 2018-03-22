from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import utils
import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils


""" param """
epoch = 200
batch_size = 64
lr = 0.0002
z_dim = 100
beta = 1
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="wgan-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)
tf.set_random_seed(1234)

# restore = False
# ckpt_dir =

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0, 9])

""" graphs """
generator = models.generator
discriminator = models.discriminator
optimizer = tf.train.AdamOptimizer



def gmm_(z, name="gmm", reuse=True):
    with tf.variable_scope(name, reuse=False):
        mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(1))
        mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(-1))
        log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0))
        log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0))
        # clusters
        z1 = mu_1 + z * tf.sqrt(tf.exp(log_sigma_sq1))
        z2 = mu_2 + z * tf.sqrt(tf.exp(log_sigma_sq2))
        init_gmm = tf.initialize_variables([mu_1, mu_2, log_sigma_sq1, log_sigma_sq2])
        return z1, z2, init_gmm


# inputs
# real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

with tf.variable_scope("gmm", reuse=False):
    mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(0))
    mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(0))
    log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0))
    log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0))
    # clusters
z1 = mu_1 + z * tf.sqrt(tf.exp(log_sigma_sq1))
z2 = mu_2 + z * tf.sqrt(tf.exp(log_sigma_sq2))
init_gmm = tf.initialize_variables([mu_1, mu_2, log_sigma_sq1, log_sigma_sq2])


# generator
fake = generator(z1, reuse=False, training=False)
fake2 = generator(z2, training=False)

# discriminator
# r_logit = discriminator(real, reuse=False, name="d1")
f1_logit = discriminator(fake, reuse=False, training=False)
f2_logit = discriminator(fake2, training=False)

#supplement discriminator
f1_c = discriminator(fake, reuse=False, name="s_discriminator")
f2_c = discriminator(fake2, name="s_discriminator")

#discriminator loss
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
# D_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.zeros_like(f1_logit)))
# D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
# d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2
# d_loss = D_loss_real + D_loss_fake1

#supplement discriminator loss
D2_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
D2_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
d2_loss = D2_loss_f1 + D2_loss_f2

#generator loss
# g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.ones_like(f1_logit)))
# g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))
# g1_loss += beta*D2_loss_f1
# g2_loss += beta*D2_loss_f2
# g_loss = g1_loss + g2_loss

#KL-divergence
latent_loss_1 = -0.5 * tf.reduce_sum(1 + log_sigma_sq1 - tf.square(mu_1) - tf.exp(log_sigma_sq1))
latent_loss_2 = -0.5 * tf.reduce_sum(1 + log_sigma_sq2 - tf.square(mu_2) - tf.exp(log_sigma_sq2))

#gmm loss
c_loss_1 = -tf.reduce_mean(f1_logit) + D2_loss_f1 + latent_loss_1
c_loss_2 = -tf.reduce_mean(f2_logit) + D2_loss_f2 + latent_loss_2
gmm_loss = c_loss_1 + c_loss_2

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
d2_var = [var for var in T_vars if var.name.startswith('s_discriminator')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d2_loss, var_list=d2_var, global_step=global_step)
gmm_step = optimizer(learning_rate=lr).minimize(gmm_loss, var_list=gmm_var)

""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
wgan_saver = tf.train.Saver(var_list=d_var+g_var)

# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('C1_loss', c_loss_1)
tf.summary.scalar('C2_loss', c_loss_1)
# tf.summary.scalar('G_loss', g_loss)
# tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
image_from_g = generator(z, training= False)
images_from_c1 = generator(z1, training= False)
images_from_c2 = generator(z2, training= False)
tf.summary.image('Generator_images', image_from_g, 12)
tf.summary.image('C1_images', images_from_c1, 12)
tf.summary.image('C2_images', images_from_c2, 12)
tf.summary.histogram('mu_1', tf.reduce_mean(mu_1))
tf.summary.histogram('mu_2', tf.reduce_mean(mu_2))
merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
# ckpt_dir = './checkpoints/mnist_wgan'
# utils.mkdir(ckpt_dir + '/')
# if not utils.load_checkpoint(ckpt_dir, sess):
sess.run(init_gmm)#initialize gmm params
sess.run(tf.initialize_variables(d2_var))#initialize supplement discriminator
tf.global_variables_initializer().run()
wgan_saver.restore(sess, "results/wgan-param-0-9/checkpoint/WGAN-model.ckpt")#restore wgan

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch
def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):

        for i in range(n_critic):
            # real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([d2_step], feed_dict={z: z_ipt})
            # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})

        # train GMM
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # _, _ = sess.run([g_step,g2_step], feed_dict={z: z_ipt})
            _ = sess.run([gmm_step], feed_dict={z: z_ipt})

        if it%10 == 0 :
            # real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={z: z_ipt})
            writer.add_summary(summary, it)

    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        training(max_it, it_offset + max_it)

def test():
    z_ipt = np.random.normal(size=[batch_size, z_dim])
    summary = sess.run(merged, feed_dict={z: z_ipt})
    writer.add_summary(summary, 1)

total_it = 0
try:
    # training(max_it,0)
    # total_it = sess.run(global_step)
    # print("Total iterations: "+str(total_it))
    test()
except Exception, e:
    traceback.print_exc()
finally:
    var = raw_input("Save sample images?")
    if var.lower() == 'y':
        list_of_generators = [images_from_c1, images_from_c1]  # used for sampling images
        list_of_names = ['c1-it%d.jpg'%total_it,'c2-it%d.jpg'%total_it]
        rows = 10
        columns = 10
        sample_imgs = sess.run(list_of_generators, feed_dict={z: np.random.normal(size=[rows*columns, z_dim])})
        save_dir = dir + "/sample_imgs"
        utils.mkdir(save_dir + '/')
        for imgs,name in zip(sample_imgs,list_of_names):
            my_utils.saveSampleImgs(imgs=imgs, full_path=save_dir+"/"+name, row=rows,column=columns)
    # save checkpoint
    var = raw_input("Save models?")
    if var.lower() == 'y':
        save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
        print("Model saved in path: %s" % save_path)

    #close session
    print(" [*] Close main session!")
    sess.close()
