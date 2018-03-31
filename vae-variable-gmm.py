from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils

#use a gmm to fit the latent space of vae
#driven by a binary discriminator->tell from real and fake
#and a supplement discriminator to push gmm apart
""" param """
epoch = 100
batch_size = 100
lr = 1e-3
z_dim = 2
n_critic = 1 #
n_generator = 1
gan_type="vae-variable-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0,9]) #range -1 ~ 1


""" graphs """
encoder = models.encoder
decoder = models.decoder
optimizer = tf.train.AdamOptimizer
discriminator = models.ss_discriminator
classifier = models.ss_discriminator

with tf.variable_scope("gmm", reuse=False):
    mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(-0.1))
    mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(0.2))
    # mu_3 = tf.get_variable("mean3", [z_dim], initializer=tf.constant_initializer(0.1))
    log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0.001))
    log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0.001))
    # log_sigma_sq3 = tf.get_variable("log_sigma_sq3", [z_dim], initializer=tf.constant_initializer(0.001))

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
random_z = tf.placeholder(tf.float32, shape=[None, z_dim])

z1 = mu_1 + random_z * tf.sqrt(tf.exp(log_sigma_sq1))
z2 = mu_2 + random_z * tf.sqrt(tf.exp(log_sigma_sq2))
# z3 = mu_3 + random_z * tf.sqrt(tf.exp(log_sigma_sq3))

# encoder
z, z_mu, z_log_sigma_sq = encoder(real, reuse=False)

#decoder
x_hat = decoder(z, reuse=False)

real_flatten = tf.reshape(real, [-1, 784])
x_hat_flatten = tf.reshape(x_hat, [-1, 784])

epsilon = 1e-10
recon_loss = -tf.reduce_sum(
    real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
            axis=1
        )
recon_loss = tf.reduce_mean(recon_loss)
# recon_loss = tf.losses.mean_squared_error(labels=real, predictions=x_hat)
# recon_loss = tf.reduce_mean(recon_loss)

# Latent loss
# Kullback Leibler divergence: measure the difference between two distributions
# Here we measure the divergence between the latent distribution and N(0, 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=1)
latent_loss = tf.reduce_mean(latent_loss)
loss = recon_loss + latent_loss

#discriminator
fake = decoder(random_z)
fake_1 = decoder(z1)
fake_2 = decoder(z2)
# fake_3 = decoder(z3)
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
d_loss = d_loss_real + d_loss_fake

#classifier
onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=3)
onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=3)
onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=3)

c_1 = classifier(fake_1, reuse=False, name="classifier")
c_2 = classifier(fake_2, name="classifier")
# c_3 = classifier(fake_3, name="classifier")
# print(sess.run(tf.shape(onehot_labels_zero)))
c_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_1, labels=tf.zeros_like(c_1)))
c_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_2, labels=tf.ones_like(c_2)))
# c_loss_3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_3, onehot_labels=onehot_labels_two))
c_loss = c_loss_1 + c_loss_2

#gmm
f_logit_1 = discriminator(fake_1)
f_logit_2 = discriminator(fake_2)
# f_logit_3 = discriminator(fake_3)
gmm_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_1, labels=tf.ones_like(f_logit_1)))
gmm_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_2, labels=tf.ones_like(f_logit_2)))
# gmm_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_3, labels=tf.ones_like(f_logit_3)))

latent_loss_1 = -0.5 * tf.reduce_sum(1 + log_sigma_sq1 - tf.square(mu_1) - tf.exp(log_sigma_sq1))
latent_loss_2 = -0.5 * tf.reduce_sum(1 + log_sigma_sq2 - tf.square(mu_2) - tf.exp(log_sigma_sq2))
gmm_loss = gmm_loss_1 + gmm_loss_2 + c_loss + latent_loss_1 + latent_loss_2

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
dis_var = [var for var in T_vars if var.name.startswith('discriminator')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]
c_var = [var for var in T_vars if var.name.startswith('classifier')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr).minimize(loss, var_list=en_var+de_var, global_step=global_step)
d_step = optimizer(learning_rate=lr).minimize(d_loss, var_list=dis_var)
c_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(c_loss, var_list=c_var)
gmm_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(gmm_loss, var_list=gmm_var)

""" train """
''' init '''
# session
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', loss)
tf.summary.scalar('D_loss', d_loss)
tf.summary.scalar('C_loss', c_loss)
images_form_de = decoder(random_z)
images_form_c1 = decoder(z1)
images_form_c2 = decoder(z2)
# images_form_c3= decoder(z3)
tf.summary.image('Generator_image', images_form_de, 12)
tf.summary.image('Generator_image_c1', images_form_c1, 12)
tf.summary.image('Generator_image_c2', images_form_c2, 12)
# tf.summary.image('Generator_image_c3', images_form_c3, 12)

tf.summary.histogram('mu_1', mu_1)
tf.summary.histogram('mu_2', mu_2)
# tf.summary.histogram('mu_3', mu_3)

merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

def sample_once(it):
    rows = 10
    columns = 10
    feed = {random_z: np.random.normal(size=[rows*columns, z_dim])}
    list_of_generators = [images_form_de, images_form_c1, images_form_c2]  # used for sampling images
    list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt = (data_pool.batch('img')+1)/2.
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _, _ = sess.run([vae_step, d_step], feed_dict={real: real_ipt, random_z:z_ipt})
        if it>10000:
            _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1)/2.
            z_ipt =  np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,random_z: z_ipt})
            writer.add_summary(summary, it)

    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        sample_once(it_offset + max_it)
        print("Save sample images")
        training(max_it, it_offset + max_it)



total_it = 0
try:
    training(max_it,0)
    total_it = sess.run(global_step)
    print("Total iterations: "+str(total_it))
except Exception, e:
    traceback.print_exc()
finally:
    var = raw_input("Save sample images?")
    if var.lower() == 'y':
        sample_once(total_it)
        # rows = 10
        # columns = 10
        # feed = {z: np.random.normal(size=[rows * columns, z_dim]),
        #         z1:np.random.normal(loc=mus[0], scale=vars[0], size=[rows * columns, z_dim]),
        #         z2: np.random.normal(loc=mus[1], scale=vars[1], size=[rows * columns, z_dim]),
        #         z3: np.random.normal(loc=mus[2], scale=vars[2], size=[rows * columns, z_dim]),
        #         z4: np.random.normal(loc=mus[3], scale=vars[3], size=[rows * columns, z_dim])}
        # list_of_generators = [images_form_g, images_form_c1, images_form_c2, images_form_c3, images_form_c4]  # used for sampling images
        # list_of_names = ['g-it%d.jpg'%total_it, 'c1-it%d.jpg'%total_it, 'c2-it%d.jpg'%total_it, 'c3-it%d.jpg'%total_it, 'c4-it%d.jpg'%total_it]
        # save_dir = dir + "/sample_imgs"
        # my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed, list_of_names=list_of_names, save_dir=save_dir)

    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
