from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils


""" param """
epoch = 100
batch_size = 100
lr = 1e-4
z_dim = 2
n_critic = 1 #
n_generator = 1
gan_type="vae-mad-gan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[1,5,9]) #range -1 ~ 1

""" graphs """
encoder = models.encoder
decoder = models.decoder
mad_discriminator = models.simple_mad_discriminator
mad_generator = models.simple_mad_generator
discriminator = models.discriminator_for_latent
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
random_z = tf.placeholder(tf.float32, shape=[None, z_dim])
random_z_2 = tf.placeholder(tf.float32, shape=[None, z_dim])

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

# Latent loss
# Kullback Leibler divergence: measure the difference between two distributions
# Here we measure the divergence between the latent distribution and N(0, 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=1)
latent_loss = tf.reduce_mean(latent_loss)
loss = recon_loss + latent_loss

#GAN
onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=4)
onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=4)
onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=4)
onehot_labels_three = tf.one_hot(indices=tf.cast(tf.scalar_mul(3,tf.ones(batch_size)), tf.int32), depth=4)

# generator
fake1,fake2,fake3 = mad_generator(random_z_2, reuse=False)

# discriminator
r_logit = mad_discriminator(z, reuse=False)
f1_logit = mad_discriminator(fake1)
f2_logit = mad_discriminator(fake2)
f3_logit = mad_discriminator(fake3)

D_loss_real = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=r_logit, onehot_labels=onehot_labels_zero))
D_loss_fake1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_one))
D_loss_fake2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_two))
D_loss_fake3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f3_logit, onehot_labels=onehot_labels_three))
d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2 + D_loss_fake3

#generator loss
g1_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_zero))
g2_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_zero))
g3_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f3_logit, onehot_labels=onehot_labels_zero))
g_loss = g1_loss + g2_loss + g3_loss


# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr).minimize(loss, var_list=en_var+de_var, global_step=global_step)
d_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=d_var)
g_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=g_var)

""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', loss)
tf.summary.scalar('D_loss', d_loss)
tf.summary.scalar('G_loss', g_loss)

images_form_de = decoder(random_z)
z1, z2, z3 = mad_generator(random_z_2, training=False)
images_form_c1 = decoder(z1)
images_form_c2 = decoder(z2)
images_form_c3 = decoder(z3)

tf.summary.image('real', real, 12)
tf.summary.image('recon', x_hat, 12)


tf.summary.image('Generator_image', images_form_de, 12)
tf.summary.image('Generator_image_1', images_form_c1, 12)
tf.summary.image('Generator_image_2', images_form_c2, 12)
tf.summary.image('Generator_image_3', images_form_c3, 12)

tf.summary.histogram('c1 points', z1)
tf.summary.histogram('c2 points', z2)
tf.summary.histogram('c3 points', z3)
# tf.summary.histogram('c4 points', z4)

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
    feed = {random_z: np.random.normal(size=[rows * columns, z_dim]),
            random_z_2: np.random.normal(size=[rows * columns, z_dim])}
    list_of_generators = [images_form_de, images_form_c1, images_form_c2, images_form_c3]  # used for sampling images
    list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it, 'it%d-c3.jpg' % it,]
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
        _ = sess.run([vae_step], feed_dict={real: real_ipt})
        _ = sess.run([d_step], feed_dict={real: real_ipt, random_z_2: z_ipt})
        # _ = sess.run([g_step], feed_dict={random_z: z_ipt})


        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1)/2.
            z_ipt =  np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,
                                                  random_z:z_ipt,
                                                  random_z_2: z_ipt})
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
    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
