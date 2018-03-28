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
epoch = 100
batch_size = 100
lr = 2e-4
lr_2 = 1e-3
z_dim = 2
n_critic = 1 #
n_generator = 1
gan_type="vade-via-aae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0,3,7]) #range -1 ~ 1

# k=4
def gmm(z):
    with tf.variable_scope("gmm", reuse=False):
        mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(0.01))
        mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(-0.01))
        mu_3 = tf.get_variable("mean3", [z_dim], initializer=tf.constant_initializer(0.005))
        log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0.001))
        log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0.001))
        log_sigma_sq3 = tf.get_variable("log_sigma_sq3", [z_dim], initializer=tf.constant_initializer(0.001))
        z1 = mu_1 + z * tf.sqrt(tf.exp(log_sigma_sq1))
        z2 = mu_2 + z * tf.sqrt(tf.exp(log_sigma_sq2))
        z3 = mu_3 + z * tf.sqrt(tf.exp(log_sigma_sq3))
        return z1, z2, z3, mu_1, mu_2, mu_3
    # mus =  tf.constant([[0.5, 0.5],[-0.5, 0.5],[-0.5,-0.5],[0.5, -0.5]],shape=[4,2],dtype=tf.float32)
    # log_sigma_sqs = tf.constant([[00.1, 00.1],[00.1, 00.1],[00.1,00.1],[00.1, 00.1]],shape=[4,2], dtype=tf.float32)

""" graphs """
encoder = models.encoder
decoder = models.decoder
discriminator = models.discriminator_for_latent
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
random_z = tf.placeholder(tf.float32, shape=[None, z_dim])

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
loss = recon_loss +latent_loss

#GAN
z1, z2, z3, mu1, mu2, mu3 = gmm(random_z)
print(z1.shape)
r_logit_1 = discriminator(z1, name="d1",reuse=False) + epsilon
r_logit_2 = discriminator(z2, name="d1") + epsilon
r_logit_3 = discriminator(z3, name="d1") + epsilon
f_logit = discriminator(z, name="d1") + epsilon
#
d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit_1, labels=tf.ones_like(r_logit_1)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit_2, labels=tf.ones_like(r_logit_2)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit_3, labels=tf.ones_like(r_logit_3)))

d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
#
d_loss_1 = d_loss_real_1 + d_loss_fake_1
#
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))

# loss +=  g_loss

f_logit_21 = discriminator(z1, name="d2",reuse=False) + epsilon
f_logit_22= discriminator(z2, name="d2") + epsilon
f_logit_23 = discriminator(z3, name="d2") + epsilon
r_logit = discriminator(z, name="d2") + epsilon

d_loss_fake_21 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_21, labels=tf.zeros_like(f_logit_21)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_22, labels=tf.zeros_like(f_logit_22)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_23, labels=tf.zeros_like(f_logit_23)))

d_loss_real_21 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))

d_loss_2 = d_loss_real_21 + d_loss_fake_21

gmm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_21, labels=tf.ones_like(f_logit_21)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_22, labels=tf.ones_like(f_logit_22)))\
    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_23, labels=tf.ones_like(f_logit_23)))

# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
d_var = [var for var in T_vars if var.name.startswith('d1')]
d2_var = [var for var in T_vars if var.name.startswith('d2')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr_2).minimize(loss, var_list=en_var+de_var, global_step=global_step)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss_1, var_list=d_var)
d2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss_2, var_list=d2_var)
gmm_step = optimizer(learning_rate=lr_2).minimize(gmm_loss, var_list=gmm_var)
# g_step = optimizer(learning_rate=lr).minimize(g_loss, var_list=de_var)

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
tf.summary.scalar('Recon_loss', recon_loss)
tf.summary.scalar('D1_loss', d_loss_1)
tf.summary.scalar('D2_loss', d_loss_2)
tf.summary.scalar('gmm_loss', gmm_loss)
# tf.summary.scalar('G_loss', g_loss)

# onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=4)
# onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=4)
# onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=4)
# onehot_labels_three = tf.one_hot(indices=tf.cast(tf.scalar_mul(3,tf.ones(batch_size)), tf.int32), depth=4)
#
# eps2 = tf.random_normal(shape=tf.shape(log_sigma_sqs),
#                        mean=0, stddev=1, dtype=tf.float32)
# cs2 = mus + tf.sqrt(tf.exp(log_sigma_sqs)) * eps2
# z1 = tf.matmul(onehot_labels_zero,mus)
# z2 = tf.matmul(onehot_labels_one,mus)
# z3 = tf.matmul(onehot_labels_two,mus)
# z4 = tf.matmul(onehot_labels_three,mus)

images_form_de = decoder(random_z)
images_form_c1 = decoder(z1)
images_form_c2 = decoder(z2)
images_form_c3 = decoder(z3)

tf.summary.image('real', real, 12)
tf.summary.image('recon', x_hat, 12)


tf.summary.image('Generator_image', images_form_de, 12)
tf.summary.image('Generator_image_1', images_form_c1, 12)
tf.summary.image('Generator_image_2', images_form_c2, 12)
tf.summary.image('Generator_image_3', images_form_c3, 12)

tf.summary.histogram('c1 points', mu1)
tf.summary.histogram('c2 points', mu2)
tf.summary.histogram('c3 points', mu3)
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
    feed = {random_z: np.random.normal(size=[rows * columns, z_dim])}
    list_of_generators = [images_form_de, images_form_c1, images_form_c2, images_form_c3]  # used for sampling images
    list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it, 'it%d-c3.jpg' % it]
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
        _ = sess.run([d2_step], feed_dict={real: real_ipt, random_z: z_ipt})
        _ = sess.run([gmm_step], feed_dict={random_z: z_ipt})


        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1)/2.
            # z_ipt =  np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt, random_z: np.random.normal(size=[batch_size, z_dim])})
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
