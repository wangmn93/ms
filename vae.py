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
lr = 1e-3
z_dim = 10
n_critic = 1 #
n_generator = 1
gan_type="vae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size) #range -1 ~ 1


""" graphs """
encoder = models.encoder
decoder = models.decoder
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
loss = recon_loss + latent_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr).minimize(loss, var_list=en_var+de_var, global_step=global_step)

""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', loss)
images_form_de = decoder(random_z)
tf.summary.image('Generator_image', images_form_de, 12)
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
    list_of_generators = [images_form_de]  # used for sampling images
    list_of_names = ['it%d-de.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt = (data_pool.batch('img')+1)/2.
        _ = sess.run([d_step], feed_dict={real: real_ipt})

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
