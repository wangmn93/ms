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
style_dim = 2
label_dim = 3
n_critic = 1 #
n_generator = 1
gan_type="aae-label-style"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[2,3,5]) #range -1 ~ 1

""" graphs """
encoder = models.encoder3
decoder = models.decoder3
optimizer = tf.train.AdamOptimizer
discriminator = models.discriminator_for_latent
# classifier = models.ss_discriminator
one_hot_labels = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
p = [1/3.]*3

def sample_from_categorical(size, k=3):
    return [one_hot_labels[index] for index in np.random.choice(range(k), size=size, p=p)]

# def sample_from_gaussian(mean, cov, size):
#     return np.random.multivariate_normal(mean=mean, cov=cov, size=size)

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# random_z = tf.placeholder(tf.float32, shape=[None, z_dim])
sample_labels = tf.placeholder(tf.float32, shape=[None, 3])
sample_styles = tf.placeholder(tf.float32, shape=[None, 2])

sample_l_1 = tf.placeholder(tf.float32, shape=[None, 3])
sample_l_2 = tf.placeholder(tf.float32, shape=[None, 3])
sample_l_3 = tf.placeholder(tf.float32, shape=[None, 3])


# encoder
labels, styles = encoder(real, reuse=False)

#decoder
x_hat = decoder(label=labels, style=styles, reuse=False)

real_flatten = tf.reshape(real, [-1, 784])
x_hat_flatten = tf.reshape(x_hat, [-1, 784])

epsilon = 1e-10
recon_loss = -tf.reduce_sum(
    real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
            axis=1
        )
recon_loss = tf.reduce_mean(recon_loss)

#discriminator for label
r_logit = discriminator(sample_labels, reuse=False)
f_logit = discriminator(labels)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
d_loss = d_loss_real + d_loss_fake

label_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))

# loss = recon_loss + label_loss
loss = recon_loss

#discriminator for style
r_logit_2 = discriminator(sample_styles, name="style_d", reuse=False)
f_logit_2 = discriminator(styles,name="style_d")

d_loss_real_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit_2, labels=tf.ones_like(r_logit_2)))
d_loss_fake_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_2, labels=tf.zeros_like(f_logit_2)))
d_loss_2 = d_loss_real_2 + d_loss_fake_2

style_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_2, labels=tf.ones_like(f_logit_2)))
# loss += style_loss
l_s_loss = label_loss + style_loss
# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
dis_var = [var for var in T_vars if var.name.startswith('discriminator')]
dis_var_2 = [var for var in T_vars if var.name.startswith('style_d')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr).minimize(loss, var_list=en_var+de_var, global_step=global_step)
d_step = optimizer(learning_rate=lr, beta1=0.9).minimize(d_loss, var_list=dis_var)
d2_step = optimizer(learning_rate=lr, beta1=0.9).minimize(d_loss_2, var_list=dis_var_2)
l_s_step = optimizer(learning_rate=lr).minimize(l_s_loss, var_list=en_var)
# l_s_step = optimizer(learning_rate=lr).minimize(label_loss+style_loss, var_list=encoder)

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
tf.summary.scalar('D_loss_2', d_loss_2)

images_form_de = decoder(sample_labels, sample_styles)
images_form_c1 = decoder(sample_l_1, sample_styles)
images_form_c2 = decoder(sample_l_2, sample_styles)
images_form_c3= decoder(sample_l_3, sample_styles)

tf.summary.image('Generator_image', images_form_de, 12)
tf.summary.image('Generator_image_c1', images_form_c1, 12)
tf.summary.image('Generator_image_c2', images_form_c2, 12)
tf.summary.image('Generator_image_c3', images_form_c3, 12)

# tf.summary.histogram('mu_1', mu_1)
# tf.summary.histogram('mu_2', mu_2)
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
    feed = {sample_labels: sample_from_categorical(size=rows*columns),
            sample_l_1: [one_hot_labels[0] for _ in range(rows*columns)],
            sample_l_2: [one_hot_labels[1] for _ in range(rows*columns)],
            sample_l_3: [one_hot_labels[2] for _ in range(rows*columns)],
            sample_styles:np.random.normal(size=[rows*columns, style_dim])}
    list_of_generators = [images_form_de, images_form_c1, images_form_c2, images_form_c3]  # used for sampling images
    list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it,'it%d-c3.jpg' % it,]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

# def plot_latent_space():
#     real_ipt = (data_pool.batch('img') + 1) / 2.
#     real_ipt2 = (data_pool.batch('img') + 1) / 2.
#     real_ipt3 = (data_pool.batch('img') + 1) / 2.
#
#     _ = sess.run([z], feed_dict={real: real_ipt})

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt = (data_pool.batch('img')+1)/2.

        _ = sess.run([vae_step], feed_dict={real: real_ipt})

        _ = sess.run([d_step], feed_dict={real: real_ipt, sample_labels: sample_from_categorical(size=batch_size)})

        _ = sess.run([d2_step], feed_dict={real: real_ipt, sample_styles: np.random.normal(size=[batch_size, style_dim])})

        _ = sess.run([l_s_step], feed_dict={real: real_ipt})


        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1)/2.
            summary = sess.run(merged, feed_dict={real: real_ipt,
                                                  sample_labels: sample_from_categorical(size=batch_size),
                                                  sample_l_1: [one_hot_labels[0] for _ in range(batch_size)],
                                                  sample_l_2: [one_hot_labels[1] for _ in range(batch_size)],
                                                  sample_l_3: [one_hot_labels[2] for _ in range(batch_size)],
                                                  sample_styles: np.random.normal(size=[batch_size, style_dim])})
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
