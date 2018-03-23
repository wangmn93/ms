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
batch_size = 128
batch_size2 = 64
lr = 0.0002
z_dim = 100
beta = 1 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="selective_sampling"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)
tf.set_random_seed(1234)

# restore = False
# ckpt_dir =

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[1, 8])

""" graphs """
generator = models.ss_generator
discriminator = models.ss_discriminator
optimizer = tf.train.AdamOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])


# generator
fake = generator(z, reuse=False, name="g1")
fake2 = generator(z, reuse=False, name="g2")

# discriminator
r_logit = discriminator(real, reuse=False, name="d1")
f1_logit = discriminator(fake, name="d1")
f2_logit = discriminator(fake2, name="d1")

#supplement discriminator
f1_c = discriminator(fake, reuse=False, name="d2")
f2_c = discriminator(fake2, name="d2")

#discriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
D_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.zeros_like(f1_logit)))
D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2
# d_loss = D_loss_real + D_loss_fake1

#supplement discriminator loss
D2_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
D2_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
d2_loss = D2_loss_f1 + D2_loss_f2

#generator loss
g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.ones_like(f1_logit)))
g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))
g1_loss += beta*D2_loss_f1
g2_loss += beta*D2_loss_f2
g_loss = g1_loss + g2_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('d1')]
d2_var = [var for var in T_vars if var.name.startswith('d2')]
g1_var = [var for var in T_vars if var.name.startswith('g1')]
g2_var = [var for var in T_vars if var.name.startswith('g2')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
d2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d2_loss, var_list=d2_var)
g_step = optimizer(learning_rate=lr).minimize(g1_loss, var_list=g1_var)
g2_step = optimizer(learning_rate=lr).minimize(g2_loss, var_list=g2_var)
G_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g1_var + g2_var)
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
tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
images_form_g1 = generator(z, name="g1", training= False)
images_form_g2 = generator(z, name="g2", training= False)
tf.summary.image('G1_images', images_form_g1, 12)
tf.summary.image('G2_images', images_form_g2, 12)
merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
# ckpt_dir = './checkpoints/mnist_wgan'
# utils.mkdir(ckpt_dir + '/')
# if not utils.load_checkpoint(ckpt_dir, sess):
sess.run(tf.global_variables_initializer())

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch
def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):

        for i in range(n_critic):
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size2, z_dim])
            _, _ = sess.run([d_step, d2_step], feed_dict={real: real_ipt, z: z_ipt})
            # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})

        # train G
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size2, z_dim])
            # _, _ = sess.run([g_step,g2_step], feed_dict={z: z_ipt})
            _ = sess.run([G_step], feed_dict={z: z_ipt})

        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size2, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt})
            writer.add_summary(summary, it)

    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
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
        list_of_generators = [images_form_g1, images_form_g2]  # used for sampling images
        list_of_names = ['g1-it%d.jpg'%total_it,'g2-it%d.jpg'%total_it]
        rows = 10
        columns = 10
        sample_imgs = sess.run(list_of_generators, feed_dict={z: np.random.normal(size=[rows*columns, z_dim])})
        save_dir = dir + "/sample_imgs"
        utils.mkdir(save_dir + '/')
        for imgs,name in zip(sample_imgs,list_of_names):
            my_utils.saveSampleImgs(imgs=imgs, full_path=save_dir+"/"+name, row=rows,column=columns)
    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
