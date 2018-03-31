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

#apply mad-gan in the latent space of vae
#idea the latent manifold is simpler than data manifold
#it's easier to learn each mode by one head of mad-gan
""" param """
epoch = 100
batch_size = 64
lr = 0.0002
z_dim = 100
beta = 1 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="mad-gan-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)
tf.set_random_seed(1234)
# restore = False
# ckpt_dir =

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0, 8])

""" graphs """
generator = models.ss_generator
discriminator = models.multi_c_discriminator2
optimizer = tf.train.AdamOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

z1 = models.transform_(z, reuse=False, name="cluster1")
z2 = models.transform_(z, reuse=False, name="cluster2")

# generator
fake = generator(z1, reuse=False, name="g1")
fake2 = generator(z2, name="g1")

# discriminator
r_logit = discriminator(real, reuse=False, name="d1")
f1_logit = discriminator(fake, name="d1")
f2_logit = discriminator(fake2, name="d1")

#supplement discriminator
# f1_c = discriminator(fake, reuse=False, name="d2")
# f2_c = discriminator(fake2, name="d2")

#discriminator loss
#since the shape of r_logit is (?,3) to create zeros of shape (?,1)
#i use batch size
# shape = tf.shape(r_logit)
onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=3)
onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=3)
onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=3)
# print(sess.run(tf.shape(onehot_labels_zero)))
D_loss_real = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=r_logit, onehot_labels=onehot_labels_zero))
D_loss_fake1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_one))
D_loss_fake2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_two))
d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2

#generator loss
g1_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_zero))
g2_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_zero))
# g1_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
# g2_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
g_loss = g1_loss + g2_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('d1')]
# d2_var = [var for var in T_vars if var.name.startswith('d2')]
g1_var = [var for var in T_vars if var.name.startswith('g1')]
# g2_var = [var for var in T_vars if var.name.startswith('g2')]
gmm_var = [var for var in T_vars if var.name.startswith('cluster')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
# d2_step = optimizer(learning_rate=lr).minimize(d2_loss, var_list=d2_var)
# g_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g1_loss, var_list=g1_var)
# g2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g2_loss, var_list=g2_var)
G_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g1_var + gmm_var)
""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('C1_loss', g1_loss)
tf.summary.scalar('C2_loss', g2_loss)
# tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
# tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
images_form_g1 = generator(z1, name="g1", training= False)
images_form_g2 = generator(z2, name="g1", training= False)
tf.summary.image('C1_images', images_form_g1, 12)
tf.summary.image('C2_images', images_form_g2, 12)
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
print("lr:"+str(lr))
print("batch size:"+str(batch_size))
# print("lr:"+str(lr))
def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):

        for i in range(n_critic):
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})

        # train G
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # _ = sess.run([g_step], feed_dict={z: z_ipt})

            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            # _ = sess.run([g2_step], feed_dict={z: z_ipt})
            _ = sess.run([G_step], feed_dict={z: z_ipt})
        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
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
