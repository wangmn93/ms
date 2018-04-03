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
from functools import partial
""" param """
epoch = 50
batch_size = 64
# batch_size2 = 64
lr = 0.0002
z_dim = 100
beta = 1 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="experiment"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# np.random.seed(0)
# tf.set_random_seed(1234)

# restore = False
# ckpt_dir =

''' data '''
# keep = range(10)
keep = [1,3,5,7]
data_pool = my_utils.getMNISTDatapool(batch_size, keep=keep)

""" graphs """
heads = 4
port_to_learn = [0,1,2,3] #max number should be less than len(keep)
classifier = partial(models.cat_conv_discriminator,out_dim=len(keep))
generator = partial(models.ss_generator_m, heads=heads)
discriminator = models.ss_discriminator

optimizer = tf.train.AdamOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# generator
fake_sets = generator(z, reuse=False)
# fake = tf.concat([fake,fake2],axis=0)
# fake2 = generator(z, reuse=False, name="g2")

# discriminator
r_logit = discriminator(real, reuse=False)
f_logit_set = []
f_c_set = []


for i in range(len(fake_sets)):
    f_logit_set.append(discriminator(fake_sets[i]))
    if i == 0:
        # f_c_set.append(models.cnn_classifier_2(x=fake_sets[i],reuse=False, keep_prob=1.))
        p, _ = models.cnn_classifier_2(x=fake_sets[i],reuse=False, keep_prob=1., out_dim=len(keep))
        # p,_ = classifier(fake_sets[i], name='classifier', reuse=False)
        f_c_set.append(p)
    else:
        # f_c_set.append(models.cnn_classifier_2(x=fake_sets[i], keep_prob=1.))
        p, _ = models.cnn_classifier_2(x=fake_sets[i], keep_prob=1., out_dim=len(keep))
        # p, _ = classifier(fake_sets[i], name='classifier')
        f_c_set.append(p)
#supplement discriminator


#discriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
D_loss_fake = 0
g_loss = 0
for f_logit in f_logit_set:
    D_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
    g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))
# D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
d_loss = D_loss_real + D_loss_fake
# d_loss = D_loss_real + D_loss_fake1

def mar_entropy2(ys):
    y2 =0
    for y in ys:
        y1 = tf.reduce_mean(y, axis=0)
        y2 += tf.reduce_sum(-y1 * tf.log(y1))
    return y2

#supplement discriminator loss
def compute_c_loss(logits,number_to_learn):
    loss = 0
    for i in range(len(logits)):
        onehot_labels = tf.one_hot(indices=tf.cast(tf.scalar_mul(number_to_learn[i], tf.ones(batch_size)), tf.int32), depth=len(keep))
        loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[i], labels=onehot_labels))
    return loss

c_loss = compute_c_loss(f_c_set, port_to_learn)
# g_loss += c_loss
g_loss += 0.5*c_loss
# g_loss += beta*c_loss + 0.1*mar_entropy2(f_c_set)

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g1_var = [var for var in T_vars if var.name.startswith('generator')]
# g2_var = [var for var in T_vars if var.name.startswith('g2')]
c_var = tf.global_variables('classifier')

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
G_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g1_var)
# c_step = optimizer(learning_rate=lr, beta1=0.5).minimize(c_loss, var_list=g1_var)
# c_aug_step = optimizer(learning_rate=lr, beta1=0.5).minimize(c_aug_loss, var_list=g1_var)
""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
c_saver = tf.train.Saver(var_list=c_var)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('C_loss', c_loss)
# tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
# images_form_g1, images_form_g2 = generator(z, training= False)
# image_sets = [images_form_g1, images_form_g2]
image_sets = generator(z, training= False)
for img_set in image_sets:
    tf.summary.image('G_images', img_set, 12)
# images_form_g2 = generator(z, name="g2", training= False)
# tf.summary.image('G1_images', images_form_g1, 12)
# tf.summary.image('G2_images', images_form_g2, 12)
merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
# ckpt_dir = './checkpoints/mnist_wgan'
# utils.mkdir(ckpt_dir + '/')
# if not utils.load_checkpoint(ckpt_dir, sess):
sess.run(tf.global_variables_initializer())
# c_saver.restore(sess, "results/cnn_classifier/checkpoint/model.ckpt") #0.96
# c_saver.restore(sess, "results/cnn_classifier-med-train/checkpoint/model.ckpt") #0.8707
# c_saver.restore(sess,"results/cnn_classifier-under-train/checkpoint/model.ckpt") #0.8427
# c_saver.restore(sess, "results/cat-gan-20180401-160253/checkpoint/model.ckpt") #all numbers
# c_saver.restore(sess,"results/cat-gan-20180401-203608/checkpoint/model.ckpt") #1,3,6
# c_saver.restore(sess,"results/cat-gan-20180401-210244/checkpoint/model.ckpt") #1,3,4,6
# c_saver.restore(sess,"results/cat-gan-20180402-083936/checkpoint/model.ckpt") #3,5 conv dis
# c_saver.restore(sess, "results/cat-gan-20180402-101436/checkpoint/model.ckpt") #4,9 conv dis
# c_saver.restore(sess, "results/cat-gan-20180402-110332/checkpoint/model.ckpt") #0,1,2,3,4,5
# c_saver.restore(sess, "results/cat-gan-20180402-140657/checkpoint/model.ckpt") #1,3,5,7
c_saver.restore(sess, "results/cat-gan-20180402-170735/checkpoint/model.ckpt") #1,3,5,7 cnn arch from supervised

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

def sample_once(it):
    rows = 10
    columns = 10
    list_of_generators = image_sets  # used for sampling images
    list_of_names = []
    for i in range(len(image_sets)):
        list_of_names.append('it%d-g%d.jpg'%(it, i))
    # list_of_names = ['g1-it%d.jpg' % it, 'g2-it%d.jpg' % it]
    feeds = {z: np.random.normal(size=[rows * columns, z_dim])}
    my_utils.sample_and_save(sess, list_of_generators, list_of_names, feeds, dir+ "/sample_imgs", rows, columns, normalize=True)
    print('save sample images')

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):

        for i in range(n_critic):
            real_ipt = (data_pool.batch('img')+1.)/2.
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _= sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
            # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})

        # train G
        for j in range(n_generator):
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # _, _ = sess.run([g_step,g2_step], feed_dict={z: z_ipt})
            _ = sess.run([G_step], feed_dict={z: z_ipt})
            # if it>3000:
            #     if it<10000:
            #     _ = sess.run([c_step], feed_dict={z: z_ipt})
            #     else:
            #         _ = sess.run([c_aug_step], feed_dict={z: z_ipt})
        if it>0 and it%5000 == 0:
            sample_once(it)#sample every 5000 it

        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1.)/2.
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt})
            writer.add_summary(summary, it)

    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        sample_once(it_offset + max_it)
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
