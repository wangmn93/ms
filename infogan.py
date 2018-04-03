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
batch_size = 64
lr = 0.0002
z_dim = 100
cat_dim = 3
# beta = 1 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="infogan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# np.random.seed(0)
# tf.set_random_seed(1234)
# restore = False
# ckpt_dir =

one_hot_labels = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
p = [1/3.]*3

def sample_from_categorical(size, k=3):
    return [one_hot_labels[index] for index in np.random.choice(range(k), size=size, p=p)]

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[1,3,5])

""" graphs """
generator = models.generator
discriminator = models.multi_c_discriminator3
optimizer = tf.train.AdamOptimizer


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
cat = tf.placeholder(tf.float32, shape=[None, cat_dim])
cat_1 = tf.placeholder(tf.float32, shape=[None, cat_dim])
cat_2 = tf.placeholder(tf.float32, shape=[None, cat_dim])
cat_3 = tf.placeholder(tf.float32, shape=[None, cat_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])


# generator
z_cat = tf.concat([z,cat],axis=1)
z_cat_1 = tf.concat([z,cat_1],axis=1)
z_cat_2 = tf.concat([z,cat_2],axis=1)
z_cat_3 = tf.concat([z,cat_3],axis=1)
print(z_cat.shape)
fake = generator(z_cat, reuse=False)

# discriminator
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)

#D loss
print(f_logit[:,3].shape)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit[:3], labels=tf.ones_like(r_logit)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit[:3], labels=tf.zeros_like(f_logit)))
d_loss = D_loss_real + D_loss_fake

#generator loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit[:3], labels=tf.ones_like(f_logit)))

#q_loss
print(f_logit[:,0:3].shape)
cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(f_logit[:,0:3] + 1e-8) * cat, 1))
ent = tf.reduce_mean(-tf.reduce_sum(tf.log(cat + 1e-8) * cat, 1))
q_loss = cond_ent + ent

#supplement discriminator
# f1_c = discriminator(fake, reuse=False, name="d2")
# f2_c = discriminator(fake2, name="d2")

#discriminator loss
#since the shape of r_logit is (?,3) to create zeros of shape (?,1)
#i use batch size
# shape = tf.shape(r_logit)
# onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=3)
# onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=3)
# onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=3)

# onehot_labels_three = tf.one_hot(indices=tf.cast(tf.scalar_mul(3,tf.ones(batch_size)), tf.int32), depth=4)
# # print(sess.run(tf.shape(onehot_labels_zero)))
# D_loss_real = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=r_logit, onehot_labels=onehot_labels_zero))
# D_loss_fake1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_one))
# D_loss_fake2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_two))
# D_loss_fake3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f3_logit, onehot_labels=onehot_labels_three))
# d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2 + D_loss_fake3
#
# #generator loss
# g1_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f1_logit, onehot_labels=onehot_labels_zero))
# g2_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f2_logit, onehot_labels=onehot_labels_zero))
# g3_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f3_logit, onehot_labels=onehot_labels_zero))
#
# # g1_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
# # g2_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
# g_loss = g1_loss + g2_loss + g3_loss # + D_loss_fake1 + D_loss_fake2 + D_loss_fake3

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]


# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
G_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)
q_step = optimizer(learning_rate=lr, beta1=0.5).minimize(q_loss, var_list=g_var+d_var)
""" train """
''' init '''
# session
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('q_loss', q_loss)
# tf.summary.scalar('G_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
# tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
images_form_g1 = generator(z_cat_1, training= False)
images_form_g2 = generator(z_cat_2, training= False)
images_form_g3 = generator(z_cat_3, training= False)
tf.summary.image('G1_images', images_form_g1, 12)
tf.summary.image('G2_images', images_form_g2, 12)
tf.summary.image('G3_images', images_form_g3, 12)

merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

def sample_once(it):
    list_of_generators = [images_form_g1, images_form_g2, images_form_g3]  # used for sampling images
    list_of_names = ['g1-it%d.jpg' % it, 'g2-it%d.jpg' % it, 'g3-it%d.jpg' % it]
    rows = 10
    columns = 10

    sample_imgs = sess.run(list_of_generators, feed_dict={z: np.random.normal(size=[rows * columns, z_dim]),
                                                          cat_1: [one_hot_labels[0] for _ in range(rows * columns)],
                                                          cat_2: [one_hot_labels[1] for _ in range(rows * columns)],
                                                          cat_3: [one_hot_labels[2] for _ in range(rows * columns)]})
    save_dir = dir + "/sample_imgs"
    utils.mkdir(save_dir + '/')
    for imgs, name in zip(sample_imgs, list_of_names):
        my_utils.saveSampleImgs(imgs=imgs, full_path=save_dir + "/" + name, row=rows, column=columns)
        # save checkpoint

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


        real_ipt = data_pool.batch('img')
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        c_ipt = sample_from_categorical(batch_size)
        _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt, cat:c_ipt})

        # train G

        z_ipt = np.random.normal(size=[batch_size, z_dim])
        c_ipt = sample_from_categorical(batch_size)
        _ = sess.run([G_step], feed_dict={z: z_ipt, cat:c_ipt})

        #train Q
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        c_ipt = sample_from_categorical(batch_size)
        _ = sess.run([q_step], feed_dict={z: z_ipt, cat: c_ipt})

        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            c_ipt = sample_from_categorical(batch_size)
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt,
                                                  cat:c_ipt,
                                                  cat_1:[one_hot_labels[0] for _ in range(batch_size)],
                                                    cat_2:[one_hot_labels[1] for _ in range(batch_size)],
                                                    cat_3:[one_hot_labels[2] for _ in range(batch_size)]})
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

    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
