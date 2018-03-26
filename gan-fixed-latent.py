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
# batch_size2 = 64
lr = 0.0002
z_dim = 2
# l_dim = 1
# beta = 1 #diversity hyper param
# clip = 0.01
n_critic = 1 #
n_generator = 1
gan_type="gan-fixed-latent"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0,4,5,7])

""" graphs """
generator = models.ss_generator
discriminator = models.ss_discriminator
optimizer = tf.train.AdamOptimizer

#sample from gmm
k = 4
mus = [[0.5, 0.5],[-0.5, 0.5],[-0.5,-0.5],[0.5, -0.5]]
cov = [[1 ,0],[0, 1]]

def sample_from_gmm(size, k):
    z = np.random.multivariate_normal(mean=mus[0], cov=cov, size=size//k)
    for i in range(1,k-1):
        temp = np.random.multivariate_normal(mean=mus[i], cov=cov, size=size // k)
        z = np.concatenate((z,temp))
        np.random.shuffle(z)
    return z

def sample_from_gaussian(mean, cov, size):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=size)

# def create_z_feedDict():


# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
z1 = tf.placeholder(tf.float32, shape=[None, z_dim])
z2 = tf.placeholder(tf.float32, shape=[None, z_dim])
z3 = tf.placeholder(tf.float32, shape=[None, z_dim])
z4 = tf.placeholder(tf.float32, shape=[None, z_dim])

# generator
fake = generator(z, reuse=False)

# discriminator
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)


#supplement discriminator
# f1_c = discriminator(fake, reuse=False, name="d2")
# f2_c = discriminator(fake2, name="d2")

#discriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
# D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
d_loss = D_loss_real + D_loss_fake
# d_loss = D_loss_real + D_loss_fake

#supplement discriminator loss
# D2_loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_c, labels=tf.zeros_like(f1_c)))
# D2_loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_c, labels=tf.ones_like(f2_c)))
# d2_loss = D2_loss_f1 + D2_loss_f2

#generator loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))
# g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))

# g1_loss += D2_loss_f1
# g2_loss +=  D2_loss_f2

# g_loss = g1_loss + g2_loss


# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
# g_var = [var for var in T_vars if var.name.startswith('g1')]
# gmm_var = [var for var in T_vars if var.name.startswith('cluster')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
# d2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d2_loss, var_list=d2_var)
g_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)
# g2_step = optimizer(learning_rate=lr).minimize(g2_loss, var_list=g_var)
""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session()
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)

images_form_g = generator(z, training= False)
images_form_c1 = generator(z1, training= False)
images_form_c2 = generator(z2, training= False)
images_form_c3 = generator(z3, training= False)
images_form_c4 = generator(z4, training= False)

tf.summary.image('Generator_image', images_form_g, 12)
tf.summary.image('Generator_image_from_c1', images_form_c1, 12)
tf.summary.image('Generator_image_from_c2', images_form_c2, 12)
tf.summary.image('Generator_image_from_c3', images_form_c3, 12)
tf.summary.image('Generator_image_from_c4', images_form_c4, 12)

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

def sample_once(it):
    rows = 10
    columns = 10
    feed = {z: sample_from_gmm(batch_size, k),
            z1: sample_from_gaussian(mean=mus[0],cov=cov, size=batch_size),
            z2: sample_from_gaussian(mean=mus[1],cov=cov, size=batch_size),
            z3: sample_from_gaussian(mean=mus[2],cov=cov, size=batch_size),
            z4: sample_from_gaussian(mean=mus[3],cov=cov, size=batch_size)}
    list_of_generators = [images_form_g, images_form_c1, images_form_c2, images_form_c3,
                          images_form_c4]  # used for sampling images
    list_of_names = ['it%d-g.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it,
                     'it%d-c3.jpg' % it, 'it%d-c4.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)


def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        for i in range(n_critic):
            real_ipt = data_pool.batch('img')
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            z_ipt = sample_from_gmm(batch_size,k)
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
            # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt, label: label_zero})


        # train G
        for j in range(n_generator):
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            z_ipt = sample_from_gmm(batch_size,k)
            _ = sess.run([g_step], feed_dict={z: z_ipt})

        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            z_ipt = sample_from_gmm(batch_size, k)
            z_ipt1 = sample_from_gaussian(mean=mus[0],cov=cov, size=batch_size)
            z_ipt2 = sample_from_gaussian(mean=mus[1], cov=cov, size=batch_size)
            z_ipt3 = sample_from_gaussian(mean=mus[2], cov=cov, size=batch_size)
            z_ipt4 = sample_from_gaussian(mean=mus[3], cov=cov, size=batch_size)
            summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt, z1:z_ipt1, z2:z_ipt2, z3:z_ipt3, z4:z_ipt4})
            # summary = sess.run(merged, feed_dict={real: real_ipt, z: z_ipt, label: label_zero})
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
