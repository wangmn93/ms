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
batch_size = 64
lr = 0.0002
z_dim = 2
n_critic = 1 #
n_generator = 1
gan_type="gan-v-gmm"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0,4,5,7])

""" graphs """
generator = models.generator
discriminator = models.ss_discriminator
classifier = models.multi_c_discriminator3
optimizer = tf.train.AdamOptimizer

#sample from gmm
# with tf.variable_scope("gmm", reuse=False):
#     mu_1 = tf.get_variable("mean1", [z_dim], initializer=tf.constant_initializer(0))
#     mu_2 = tf.get_variable("mean2", [z_dim], initializer=tf.constant_initializer(0))
#     mu_3 = tf.get_variable("mean3", [z_dim], initializer=tf.constant_initializer(0))
#     mu_4 = tf.get_variable("mean4", [z_dim], initializer=tf.constant_initializer(0))
#     log_sigma_sq1 = tf.get_variable("log_sigma_sq1", [z_dim], initializer=tf.constant_initializer(0.001))
#     log_sigma_sq2 = tf.get_variable("log_sigma_sq2", [z_dim], initializer=tf.constant_initializer(0.001))
#     log_sigma_sq3 = tf.get_variable("log_sigma_sq3", [z_dim], initializer=tf.constant_initializer(0.001))
#     log_sigma_sq4 = tf.get_variable("log_sigma_sq4", [z_dim], initializer=tf.constant_initializer(0.001))
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
fake_1 = generator(z1, reuse=False)
fake_2 = generator(z2)
fake_3 = generator(z3)
fake_4 = generator(z4)

# discriminator
r_logit = discriminator(real, reuse=False)
f_logit_1 = discriminator(fake_1)
f_logit_2 = discriminator(fake_1)
f_logit_3 = discriminator(fake_1)
f_logit_4 = discriminator(fake_1)


#supplement classifier
c_1 = classifier(fake_1, reuse=False, name="supplement_c")
c_2 = classifier(fake_2, name="supplement_c")
c_3 = classifier(fake_2, name="supplement_c")
c_4 = classifier(fake_2, name="supplement_c")

#discriminator loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_1, labels=tf.zeros_like(f_logit_1)))
d_loss_fake_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_2, labels=tf.zeros_like(f_logit_2)))
d_loss_fake_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_3, labels=tf.zeros_like(f_logit_3)))
d_loss_fake_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_4, labels=tf.zeros_like(f_logit_4)))
d_loss_fake = d_loss_fake_1 + d_loss_fake_2 + d_loss_fake_3 + d_loss_fake_4

d_loss = d_loss_real + d_loss_fake


#classifier loss
onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=4)
onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=4)
onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=4)
onehot_labels_three = tf.one_hot(indices=tf.cast(tf.scalar_mul(3,tf.ones(batch_size)), tf.int32), depth=4)

c_loss_1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_1, onehot_labels=onehot_labels_zero))
c_loss_2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_2, onehot_labels=onehot_labels_one))
c_loss_3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_3, onehot_labels=onehot_labels_two))
c_loss_4 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_4, onehot_labels=onehot_labels_three))
c_loss = c_loss_1 + c_loss_2 + c_loss_3 + c_loss_4

#generator loss
g_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_1, labels=tf.ones_like(f_logit_1)))
g_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_2, labels=tf.ones_like(f_logit_2)))
g_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_3, labels=tf.ones_like(f_logit_3)))
g_loss_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit_4, labels=tf.ones_like(f_logit_4)))

g_loss = g_loss_1 + g_loss_2 + g_loss_3 + g_loss_4 + c_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
c_var = [var for var in T_vars if var.name.startswith('supplement_c')]


# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
c_step = optimizer(learning_rate=lr, beta1=0.5).minimize(c_loss, var_list=c_var)
g_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)
# g2_step = optimizer(learning_rate=lr).minimize(g2_loss, var_list=g_var)
""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)
tf.summary.scalar('Classifier_loss', c_loss)

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

def create_feed(size, additional_items=None):
    z = {
        z1: sample_from_gaussian(mean=mus[0], cov=cov, size=size),
        z2: sample_from_gaussian(mean=mus[1], cov=cov, size=size),
        z3: sample_from_gaussian(mean=mus[2], cov=cov, size=size),
        z4: sample_from_gaussian(mean=mus[3], cov=cov, size=size)}
    if additional_items is not None:
        return dict(additional_items.items()+z.items())
    else:
        return z

def sample_once(it):
    rows = 10
    columns = 10
    feed = create_feed(size=rows*columns, additional_items={z: sample_from_gmm(size=rows*columns, k=k)})
    list_of_generators = [images_form_g, images_form_c1, images_form_c2, images_form_c3,
                          images_form_c4]  # used for sampling images
    list_of_names = ['it%d-g.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it,
                     'it%d-c3.jpg' % it, 'it%d-c4.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    for it in range(it_offset, it_offset + max_it):
        for i in range(n_critic):
            real_ipt = data_pool.batch('img')
            _ = sess.run([d_step],feed_dict=create_feed(size=batch_size, additional_items={real: real_ipt}))



        # train G and C
        for j in range(n_generator):
            _, _ = sess.run([g_step, c_step],feed_dict=create_feed(size=batch_size))

        if it%10 == 0 :
            real_ipt = data_pool.batch('img')
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            z_ipt = sample_from_gmm(batch_size, k)
            summary = sess.run(merged, feed_dict=create_feed(size=batch_size, additional_items={real: real_ipt,z: z_ipt}))
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

    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
