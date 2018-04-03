from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from functools import partial

#fix the shape of latent space to a 2d gmm
#use supplement discriminator/classifier to encourage each mode of gmm to embed different number
""" param """
epoch = 100
batch_size = 100
lr = 0.0002
z_dim = 2
n_critic = 1 #
n_generator = 1
gan_type="mgan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, keep=[1,3,5,7])

""" graphs """
generator = partial(models.ss_generator_m, heads=4)
discriminator = models.ss_discriminator
classifier = models.multi_c_discriminator3 #4 heads
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# generator
fake_set = generator(z, reuse=False)
fake_1 = fake_set[0]
fake_2 = fake_set[1]
fake_3 = fake_set[2]
fake_4 = fake_set[3]
fake = tf.concat([fake_1, fake_2, fake_3, fake_4], 0)

# discriminator
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)

#discriminator loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
d_loss = d_loss_real + d_loss_fake

#marginal entropy
def mar_entropy(y):
    # y1 = F.sum(y, axis=0) / batchsize
    # y2 = F.sum(-y1 * F.log(y1))
    # return y2
    y1 = tf.reduce_mean(y,axis=0)
    y2=tf.reduce_sum(-y1*tf.log(y1))
    return y2

#conditional entropy
def cond_entropy(y):
    # y1 = -y * F.log(y)
    # y2 = F.sum(y1) / batchsize
    # return y2
    y1=-y*tf.log(y)
    y2 = tf.reduce_sum(y1)/batch_size
    return y2

#supplement classifier
c_1 = classifier(fake_1, reuse=False, name="supplement_c")
c_2 = classifier(fake_2, name="supplement_c")
c_3 = classifier(fake_3, name="supplement_c")
c_4 = classifier(fake_4, name="supplement_c")

#classifier loss
onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=4)
onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=4)
onehot_labels_two = tf.one_hot(indices=tf.cast(tf.scalar_mul(2,tf.ones(batch_size)), tf.int32), depth=4)
onehot_labels_three = tf.one_hot(indices=tf.cast(tf.scalar_mul(3,tf.ones(batch_size)), tf.int32), depth=4)

c_loss_1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_1, onehot_labels=onehot_labels_zero))
c_loss_2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_2, onehot_labels=onehot_labels_one))
c_loss_3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_3, onehot_labels=onehot_labels_two))
c_loss_4 = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=c_4, onehot_labels=onehot_labels_three))

r_c = classifier(real, name="supplement_c")
real_y = tf.nn.softmax(r_c)
fake_y = tf.nn.softmax(tf.concat([c_1,c_2,c_3,c_4],axis=0))
c_loss = c_loss_1 + c_loss_2 + c_loss_3 + c_loss_4 -1 * (mar_entropy(real_y) - cond_entropy(real_y) + cond_entropy(fake_y))

#generator loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))

# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
c_var = [var for var in T_vars if var.name.startswith('supplement_c')]


# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr).minimize(d_loss, var_list=d_var, global_step=global_step)
c_step = optimizer(learning_rate=lr).minimize(c_loss, var_list=c_var+g_var)
g_step = optimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)

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
image_set = generator(z, training= False)

images_form_c1 = image_set[0]
images_form_c2 = image_set[1]
images_form_c3 = image_set[2]
images_form_c4 = image_set[3]

# tf.summary.image('Generator_image', images_form_g, 12)
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
    feed = {z: np.random.normal(size=[rows*columns, z_dim])}
    list_of_generators = [ images_form_c1, images_form_c2, images_form_c3,
                          images_form_c4]  # used for sampling images
    list_of_names = ['it%d-c1.jpg' % it, 'it%d-c2.jpg' % it,
                     'it%d-c3.jpg' % it, 'it%d-c4.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    for it in range(it_offset, it_offset + max_it):
        for i in range(n_critic):
            real_ipt = (data_pool.batch('img')+1.)/2.
            _ = sess.run([d_step],feed_dict={real: real_ipt, z: np.random.normal(size=[batch_size, z_dim])})


        # train G and C
        for j in range(n_generator):
            real_ipt = (data_pool.batch('img') + 1.) / 2.
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            _ = sess.run([g_step],feed_dict={z: z_ipt})
            _ = sess.run([c_step], feed_dict={z: z_ipt, real:real_ipt})

        if it%10 == 0 :
            real_ipt = (data_pool.batch('img')+1.)/2.
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            summary = sess.run(merged, feed_dict={real: real_ipt, z: z_ipt})
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
