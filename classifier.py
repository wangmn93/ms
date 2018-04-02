from __future__ import print_function
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import



import tensorflow as tf
import models_mnist

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def cnn_classifier(x_image, keep_prob, name="classifier", reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        # Convolutional layer 1
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Convolutional layer 2
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Fully connected layer 2 (Output layer)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
        return y


def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Input layer
    x  = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
    keep_prob  = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    y = models_mnist.cnn_classifier_2(x=x_image, name='classifier',keep_prob=keep_prob, reuse=False)#create model

    # Evaluation functions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Training algorithm
    c_var = tf.trainable_variables('classifier')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=c_var)

    # Saver

    c_saver = tf.train.Saver(var_list=c_var)
    saver = tf.train.Saver()

    # Training steps
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      # c_saver.restore(sess, "results/cnn_classifier/checkpoint/model.ckpt")
      max_steps = 120
      for step in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(50) # 0 ~ 1
        # batch_xs = batch_xs*2-1 # -1 ~ 1, bad results
        if (step % 10) == 0:
          print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
      print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
      save_path = saver.save(sess, "results/cnn_classifier-med-train/checkpoint/model.ckpt")
      # print('Test Acc', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
      print("Model saved in path: %s" % save_path)
      print(" [*] Close main session!")
      sess.close()



main()

# import utils
# import traceback
# import numpy as np
# import tensorflow as tf
# import models_mnist as models
# import datetime
# import my_utils
# # from classifier import cnn_classifier
#
#
# """ param """
# epoch = 200
# batch_size = 128
# batch_size2 = 64
# lr = 0.0002
# z_dim = 100
# beta = 1 #diversity hyper param
# # clip = 0.01
# n_critic = 1 #
# n_generator = 1
# gan_type="experiment"
# dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
# np.random.seed(0)
# tf.set_random_seed(1234)
#
# # restore = False
# # ckpt_dir =
#
# ''' data '''
# data_pool = my_utils.getMNISTDatapool(batch_size, keep=[0, 1,8])
#
# """ graphs """
# generator = models.ss_generator_2
# discriminator = models.ss_discriminator
# optimizer = tf.train.AdamOptimizer
#
#
# # inputs
# real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# z = tf.placeholder(tf.float32, shape=[None, z_dim])
#
#
# # generator
# fake = generator(z, reuse=False, name="g1")
# fake2 = generator(z, reuse=False, name="g2")
#
# # discriminator
# r_logit = discriminator(real, reuse=False, name="d1")
# f1_logit = discriminator(fake, name="d1")
# f2_logit = discriminator(fake2, name="d1")
#
# #supplement discriminator
# f1_c = cnn_classifier(x_image=fake,keep_prob=1., reuse=False)#create model
# f2_c = cnn_classifier(x_image=fake2, keep_prob=1.)#create model
# # f1_c = discriminator(fake, reuse=False, name="d2")
# # f2_c = discriminator(fake2, name="d2")
#
# #discriminator loss
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
# D_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.zeros_like(f1_logit)))
# D_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.zeros_like(f2_logit)))
# d_loss = D_loss_real + D_loss_fake1 + D_loss_fake2
# # d_loss = D_loss_real + D_loss_fake1
#
# #supplement discriminator loss
# onehot_labels_zero = tf.one_hot(indices=tf.zeros(batch_size, tf.int32), depth=10)
# onehot_labels_one = tf.one_hot(indices=tf.ones(batch_size, tf.int32), depth=10)
# D2_loss_f1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f1_c, labels=onehot_labels_zero))
# D2_loss_f2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f2_c, labels=onehot_labels_one))
# # d2_loss = D2_loss_f1 + D2_loss_f2
#
# #generator loss
# g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f1_logit, labels=tf.ones_like(f1_logit)))
# g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f2_logit, labels=tf.ones_like(f2_logit)))
# g1_loss += beta*D2_loss_f1
# g2_loss += beta*D2_loss_f2
# g_loss = g1_loss + g2_loss
#
# # trainable variables for each network
# T_vars = tf.trainable_variables()
# # G_vars = tf.global_variables()
# d_var = [var for var in T_vars if var.name.startswith('d1')]
# g1_var = [var for var in T_vars if var.name.startswith('g1')]
# g2_var = [var for var in T_vars if var.name.startswith('g2')]
# c_var = [var for var in T_vars if var.name.startswith('classifier')]
#
# # optims
# global_step = tf.Variable(0, name='global_step',trainable=False)
# d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var, global_step=global_step)
# # d2_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d2_loss, var_list=d2_var)
# # g_step = optimizer(learning_rate=lr).minimize(g1_loss, var_list=g1_var)
# # g2_step = optimizer(learning_rate=lr).minimize(g2_loss, var_list=g2_var)
# G_step = optimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g1_var + g2_var)
# """ train """
# ''' init '''
# # session
# sess = tf.InteractiveSession()
#
# # saver
# saver = tf.train.Saver(max_to_keep=5)
# c_saver = tf.train.Saver(var_list=c_var)
# # summary writer
# # Send summary statistics to TensorBoard
# tf.summary.scalar('G1_loss', g1_loss)
# tf.summary.scalar('G2_loss', g2_loss)
# tf.summary.scalar('G_loss', g_loss)
# tf.summary.scalar('Discriminator_loss', d_loss)
# # tf.summary.scalar('Supplement_Discriminator_loss', d2_loss)
# images_form_g1 = generator(z, name="g1", training= False)
# images_form_g2 = generator(z, name="g2", training= False)
# tf.summary.image('G1_images', images_form_g1, 12)
# tf.summary.image('G2_images', images_form_g2, 12)
# merged = tf.summary.merge_all()
# logdir = dir+"/tensorboard"
# writer = tf.summary.FileWriter(logdir, sess.graph)
# print('Tensorboard dir: '+logdir)
#
# ''' initialization '''
# # ckpt_dir = './checkpoints/mnist_wgan'
# # utils.mkdir(ckpt_dir + '/')
# # if not utils.load_checkpoint(ckpt_dir, sess):
# sess.run(tf.global_variables_initializer())
# c_saver.restore(sess, "results/cnn_classifier/checkpoint/model.ckpt")
#
# ''' train '''
# batch_epoch = len(data_pool) // (batch_size * n_critic)
# max_it = epoch * batch_epoch
# def training(max_it, it_offset):
#     print("Max iteration: " + str(max_it))
#     total_it = it_offset + max_it
#     for it in range(it_offset, it_offset + max_it):
#
#         for i in range(n_critic):
#             real_ipt = (data_pool.batch('img')+1.)/2.
#             z_ipt = np.random.normal(size=[batch_size2, z_dim])
#             _, _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
#             # _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
#
#         # train G
#         for j in range(n_generator):
#             z_ipt = np.random.normal(size=[batch_size2, z_dim])
#             # _, _ = sess.run([g_step,g2_step], feed_dict={z: z_ipt})
#             _ = sess.run([G_step], feed_dict={z: z_ipt})
#
#         if it%10 == 0 :
#             real_ipt = (data_pool.batch('img')+1.)/2.
#             z_ipt = np.random.normal(size=[batch_size2, z_dim])
#             summary = sess.run(merged, feed_dict={real: real_ipt,z: z_ipt})
#             writer.add_summary(summary, it)
#
#     var = raw_input("Continue training for %d iterations?" % max_it)
#     if var.lower() == 'y':
#         training(max_it, it_offset + max_it)
#
# total_it = 0
# try:
#     training(max_it,0)
#     total_it = sess.run(global_step)
#     print("Total iterations: "+str(total_it))
# except Exception, e:
#     traceback.print_exc()
# finally:
#     var = raw_input("Save sample images?")
#     if var.lower() == 'y':
#         list_of_generators = [images_form_g1, images_form_g2]  # used for sampling images
#         list_of_names = ['g1-it%d.jpg'%total_it,'g2-it%d.jpg'%total_it]
#         rows = 10
#         columns = 10
#         sample_imgs = sess.run(list_of_generators, feed_dict={z: np.random.normal(size=[rows*columns, z_dim])})
#         save_dir = dir + "/sample_imgs"
#         utils.mkdir(save_dir + '/')
#         for imgs,name in zip(sample_imgs,list_of_names):
#             my_utils.saveSampleImgs(imgs=imgs, full_path=save_dir+"/"+name, row=rows,column=columns)
#     # save checkpoint
#     save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
#     print("Model saved in path: %s" % save_path)
#     print(" [*] Close main session!")
#     sess.close()
