
import numpy as np
import tensorflow as tf


from matplotlib import pyplot as plt
import matplotlib.image as img
import pandas as pd
#
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
ii = input_data.read_data_sets('MNIST_data', one_hot=True)
batch = ii.train.next_batch(1)
# sess = tf.Session()
sess = tf.InteractiveSession()
#
#
#
W0=tf.Variable(tf.truncated_normal([200, 200, 3, 3], stddev=0.1))

#
# First Convolutional Layer

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# [batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])
# [filter_height, filter_width, in_channels, out_channels]
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))

# [batch, out_height, out_width,filter_height * filter_width * in_channels]
z1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
#
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(z1 + b1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# # Second Convolutional Layer
W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
z2=tf.nn.conv2d(h_pool1,W2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(z2 + b2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

#
#
# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




sess.run(tf.global_variables_initializer())

# print(sess.run(tf.shape(batch[0])))
# print(sess.run(tf.shape(x)))
# print(sess.run(tf.shape(W1)))
# print(sess.run(tf.shape(z1)))
# print(sess.run(tf.shape(h_conv1)))
# print(sess.run(tf.shape(h_pool1)))
# print(sess.run(tf.shape(z2)))
# print(sess.run(tf.shape(h_conv2)))
# print(sess.run(tf.shape(h_pool2)))
# print(sess.run(tf.shape(h_pool2_flat)))


# accuracy=sess.run(fetches=accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob: 0.5})


# print(sess.run(tf.cast([True], tf.float32)))
# print(sess.run(tf.argmax([1,2,3,4], 0)))
# print(accuracy)
# print(correct_prediction.shape)

# np.shape([1,1,1,1])
# Out[4]: (4,)
# np.shape([[1,1,1,1]])
# Out[6]: (1, 4)


# [  1 784]
# [ 1 28 28  1]x(input)  1 in_channels

# [ 5  5  1 32]W(filter)

# [filter_height, filter_width, in_channels, out_channels] 1 in_channels 32 output_channels
# [ 1 1 1 1]strides

# Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
#   horizontal and vertices strides, `strides = [1, stride, stride, 1]`

# padding: A `string` from: `"SAME", "VALID"`.
# [ 1 28 28 32]conv2d return  32 output_channels

# [  1 784] batch[0]
# [ 1 28 28  1] x
# [ 5  5  1 32] W1
# [ 1 28 28 32] conv1
# [ 1 28 28 32] relu
# [ 1 14 14 32] max_pooling
# [ 1 14 14 64] conv2
# [ 1 14 14 64] relu
# [ 1  7  7 64] max_pooling
# [   1 3136] flat
# (1,1024) h_fc1
# (1,10)y_conv



# dd=h_pool1.eval()
# # print(batch[0].shape)
# f, ax = plt.subplots(4, 8)
# ax=np.reshape(ax,[-1,32])
# for i in range(32):
#     bb = np.reshape(dd[:,:,:,i:i+1], [-1, 14, 14, 1])
#     # print(bb.shape)
#     ax[0][i].imshow(bb.squeeze(), cmap=plt.cm.gray)
#     ax[0][i].set_title('%d'%i)
# plt.show()

# #
file_name = './my_data/qqq.jpeg'
image = img.imread(file_name)
# tensor_image=tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32),0)
tensor_image=tf.expand_dims(tf.constant(image, dtype=tf.float32),0)
# tensor_image=tf.reshape(tf.convert_to_tensor(image, dtype=tf.float32),[1,320, 400,3])


z0=tf.nn.conv2d(tensor_image, W0, strides=[1, 1, 1, 1], padding='SAME')
kk=z0.eval(session=sess)
print(kk.shape)
plt.imshow(kk[:,:,:,1:2].squeeze())
plt.show()



# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train = pd.read_csv('./MNIST_data/mnist_train.csv', header = 0, dtype={'Age': np.float64})
# print(train)
#
# filename = './my_data/qqq.jpeg'
# image_f = tf.gfile.FastGFile(filename, 'rb').read()
# jpeg = tf.constant(image_f)
# print(jpeg)
# # Input node that doesn't lead anywhere.
# i_d=tf.image.decode_jpeg(jpeg, name='DecodeJpeg',channels=3)
# resized_image = tf.image.resize_images(i_d, [320,400])
# print(resized_image.eval())
# plt.imshow(resized_image.eval())
# plt.show()

aa = tf.constant(
    [  # 1th batch, first image that height is three, width is two ,channel is three
        [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],
        # 2th batch, second image
        [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],
    ])
print(aa.shape)
print(aa[1].eval(session=session))
