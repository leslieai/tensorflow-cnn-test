import tensorflow as tf
import os
import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1495430183/checkpoints",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
FLAGS = tf.flags.FLAGS


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10],name='label_y')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def conv2d(x, W,name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)


def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME',name=name)


# First Convolutional Layer
with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32],'W_conv1')
    b_conv1 = bias_variable([32],'b_conv1')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1_re = tf.nn.relu(conv2d(x_image, W_conv1,'h_conv1') + b_conv1,name='h_conv1_re')
    h_pool1 = max_pool_2x2(h_conv1_re,'h_pool1')

# Second Convolutional Layer
with tf.name_scope("conv2"):
    W_conv2 = weight_variable([5, 5, 32, 64],'W_conv2')
    b_conv2 = bias_variable([64],'b_conv2')
    h_conv2_re = tf.nn.relu(conv2d(h_pool1, W_conv2,'h_conv2') + b_conv2,name='h_conv2_re')
    h_pool2 = max_pool_2x2(h_conv2_re,'h_pool2')

# Densely Connected Layer
with tf.name_scope("Densely"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024],'W_fc1')
    b_fc1 = bias_variable([1024],'b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='h_fc1')

# Dropout
with tf.name_scope("dropout"):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='h_fc1_drop')

# Readout Layer
with tf.name_scope("output"):
    W_fc2 = weight_variable([1024, 10],'W_fc2')
    b_fc2 = bias_variable([10],'W_fc2')
    y_conv = tf.nn.xw_plus_b(h_fc1_drop,W_fc2,b_fc2,name='y_conv')
    y_eval = tf.nn.xw_plus_b(h_fc1,W_fc2,b_fc2,name='y_eval')  # evaluation without dropout

with tf.name_scope("cost"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv),name='loss')
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='acc')


def train():
    sess = tf.Session()
    with sess.as_default():
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # # Dev summaries
        # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())

        def train_step(feed_dict):
            """
            A single training step
            """
            _, step, summaries, loss, acc = sess.run(
                [train_op, global_step, train_summary_op, cross_entropy, accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            train_summary_writer.add_summary(summaries, step)

        for i in range(300):
            batch = mnist.train.next_batch(50)
            train_step({
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            current_step = tf.train.global_step(sess, global_step)
            # if i % 100 == 0:
            #     train_accuracy = accuracy.eval(feed_dict={
            #         x: batch[0], y_: batch[1], keep_prob: 1.0})
            #     print("step %d, training accuracy %g" % (i, train_accuracy))
            # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


            # if current_step % FLAGS.checkpoint_every == 0:
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                # print("test accuracy %g" % accuracy.eval(feed_dict={
                #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def eval():
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            xx = graph.get_operation_by_name("input/Placeholder").outputs[0]
            kk = graph.get_operation_by_name("input/Placeholder_2").outputs[0]
            y_eval = graph.get_operation_by_name("output/y_eval").outputs[0]
            y_hat=sess.run(y_eval, {xx: mnist.test.images[:1,:], kk: 1.0})
            print(y_hat)
            print(mnist.test.labels[:1, :])


# print(mnist.test.images[:1,:])
# print(mnist.test.labels[:1,:])
def main():
    train()
if __name__ == '__main__':
    # train()
    eval()
    pass


