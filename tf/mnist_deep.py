import tensorflow as tf
import numpy as np
import pickle
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

NETWORK_CONFIG = {
	"Conv1Features":32,
	"Conv2Features":64,
	"FC1Features":1024,
}

def WeightVar(shape):

	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def BiasVar(shape):

	return tf.Variable(tf.constant(0.1, shape = shape))

def Conv2D(x, W):

	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def MaxPool(x, shape):

	return tf.nn.max_pool(x, ksize = [1, shape[0], shape[1], 1], strides = [1, shape[0], shape[1], 1], padding = "SAME")

def Main():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# ReLU(Wx + b)
	W_conv1 = WeightVar([5, 5, 1, NETWORK_CONFIG["Conv1Features"]])
	b_conv1 = BiasVar([NETWORK_CONFIG["Conv1Features"]])
	h_conv1 = tf.nn.relu(Conv2D(x_image, W_conv1) + b_conv1)

	# Pooling
	h_pool1 = MaxPool(h_conv1, [2, 2])

	# ReLU(Wx + b)
	W_conv2 = WeightVar([5, 5, NETWORK_CONFIG["Conv1Features"], NETWORK_CONFIG["Conv2Features"]])
	b_conv2 = BiasVar([NETWORK_CONFIG["Conv2Features"]])
	h_conv2 = tf.nn.relu(Conv2D(h_pool1, W_conv2) + b_conv2)

	# Pooling
	h_pool2 = MaxPool(h_conv2, [2, 2])

	# FC1
	W_fc1 = WeightVar([7 * 7 * NETWORK_CONFIG["Conv2Features"], NETWORK_CONFIG["FC1Features"]])
	b_fc1 = BiasVar([NETWORK_CONFIG["FC1Features"]])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * NETWORK_CONFIG["Conv2Features"]])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	keep_prob = tf.placeholder(tf.float32) # probabiity to keep signals
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# FC2
	W_fc2 = WeightVar([NETWORK_CONFIG["FC1Features"], 10])
	b_fc2 = BiasVar([10])
	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	config = tf.ConfigProto(
#		device_count = {"GPU":0}
#		log_device_placement = True
		)
	sess = tf.Session(config = config)
	sess.run(init)

	history = {"TRAIN":[], "TEST":[]}
	for i in range(20000):
		batch_xs, batch_ys = mnist.train.next_batch(50)
		if i % 50 == 0:
			accuracy_train = sess.run(accuracy, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob:1.0})
			print "TRAIN: step %d, accuracy = %f" % (i, accuracy_train)
			history["TRAIN"].append((i, accuracy_train))
		if i % 1000 == 0:
			accuracy_test = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
			print "TEST: step %d, accuracy = %f" % (i, accuracy_test)
			history["TEST"].append((i, accuracy_test))
		sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob:0.5})

	accuracy_test = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
	print "TEST: final accuracy = %f" % accuracy_test
	history["TEST"].append((i, accuracy_test))

	params = {}
	params["NetworkConfig"] = NETWORK_CONFIG
	params["W_conv1"] = W_conv1.eval(sess)
	params["b_conv1"] = b_conv1.eval(sess)
	params["W_conv2"] = W_conv2.eval(sess)
	params["b_conv2"] = b_conv2.eval(sess)
	params["W_fc1"] = W_fc1.eval(sess)
	params["b_fc1"] = b_fc1.eval(sess)
	params["W_fc2"] = W_fc2.eval(sess)
	params["b_fc2"] = b_fc2.eval(sess)

	pickle.dump({"history":history, "params":params}, open("mnist_deep.pickled", "w"))

if __name__ == "__main__":
	Main()

