import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = "./mnist_deep_log"

NETWORK_CONFIG = {
	"Conv1Features":32,
	"Conv2Features":64,
	"FC1Features":1024,
}

def WeightVar(shape):

	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = "W")

def BiasVar(shape):

	return tf.Variable(tf.constant(0.1, shape = shape), name = "b")

def Conv2D(x, W):

	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def MaxPool(x, shape):

	return tf.nn.max_pool(x, ksize = [1, shape[0], shape[1], 1], strides = [1, shape[0], shape[1], 1], padding = "SAME")

def Main():

	if tf.gfile.Exists(LOGDIR):
		tf.gfile.DeleteRecursively(LOGDIR)
	tf.gfile.MakeDirs(LOGDIR)

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# Input
	with tf.name_scope("input"):
		x = tf.placeholder(tf.float32, [None, 784], name = "x-input")
		y_ = tf.placeholder(tf.float32, [None, 10], name = "y-input")

	with tf.name_scope("input_reshaped"):
		x_image = tf.reshape(x, [-1, 28, 28, 1], name = "x-reshaped")
		tf.summary.image("input", x_image, 10)

	# ReLU(Wx + b)
	with tf.name_scope("Conv1"):
		W_conv1 = WeightVar([5, 5, 1, NETWORK_CONFIG["Conv1Features"]])
		b_conv1 = BiasVar([NETWORK_CONFIG["Conv1Features"]])
		h_conv1 = tf.nn.relu(Conv2D(x_image, W_conv1) + b_conv1, name = "ReLU")

	# Pooling
	with tf.name_scope("MaxPool1"):
		h_pool1 = MaxPool(h_conv1, [2, 2])

	# ReLU(Wx + b)
	with tf.name_scope("Conv2"):
		W_conv2 = WeightVar([5, 5, NETWORK_CONFIG["Conv1Features"], NETWORK_CONFIG["Conv2Features"]])
		b_conv2 = BiasVar([NETWORK_CONFIG["Conv2Features"]])
		h_conv2 = tf.nn.relu(Conv2D(h_pool1, W_conv2) + b_conv2, name = "ReLU")

	# Pooling
	with tf.name_scope("MaxPool2"):
		h_pool2 = MaxPool(h_conv2, [2, 2])

	# FC1
	with tf.name_scope("FC1"):
		W_fc1 = WeightVar([7 * 7 * NETWORK_CONFIG["Conv2Features"], NETWORK_CONFIG["FC1Features"]])
		b_fc1 = BiasVar([NETWORK_CONFIG["FC1Features"]])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * NETWORK_CONFIG["Conv2Features"]], name = "reshape")
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = "ReLU")

	# Dropout
	with tf.name_scope("DropOut"):
		keep_prob = tf.placeholder(tf.float32, name = "keep_prob") # probabiity to keep signals
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# FC2
	with tf.name_scope("FC2"):
		W_fc2 = WeightVar([NETWORK_CONFIG["FC1Features"], 10])
		b_fc2 = BiasVar([10])
		y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# Loss
	with tf.name_scope("Loss"):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# Accuracy
	with tf.name_scope("Accuracy"):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	config = tf.ConfigProto(
#		device_count = {"GPU":0}
#		log_device_placement = True
		)
	sess = tf.Session(config = config)
	sess.run(init)

	writer = tf.summary.FileWriter(LOGDIR + "/train", sess.graph)

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

	writer.close()

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

def Vis():

	data = pickle.load(open("mnist_deep.pickled"))
	params = data["params"]
	W_conv1 = params["W_conv1"]

	nRows, nCols = 4, 8
	cmap = plt.get_cmap("gray")
	fig, axes = plt.subplots(nRows, nCols)
	axesFlat = axes.flatten()
	for row in range(nRows):
		for col in range(nCols):
			i = (row * nCols) + col
			img = W_conv1[:, :, 0, i]
			axesFlat[i].get_xaxis().set_visible(False)
			axesFlat[i].get_yaxis().set_visible(False)
			axesFlat[i].imshow(img, cmap = cmap)
	plt.show()

if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == "vis":
		Vis()
	else:
		Main()

