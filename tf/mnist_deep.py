import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

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
	W_conv1 = WeightVar([5, 5, 1, 32])
	b_conv1 = BiasVar([32])
	h_conv1 = tf.nn.relu(Conv2D(x_image, W_conv1) + b_conv1)

	# Pooling
	h_pool1 = MaxPool(h_conv1, [2, 2])

	# ReLU(Wx + b)
	W_conv2 = WeightVar([5, 5, 32, 64])
	b_conv2 = BiasVar([64])
	h_conv2 = tf.nn.relu(Conv2D(h_pool1, W_conv2) + b_conv2)

	# Pooling
	h_pool2 = MaxPool(h_conv2, [2, 2])

	# FC1
	W_fc1 = WeightVar([7 * 7 * 64, 1024])
	b_fc1 = BiasVar([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	keep_prob = tf.placeholder(tf.float32) # probabiity to keep signals
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# FC2
	W_fc2 = WeightVar([1024, 10])
	b_fc2 = BiasVar([10])
	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	for i in range(20000):
		batch_xs, batch_ys = mnist.train.next_batch(50)
		if i % 50 == 0:
			accuracy_train = sess.run(accuracy, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob:1.0})
			print "TRAIN: step %d, accuracy = %f" % (i, accuracy_train)
		if i % 1000 == 0:
			print "TEST: accuracy = %f" % sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
		sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob:0.5})

	print "TEST: accuracy = %f" % sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})

if __name__ == "__main__":
	Main()
