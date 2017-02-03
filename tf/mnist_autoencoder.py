import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def WeightVar(shape):

	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def BiasVar(shape):

	return tf.Variable(tf.constant(0.1, shape = shape))

def Main():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])
	W_encode1 = WeightVar([784, 20])
	b_encode1 = BiasVar([20])

	h_sigmoid1 = tf.sigmoid(tf.matmul(x, W_encode1) + b_encode1)

	W_decode1 = WeightVar([20, 784])
	b_decode1 = BiasVar([784])
	h_decode1 = tf.matmul(h_sigmoid1, W_decode1) + b_decode1

	l2norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x - h_decode1), reduction_indices = [1])))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(l2norm)

	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x:batch_xs})
		if i % 50 == 0:
			print i, sess.run(l2norm, feed_dict = {x:batch_xs})

	w = sess.run(W_encode1)

	nRows, nCols = 2, 10
	cmap = plt.get_cmap("gray")
	fig, axes = plt.subplots(nRows, nCols)
	axesFlat = axes.flatten()
	for row in range(nRows):
		for col in range(nCols):
			i = (row * nCols) + col
			img = w[:, i].reshape(28, 28)
			axesFlat[i].get_xaxis().set_visible(False)
			axesFlat[i].get_yaxis().set_visible(False)
			axesFlat[i].imshow(img, cmap = cmap)
	plt.show()

if __name__ == "__main__":
	Main()

