import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.examples.tutorials.mnist import input_data

def WeightVar(shape):

	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = "W")

def BiasVar(shape):

	return tf.Variable(tf.constant(0.1, shape = shape), name = "b")

def Main():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784], name = "input")

	with tf.name_scope("Encode1"):
		W_encode1 = WeightVar([784, 20])
		b_encode1 = BiasVar([20])
		h_sigmoid1 = tf.sigmoid(tf.matmul(x, W_encode1) + b_encode1)

	with tf.name_scope("Decode1"):
		#W_decode1 = WeightVar([20, 784])
		b_decode1 = BiasVar([784])
		#h_decode1 = tf.matmul(h_sigmoid1, W_decode1) + b_decode1
		h_decode1 = tf.matmul(h_sigmoid1, tf.transpose(W_encode1)) + b_decode1

	with tf.name_scope("L2Norm"):
		l2norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x - h_decode1), reduction_indices = [1])))

	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(l2norm)
	#train_step = tf.train.AdamOptimizer(1e-2).minimize(l2norm)

	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	for i in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x:batch_xs})
		if i % 50 == 0:
			print i, sess.run(l2norm, feed_dict = {x:batch_xs})

	w = sess.run(W_encode1)

	# Save the decomposition
	#decomp = []
	#for i in range(10):
	#	image = mnist.train.images[i, :].reshape(1,-1)
	#	sess.run(h_decode1, feed_dict = {x:image})
	#	decomp.append({"image":image, "coefficients":h_sigmoid1.eval(sess), "reconstructed":h_decode1.eval(sess)})
	#pickle.dump(decomp, open("mnist_autoencoder.pickled", "w"))

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

