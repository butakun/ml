import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np

def Main():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	N = 1000
	X = mnist.train.images[:N,:]
	C = np.argmax(mnist.train.labels[:N], axis = 1)

	mds = manifold.MDS(2, max_iter = 10000, n_init = 1, verbose = 2)
	Y = mds.fit_transform(X)

	plt.scatter(Y[:, 0], Y[:, 1], s = 100, c = C, cmap = plt.cm.Spectral)
	plt.show()

if __name__ == "__main__":
	Main()

