import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def Main():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	print mnist.train.images.shape

if __name__ == "__main__":
	Main()

