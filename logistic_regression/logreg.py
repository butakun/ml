import sklearn.datasets
import numpy as np
import scipy.optimize
import pylab

def f(theta, x, y):

	h = 1.0 / (1.0 + np.exp(-np.dot(theta, x)))
	#print "theta x = ", np.dot(theta, x)
	J = -np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
	print "cost = ", J
	return J

def fprime(theta, x, y):

	h = 1.0 / (1.0 + np.exp(-np.dot(theta, x)))
	gradJ = np.dot(x, (h - y))
	#print "h - y = ", h - y
	#print "x = ", x
	#print "gradJ = ", gradJ
	return gradJ

def monitor(theta, x, y):
	return
	print f(theta, x, y)

def Test(theta, x, y):
	ytest = 1.0 / (1.0 + np.exp(-np.dot(theta, x)))
	d = np.rint(ytest) - y
	f = np.sum(np.abs(d))
	print "wrong classifications = %d out of %d samples" % (f, len(y))

def Main():

	mnist = sklearn.datasets.fetch_mldata("MNIST original")
	data = mnist.data.T
	target = mnist.target

	# Extract only 0's and 1's
	mask = target < 2
	data = data[:, mask]
	target = target[mask]
	print "data.shape = ", data.shape, target.shape

	# Normalization
	mean = np.mean(data)
	std = np.std(data)
	data = data - mean
	data = data / std

	print "Original data statistics: mean = %f, std = %f" % (mean, std)
	print "Normalized data statistics: mean = %f, std = %f" % (np.mean(data), np.std(data))

	# Split data into training and testing data
	Ntrain = 10000
	indices = np.arange(target.shape[0])
	np.random.shuffle(indices)
	Xtrain = data[:, indices[:Ntrain]]
	Ytrain = target[indices[:Ntrain]]
	Xtest = data[:, indices[Ntrain:]]
	Ytest = target[indices[Ntrain:]]

	m = Xtrain.shape[0]
	print "%d samples of %d" % (Ntrain, m)

	theta = np.random.random((m)) * 0.001

	print "initial cost function = ", f(theta, Xtrain, Ytrain)

	#theta = scipy.optimize.fmin_bfgs(f, theta, fprime, args=(Xtrain, Ytrain), callback = lambda theta: monitor(theta, Xtrain, Ytrain), disp = True)
	#theta = scipy.optimize.fmin(f, theta, args=(Xtrain, Ytrain), disp = True)
	#theta, nfeval, rc = scipy.optimize.fmin_tnc(f, theta, approx_grad = True, args=(Xtrain, Ytrain), disp = True)
	theta, nfeval, rc = scipy.optimize.fmin_tnc(f, theta, fprime, args=(Xtrain, Ytrain), disp = True)

	print theta
	print "minimized cost function = ", f(theta, Xtrain, Ytrain)

	print "For training set:"
	Test(theta, Xtrain, Ytrain)

	print "For validation set:"
	Test(theta, Xtest, Ytest)

if __name__ == "__main__":
	Main()

