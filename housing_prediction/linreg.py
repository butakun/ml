import numpy as np
import scipy.optimize
import pylab

def f(theta, x, y):

	return 0.5 * np.sum(np.square(np.dot(theta, x) - y))

def fprime(theta, x, y):

	return np.dot(x, (np.dot(theta, x) - y))

def monitor(theta, x, y):
	print f(theta, x, y)

def Main(fname):

	data = np.loadtxt(open(fname))
	data = data.T

	Ntrain = 400
	Xtrain = data[:-1, :Ntrain]
	Ytrain = data[-1, :Ntrain]
	Xtest = data[:-1, Ntrain:]
	Ytest = data[-1, Ntrain:]

	ishuffle = np.arange(Ntrain)
	np.random.shuffle(ishuffle)
	print ishuffle
	Xtrain = Xtrain[:, ishuffle]
	Ytrain = Ytrain[ishuffle]

	m = Xtrain.shape[0]
	print "%d samples of %d" % (Ntrain, m)

	theta = np.random.random((m))

	print "initial cost function = ", f(theta, Xtrain, Ytrain)

	theta = scipy.optimize.fmin_bfgs(f, theta, fprime, args=(Xtrain, Ytrain), callback = lambda theta: monitor(theta, Xtrain, Ytrain), disp = True)
	#theta = scipy.optimize.fmin(f, theta, args=(Xtrain, Ytrain), callback = lambda theta: monitor(theta, Xtrain, Ytrain), disp = True)

	print theta
	print "minimized cost function = ", f(theta, Xtrain, Ytrain)

	rmsTrain = np.sqrt(np.sum(np.square(np.dot(theta, Xtrain) - Ytrain)) / Ytrain.shape[0])
	print "training RMS error = ", rmsTrain

	Y = np.dot(theta, Xtest)
	isort = Y.argsort()

	#for i in range(Y.shape[0]):
	#	print Y[isort[i]], Ytest[isort[i]]

	rmsTest = np.sqrt(np.sum(np.square(Y - Ytest)) / Ytest.shape[0])
	print "test RMS error = ", rmsTest

	pylab.plot(Y[isort], 'ko')
	pylab.plot(Ytest[isort], 'ro')
	pylab.show()

def GenerateStupidLinearTestData(N, fname):

	m = 8
	theta = np.random.random((m))
	X = np.random.random((m, N))
	Y = np.dot(theta, X)

	data = np.zeros((X.shape[1], X.shape[0] + 1))
	data[:, :m] = X.T
	data[:, -1] = Y
	np.savetxt(fname, data)
	open(fname, "a").write("# " + reduce(lambda a, b: a + ' ' + b, map(lambda a: str(a), theta)))

if __name__ == "__main__":
	#GenerateStupidLinearTestData(500, "stupid.data")
	Main("housing.data")

