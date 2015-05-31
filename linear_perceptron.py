''' This is an implimentation of the simplest linear classifier (perceptron) with only one layer
More information could be found here: http://en.wikipedia.org/wiki/Perceptron

Sarunas Girdenas, May, 2015, sg325@exeter.ac.uk


'''

import numpy as np
from random import randint
import matplotlib.pyplot as plt

# compute sample data for perceptron

n = 100
A = 0.3*np.random.randn(n, 2)+[1, 1]
B = 0.3*np.random.randn(n, 2)+[3, 3]
X = np.hstack((np.ones(2*n).reshape(2*n, 1), np.vstack((A, B))))
Y = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))
labels = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))



# firstly, lets use some n
# TODO -> rewrite using 'while'

alpha   = 0.5      # learning rate
err     = []       # save errors for plotting
w       = np.array([0,0,0]) # initial guess about the weights
ress1   = []      # append some result to see the input to step_function
ress2   = []      # append some result to see the input to step_function
ress3   = []

max_iter = 10

ress    = [[] for i in range(max_iter)]

no_iterations = 20

for t in xrange(max_iter):

	for i in xrange(no_iterations):

		index = randint(0,len(X)-1)

		x = X[index,:]
		y = labels[index][0]

		result = np.dot(w,x)

		prediction = 0

		if result >= 0:

			prediction = 1

		else:

			prediction = -1

		w = w + alpha*(y-prediction)*x

		heh = y-prediction

		ress[t].append(heh) # plot this to see mistakes made by classifier

	ress1.append(w[0])
	ress2.append(w[1])
	ress3.append(w[2])




plt.figure(1)
plt.plot(ress1,'k-',label = 'Weight 1')
plt.plot(ress2,'r-',label = 'Weight 2')
plt.plot(ress3,'b-',label = 'Weight 3')
plt.legend(loc = 'upper right')
plt.title('Weights of Classifier')
plt.draw()

leng = len(X)/2

# plot the perceptron result

plt.figure(2)
a,b = -w[1]/w[2], -w[0]/w[2]
l = np.linspace(min(X[:,1]),max(X[:,1]))
p1 = plt.plot(l,a*l+b,'--k',label='hehe')
plt.legend(loc='upper right')
p2 = plt.scatter(X[0:leng,1],X[0:leng,2],color='black')
p3 = plt.scatter(X[leng::,1],X[leng::,2],color='orange')
plt.legend((p2,p3),('Bad News','Good News'),loc='upper left')
plt.title('Linear Perceptron Classifier')
plt.show()