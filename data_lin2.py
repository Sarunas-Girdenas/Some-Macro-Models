'''

This file combines classical cobweb model with simple Linear Perceptron Classifier.

We show that when agents receive some news (random iid gaussian variable, linearly seperable) and updates
estimates only when the guess about the news is correct, additional volatility of estimates occurs.

Sarunas Girdenas, Spring, 2015, sg325@exeter.ac.uk


'''

import matplotlib.pyplot as plt
import numpy as np
from random import randint

Shocks = np.random.rand(500)
g_var  = np.random.rand(500)

# Specify parameters for the cobweb model

time     = 100 # model horizon
n        = time # number of data samples
alpha_1  = 5                        # price equation intercept
c        = -0.5                     # beta_0 + beta_1 in Economic Model
sigma    = 0.5                      # variance of shock
a        = np.zeros([time,1])       # expected price level
p        = np.zeros([time,1])       # actual price level
alpha_2  = np.zeros([time,1])       # parameter alpha_2 in Economic Model
beta_2   = np.zeros([time,1])       # parameter beta_2 in Economic Model
p_avg    = np.zeros([time,1])       # mean for OLS estimatator
wm       = np.zeros([time,1])       # mean of exogenous shock for OLS estimator
delta    = 1                        # define delta next to w in price equation
g_lag    = np.zeros([len(g_var),1]) # exogenous variable g, lagged by 1, g(t-1)
alpha    = 0.1 # learning rate

# create g_lag now

for z in range(1,len(g_var)):
    g_lag[z] = g_var[z-1]

g_lag[0] = 0.5*np.random.randn(1)

# REE values

a2 = alpha_1/(1-c)
b2 = delta/(1-c)

# initialize p and a variables of the economic model
beta_2_initial  = 2 
alpha_2_initial = 1 

a[0] = alpha_2_initial+beta_2_initial*g_lag[0]
p[0] = alpha_1+c*a[0]+delta*g_lag[0]+sigma*Shocks[0] #Initial values of p and a

beta_2[0]  = beta_2_initial
alpha_2[0] = alpha_2_initial

p_avg[0] =  p[0]
wm[0]    = g_lag[0]

# initialize the news for agents to observe. Expand training set after each observation

max_iter = 1
w        = np.array([0,0,0]) # initial guess about the weights
A        = 0.3*np.random.randn(n, 2)+[1, 1]
B        = 0.3*np.random.randn(n, 2)+[3, 3]
X        = np.hstack((np.ones(2*n).reshape(2*n, 1), np.vstack((A, B))))
Y        = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))
labels   = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))

# combine labels with training data

training_data_set = np.concatenate((X,labels),axis=1)
# now shuffle the training set randomly

training_data_set = np.random.permutation([training_data_set])

# now create news from separate data set

A1        = 0.3*np.random.randn(n, 2)+[1, 1]
B1        = 0.3*np.random.randn(n, 2)+[3, 3]
X1        = np.hstack((np.ones(2*n).reshape(2*n, 1), np.vstack((A, B))))
Y1        = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))
labels1   = np.vstack(((-1*np.ones(n)).reshape(n, 1), np.ones(n).reshape(n, 1)))

# combine labels with training data

training_data_set1 = np.concatenate((X1,labels1),axis=1)
# now shuffle the training set randomly

training_data_set1 = np.random.permutation([training_data_set1])

ress1   = [] # append weights
ress2   = [] # append weights
ress3   = [] # append weights

eta = 0.01 # learning rate
errors = []

# the main loop of the model

for i in range(1,time-1):

	# macroeconomic model

	a[i] = alpha_2[i-1] + beta_2[i-1] * g_var[i-1]
	p[i] = alpha_1 + c * a[i] + delta * g_var[i-1] + sigma * Shocks[i]

	# calculate the mean

	p_avg[i] = np.mean(p[0:i])
	wm[i]    = np.mean(g_lag[0:i])

    # index for the news

	index0  = randint(0,2*time-1)
	news    = training_data_set1[0][index0,0:training_data_set.shape[2]-1]
	actual  = training_data_set1[0][index0,X.shape[1]]

    # train classifier on the available data

	for t in xrange(max_iter):

		for h in xrange(1):

			index = randint(0,i)

			x = training_data_set[0][index,0:training_data_set.shape[2]-1]
			y = training_data_set[0][index,X.shape[1]]

			result = np.dot(w,x)

			prediction = 0
		
			if result >= 0:

				prediction = 1

			else:

				prediction = -1

			w = w + eta*(y-prediction)*x

	ress1.append(w[0])
	ress2.append(w[1])
	ress3.append(w[2])

	result2 = np.dot(w,news) # guess based on training data

	prediction2 = 0

	if result2 >= 0:

		prediction2 = 1

	else:

		prediction2 = -1

	# now update estimates only if the news is good

	if i == 1:

		beta_2[i]  = beta_2_initial
		alpha_2[i] = alpha_2_initial

	elif (actual-prediction2) == 0:

		beta_2[i]  = np.sum((g_lag[0:i]-wm[i])*(p[0:i]-p_avg[i]))/np.sum((g_lag[0:i]-wm[i])**2)
		alpha_2[i] = p_avg[i]-beta_2[i]*wm[i]

	else:

		beta_2[i]  = beta_2[i-1]
		alpha_2[i] = alpha_2[i-1]

	heh = actual-prediction2

	errors.append(heh)



plt.figure(1)
plt.plot(beta_2,color='k')
plt.title('Beta_2 Estimate')
plt.ylabel('Value')
plt.xlabel('Time')
plt.draw()

plt.figure(2)
plt.plot(alpha_2,color='g')
plt.title('Alpha_2 Estimate')
plt.ylabel('Value')
plt.xlabel('Time')
plt.draw()

plt.figure(3)
plt.plot(errors,color='b')
plt.title('Prediction Errors from Classifier')
plt.ylabel('Value')
plt.xlabel('Time')
plt.draw()

plt.figure(4)
plt.plot(ress1,'k--o',label = 'Weight 1')
plt.plot(ress2,'r--o',label = 'Weight 2')
plt.plot(ress3,'b--o',label = 'Weight 3')
plt.legend(loc = 'upper right')
plt.title('Weights of Classifier')
plt.show()






