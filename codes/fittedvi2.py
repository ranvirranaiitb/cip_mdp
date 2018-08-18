import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation

import matplotlib.pyplot as plt


import scipy.io as sio
import matplotlib
import h5py


from scipy.stats import norm
import math

Nmax2 = 100
Nmax = 1000	
gamma = 0.9
discount_factor = 0.9
gaussian_sd = 0.1
m = 1000
K = 1
V = np.zeros([Nmax2])
Q_temp = np.zeros([Nmax])
target = np.zeros(Nmax2)

value_error = 100.0
value_error_threshold = 0.1



model = Sequential()
model.add(Dense(1000,input_shape=(1,)))
model.add(Activation('tanh'))
model.add(Dense(1000))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')



def evaluate(s1):
	#s_np = np.zeros([1])
	#s_np[0] = 8*(s/Nmax) -4
	s1 = 2*s1/Nmax - 1
	return model.predict(s1)


def train(s2,target2):
	#s_np = np.zeros([1])
	#s_np[0] = 8*(s/Nmax) -4
	'''
	s_np = np.full([data_repeat],(8*s/Nmax -4))
	target_np = np.zeros([data_repeat,Nmax])
	for i in range(data_repeat):
		target_np[i,:] = target[0,:]
	'''
	#target = 2*target/Nmax - 1
	s2=2*s2/Nmax - 1
	model.fit(s2, target2, nb_epoch=100, batch_size=Nmax2/2)
	return None

print "check1"

state = np.random.uniform(0,Nmax,Nmax2)

while value_error > value_error_threshold :
	
	#print state
	V = model.predict(state)
	print V
	V_temp = V.copy()
	count = 0
	for s in state:
		next_state = s**gamma
		next_state_discrete = int(s**gamma)
		for a in range(next_state_discrete):
			temp1 = next_state - a
			temp_states = np.random.normal(loc=temp1, scale = 0 ,size=K)
			temp_values = model.predict(temp_states)
			if a==0:
				reward = 0
			else:
				reward = math.log(a)
			temp_reward = np.full([K],reward)
			temp_reward = temp_reward + discount_factor*temp_values
			Q_temp[a] = np.mean(temp_reward)
		target[count] = np.amax(Q_temp)
		count = count+1
	print target
	print state
	model.fit(state,target,nb_epoch=100, batch_size=Nmax2/2)
	value_error = np.mean(model.predict(state)-V_temp)
	value_error = abs(value_error)
	print count
	print value_error




