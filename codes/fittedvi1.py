import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine import Layer

import scipy.io as sio
import matplotlib
import h5py


from scipy.stats import norm
import math

Nmax2 = 10
Nmax = 1000	
gamma = 1
discount_factor = 0.9
gaussian_sd = 0.1
m = 1000
K = 10
V = np.zeros([Nmax2])
Q_temp = np.zeros([Nmax])
target = np.zeros(Nmax2)

value_error = 100.0
value_error_threshold = 0.01



num_epoch = 10
learning_rate = 0.01                                    
dropout_rate = 1.0 

f1 = Dense(Nmax, activation='tanh',name='layer1',kernel_initializer ='random_uniform', bias_initializer='zeros')
f2 = Dense(2*Nmax,activation='relu', name='layer2',kernel_initializer ='random_uniform', bias_initializer='zeros')
f3 = Dense(1,activation='relu', name='layer3',kernel_initializer ='random_uniform', bias_initializer='zeros')

inputs = Input(shape=(1,))
x = inputs
x = f3(f2(f1(x)))
prediction = x

model = Model(inputs = inputs,outputs=prediction)
optimizer= keras.optimizers.adam(lr=learning_rate, clipnorm = 1.)
#sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer,loss='mean_squared_error')

def evaluate(s):
	#s_np = np.zeros([1])
	#s_np[0] = 8*(s/Nmax) -4
	return model.predict(s,batch_size=K)


def train(s,target):
	#s_np = np.zeros([1])
	#s_np[0] = 8*(s/Nmax) -4
	'''
	s_np = np.full([data_repeat],(8*s/Nmax -4))
	target_np = np.zeros([data_repeat,Nmax])
	for i in range(data_repeat):
		target_np[i,:] = target[0,:]
	'''
	model.fit(x=s,y=target, batch_size=Nmax2/2, epochs=num_epoch)
	return None

print "check1"

while value_error > value_error_threshold :
	state = np.random.uniform(0,Nmax,Nmax2)
	V = evaluate(state)
	V_temp = V.copy()
	count = 0
	for s in state:
		for a in range(int(s**gamma)):
			temp1 = s**gamma - a
			temp_states = np.random.normal(loc=temp1, scale = gaussian_sd*temp1,size=K)
			temp_values = evaluate(temp_states)
			if a==0:
				reward = 0
			else:
				reward = math.log(a)
			temp_reward = np.full([K],reward)
			temp_reward = temp_reward + discount_factor*temp_values
			Q_temp[a] = np.mean(temp_reward)
		target[count] = np.amax(Q_temp)
		count = count+1
	train(state,target)
	value_error = np.mean(evaluate(state)-V_temp)
	value_error = abs(value_error)
	print value_error




