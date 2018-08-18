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



Nmax = 1000	
gamma = 1
discount_factor = 0.9
gaussian_sd = 0.1
initial_state = 100
epsilon = 0.9
data_repeat = 10

V = np.zeros([Nmax])
V_temp = np.zeros([Nmax])
Q = np.zeros([Nmax,Nmax])
H = np.zeros([Nmax])
for i in range(Nmax):
	H[i] = i**gamma
U = np.zeros([Nmax])
for i in range(1,Nmax):
	U[i] = math.log(i)
value_error = 100.0
value_error_threshold = 0.01

num_epoch = 10
learning_rate = 0.1                                    
dropout_rate = 1.0                  

model = Sequential()
model.add(Dense(Nmax,input_shape=(1,)))
model.add(Activation('tanh'))
model.add(Dense(Nmax))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')


def evaluate(s):
	s_np = np.zeros([1])
	s_np[0] = 8*(s/Nmax) -4
	return model.predict(s_np,batch_size=1)

def train(s,target):
	s_np = np.zeros([1])
	s_np[0] = 8*(s/Nmax) -4
	'''
	s_np = np.full([data_repeat],(8*s/Nmax -4))
	target_np = np.zeros([data_repeat,Nmax])
	for i in range(data_repeat):
		target_np[i,:] = target[0,:]
	'''
	model.fit(s_np, target, nb_epoch=num_epoch, batch_size=1)
	return None

game_iterations = 50
play_length = np.zeros([game_iterations])


for game_iter in range(game_iterations):	
	s = 2 #np.random.uniform(0,Nmax)
	target = evaluate(s)
	#print target[0,1]
	while (s>=1) and (s<(Nmax-1)) :
		temp3 = np.random.uniform(0,1)
		if(temp3>epsilon):
			a = np.argmax(target[0,:])
			
			if a>s**gamma:
				temp2 = 0
				a = np.random.randint(0,int(s**gamma))
				for i in range(0,int(s**gamma)+1):
					if temp2<target[0,i]:
						temp2 = target[0,i]
						a = i
			
		else:
			a = np.random.randint(0,int(s**gamma)+1)
		print 'actions' + str(a)
		print  'states' + str(s)
		#print s**gamma

		if a>0:
			reward = math.log(a)
		else:
			reward = 0
		temp1 = s**gamma - a
		s2 = np.random.normal(loc= temp1 , scale= 0.1)
		t = evaluate(s2)
		maxQ = np.amax(t)
		target[0,a] = reward + discount_factor*maxQ
		
		for i in range(int(s**gamma)+1,Nmax):
			target[0,i] = 0
		
		train(s,target)
		s = s2	
		#print "iterations"
		#print "s="
		#print s
		play_length[game_iter] = play_length[game_iter] + 1
	print 'Game_iterations' + str(game_iter)

for i in range(Nmax):
	temp_target = evaluate(i)
	V[i] = np.amax(temp_target)

np.save("qlrearning1V.npy",V)
np.save("qlearning1PL.npy",play_length)

model.save_weights('qlearning1.h5')




