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

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
print '[Test][Warining] Restrict GPU memory usage to 50%'

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

num_epoch = 2
learning_rate = 0.1                                    
dropout_rate = 1.0                  

def scheduler(epoch):

    if epoch > 2 and epoch <=3:
        print 'changing by /10 lr'
        lr = learning_rate/10.0
    elif epoch >3 and epoch <=5:
        print 'changing by /100 lr'
        lr = learning_rate/100.0
    elif epoch >5 and epoch <=7:
        print 'changing by /1000 lr'
        lr = learning_rate/1000.0
    elif epoch > 7:
        print 'changing by /10000 lr'
        lr = learning_rate/10000.0
    else:
        lr = learning_rate

    return lr

change_lr = LearningRateScheduler(scheduler)

f1 = Dense(Nmax, activation='sigmoid',name='layer1',kernel_initializer ='random_uniform', bias_initializer='zeros')
f2 = Dense(2*Nmax,activation='relu', name='layer2',kernel_initializer ='random_uniform', bias_initializer='zeros')
f3 = Dense(Nmax,activation='relu', name='layer3',kernel_initializer ='random_uniform', bias_initializer='zeros')

inputs = Input(shape=(1,))
x = inputs
x = f3(f2(f1(x)))
prediction = x

model = Model(inputs = inputs,outputs=prediction)
#optimizer= keras.optimizers.adam(lr=learning_rate, clipnorm = 1.)
sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='mean_absolute_error')

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
	model.fit(x=s_np,y=target, batch_size=1, epochs=num_epoch)
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
		s = s+10	
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




