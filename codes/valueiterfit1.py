import numpy as np


from keras.models import Sequential
from keras.layers.core import Dense, Activation

import matplotlib.pyplot as plt

import scipy.io as sio
import matplotlib
import h5py

from scipy.stats import norm
import math

Nmax2 = 1000
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

S = np.zeros([Nmax])
for i in range(Nmax):
	S[i] = float(i)

V = np.load("valiter1.npy")

model.fit(S, V, nb_epoch=100, batch_size=50)