import numpy as np

import matplotlib.pyplot as plt

import scipy.io as sio
import matplotlib
import h5py

from scipy.stats import norm
import math


Nmax = 1000	
gamma = 0.9
discount_factor = 0.9
gaussian_sd = 0.1

A = np.zeros([Nmax])
V = np.load("valiter1.npy")

for s in range(Nmax):
	temp = np.zeros([Nmax])
	for a in range(0,int(s**gamma)):
		if a==0:
			reward = 0
		else:
			reward = math.log(a)
		temp[a] = reward + discount_factor*V[int(s**gamma)-a]
	A[s] = np.argmax(temp)

np.save("valiter1A.npy",A)
