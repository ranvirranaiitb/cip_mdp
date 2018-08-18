import numpy as np

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

def discrete_pdf(x):
	dpdf = np.zeros([Nmax])
	if x == 0 :
		return dpdf
	else :
		for i in range(Nmax):
			dpdf[i] = norm.pdf(i, loc = x , scale = gaussian_sd*x)
		dpdf = dpdf/np.sum(dpdf)
		return dpdf

dtpdf = np.zeros([Nmax,Nmax])
for s in range(Nmax):
	dtpdf[s,:] = discrete_pdf(s)




while value_error>=value_error_threshold :
	for s in range(Nmax-1,-1,-1):
		for a in range(0,int(H[s])):
			temp1 = 0
			for s2 in range(Nmax):
				temp1 = temp1 + dtpdf[(int(H[s])-a),s2]*V[s2]
			Q[s,a] = U[a] + discount_factor*temp1
		V[s] = np.amax(Q[s,:])
		#print V[s]
	value_error = np.sum(V_temp-V)/Nmax
	value_error = abs(value_error)
	print value_error
	V_temp = V.copy()

legend = []
plt.plot(np.range(0,Nmax), V)
legend.append('Value function')
#plt.plot(snr_collect, NN_ber_collect)
#legend.append('NN')

#plt.yscale('log')
plt.xlabel('states')
plt.ylabel('V[s]') 
plt.legend(legend, loc=3)
plt.show()