##########################################################
# This file reproduces the MSE curve in 
# 'An Exploratory Method for Smooth/Transient Decomposition', 
# ilker bayram, ibayram@ieee.org
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import GPdual as GPe

#set the random seed used in the experiments
np.random.seed(109)

# GP generator used for constructing the components of the smooth and transient components
def GenerateGP(t, param):
    N = len(t) # number of time samples
    C = (t**2) + (t**2).T - 2 * np.dot(t, t.T) 
    K = param[2] * np.exp( - C / param[0]) + param[1] * np.eye(N) # covariance matrix
    R = np.linalg.cholesky(K) # this gives K = R * R.T
    x = np.dot( R, np.random.normal(0,1,(N,1)) ) 
    return x

#%%
f0 = 0.025 # fundamental frequency for sinusoid producing the smooth signal

# definition of the prototype signal
signal = lambda x : np.cos(2 * np.pi * f0 * x)

# declare the nonlinearity used for defining the transient signal
nonlin2 = lambda f, p : np.sign(f) * ( (np.power(np.abs(f), p) * (np.abs(f) < 1))/p + (np.abs(f) * (np.abs(f) >= 1)))
#%%
# sampling frequency
fs = 10 

# time variable
Len = 5000
t = np.arange(Len).reshape(Len,1)/fs

# parameters for warping the time variable
param0 = np.array([5e2, 1e-3, 25])

# parameters for the smooth magnitude signal
param1 = np.array([2.5e3, 5e-4, 25])

# parameters for the transient signal
param2 = [10, 1e-5, 0.1]


#%%
er_list = []
for trial in range(20):
    
    tpr = GenerateGP(t, param0) + t
    
    mag = 0.05 * GenerateGP(t, param1) + 1.0
    
    # the smooth component
    smooth = mag * signal(tpr)
    
    transient = nonlin2( GenerateGP(t, param2) ,2).reshape(-1)
    
    smooth = smooth.reshape(-1)
    data = smooth + transient
    
    #%% estimation starts here
    params = {'thold' : 1e-10, 'sigma' : 50, 'lambda' : 0.5, 'sigma0' : 5, 'lambda0' : 0.3, 'sigma1': 10, 'lambda1': 0.01, 'alp' : 0.5, 'bet' : 0.75}
    
    x_smooth = GPe.Algorithm1(data.reshape(-1), fs, params)
    #%%
    K = 30
    filt = np.hamming(2 * K + 1)
    filt = filt / np.sum(filt)
    est2 = np.convolve(data, filt, 'same')
    
    K = 40
    filt = np.hamming(2 * K + 1)
    filt = filt / np.sum(filt)
    est3 = np.convolve(data, filt, 'same')
    
    K = 50
    filt = np.hamming(2 * K + 1)
    filt = filt / np.sum(filt)
    est4 = np.convolve(data, filt, 'same')
    
    K = 60
    filt = np.hamming(2 * K + 1)
    filt = filt / np.sum(filt)
    est5 = np.convolve(data, filt, 'same')
    
    #%%
    M = 1
    k = 2
    error_GP = np.mean(np.abs((smooth  - x_smooth))[M:-M]**k)
    error_LTI1 = np.mean(np.abs((smooth - est2))[M:-M]**k)
    error_LTI2 = np.mean(np.abs((smooth  - est3))[M:-M]**k)
    error_LTI3 = np.mean(np.abs((smooth  - est4))[M:-M]**k)
    error_LTI4 = np.mean(np.abs((smooth  - est4))[M:-M]**k)

    er = np.array([error_GP, error_LTI1, error_LTI2, error_LTI3, error_LTI4])
    er_list.append(er)

    print('----------------Errors, Trial : {}----------------\nLTI : {:.3e}, {:.3e}, {:.3e}, {:.3e}\nGP : {:.3e}'.format(trial, error_LTI1, error_LTI2, error_LTI3, error_LTI4, error_GP))
    
#%% plot the error curve
er_list = np.asarray(er_list)
MS = 6
ind = np.argsort(er_list[:,0])
index = range(1,er_list.shape[0]+1)
fig, ax = plt.subplots(figsize = (6,3))
ax.plot(index, er_list[ind,0], 'k.:', markersize = 8, label = 'Proposed')
ax.plot(index, er_list[ind,1], 'ko:', markerfacecolor='none', markersize = MS, label = 'K = 61')
ax.plot(index, er_list[ind,2], 'kv:', markerfacecolor='none', markersize = MS, label = 'K = 81')
ax.plot(index, er_list[ind,3], 'k^:', markerfacecolor='none', markersize = MS, label = 'K = 101')
ax.plot(index, er_list[ind,4], 'ks:', markerfacecolor='none', markersize = MS, label = 'K = 121')
ax.legend()
ax.set_xlabel('Trial Index')
fig.savefig('../results/ErrorFig.png', format = 'png')