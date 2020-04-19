###################################################################
# this file lays out Algorithm-1 in the paper,                    
# and produces some figures.
# ilker bayram, ibayram@ieee.org, 2019                            
###################################################################

import numpy as np
import matplotlib.pyplot as plt
import GPdual as GPe

#%% load data

data_mat = np.genfromtxt('../data/Data_Exp.csv', delimiter=',')

# data_mat is a matrix that contains 
# the time variable in its first column,
# smooth component in its second column, 
# transient component in its third column

t = data_mat[:,0] # time variable

fs = 10 # sampling frequency : this should be equal to 1/(t[1] - t[0])

# the observation is obtained by adding the smooth/transient components
smooth = data_mat[:,1]
transient = data_mat[:,2]
data = smooth + transient

#%% estimation starts here

# set the paramaters used in the algorithm, see Algorithm-1 in the letter
params = {'thold' : 1e-10, 'sigma':50, 'lambda' : 0.5, 'sigma0':5, 'lambda0' : 0.3, 'sigma1': 10, 'lambda1' : 0.01,'alp' : 0.5, 'bet' : 0.75}

# declare a projector onto the constraint set
Proj = lambda z, a0, a1 : np.minimum(a1, np.maximum(z, a0))


# step numbers below refer to the pseudocode of Algorithm-1 in the letter
N = data.size

### step-2 -- extract ell0

# frequency response of the large circulant matrix to be used in steps 2, 3
H = GPe.KernelGP(N, fs, params['sigma'], params['thold'])

MAX_ITER = 2000 # for steps 2,3

# set the constraints
lower = (np.min(data) - 2) * np.ones(N)
upper = data

z = GPe.EnvGP(data, lower, upper, H, params['alp'], params['lambda'], params['bet'], MAX_ITER)

ell0 = Proj(data - z[:N]/params['lambda'], lower, upper)

### step-3 -- extract u0

# set the constraints
upper = (np.max(data) +2) * np.ones(N)
lower = data

z = GPe.EnvGP(data, lower, upper, H, params['alp'], params['lambda'], params['bet'], MAX_ITER)

u0 = Proj(data - z[:N]/params['lambda'], lower, upper)

### step-4 -- extract ell
H = GPe.KernelGP(N, fs, params['sigma0'], params['thold'])

MAX_ITER = 5000

upper = data - u0
lower = (np.min(data - u0) - 1) * np.ones(N)

z = GPe.EnvGP(data - u0, lower, upper, H, params['alp'], params['lambda0'], params['bet'], MAX_ITER)

ell = Proj(data - u0 - z[:N]/params['lambda0'], lower, upper) + u0

### step-5 -- extract u

# set the constraints
upper = (np.max(data - ell0) + 1) * np.ones(N)
lower = data - ell0

z = GPe.EnvGP(data - ell0, lower, upper, H, params['alp'], params['lambda0'], params['bet'], MAX_ITER)

u = Proj(data - ell0 - z[:N]/params['lambda0'], lower, upper) + ell0

### step-6 -- estimate the smoother signal

# frequency response of the large circulant matrix to be used in steps 2, 3
H = GPe.KernelGP(N, fs, params['sigma1'], params['thold'])

# set the constraints
upper = u
lower = ell

z = GPe.EnvGP(data, lower, upper, H, params['alp'], params['lambda1'], params['bet'], MAX_ITER)

x_smooth = Proj(data - z[:N]/params['lambda1'], lower, upper)

x2 = np.real(np.fft.ifft(np.fft.fft(z) * H)[:N])

print('***Convergence Check***\n{} <- This number should be practically zero'.format(np.max(np.abs(x_smooth - x2))))

#%% LTI estimation using Hamming filters, for comparison
K = 20
filt = np.hamming(2 * K + 1)
filt = filt / np.sum(filt)
est2 = np.convolve(data, filt, 'same')

K = 35
filt = np.hamming(2 * K + 1)
filt = filt / np.sum(filt)
est3 = np.convolve(data, filt, 'same')

K = 50
filt = np.hamming(2 * K + 1)
filt = filt / np.sum(filt)
est4 = np.convolve(data, filt, 'same')

#%% print out the errors 

M = 1
k = 2
error_GP = np.mean(np.abs((smooth  - x_smooth))[M:-M]**k)
error_LTI = np.mean(np.abs((smooth  - est2))[M:-M]**k)
error_LTI2 = np.mean(np.abs((smooth  - est3))[M:-M]**k)
error_LTI3 = np.mean(np.abs((smooth  - est4))[M:-M]**k)

print('\n\n*** Errors (MSE) ***\nLTI : {:.4f}, {:.4f}, {:.4f} \nProposed : {:.4f}'.format(error_LTI, error_LTI2, error_LTI3, error_GP))


#%% figures...

fsize = (10,5)
er_lim = [-0.5, 0.5]

fig, ax = plt.subplots(figsize = fsize)  
ax.plot(t, data, label = 'Observation')
ax.plot(t, smooth, label = 'Smooth Component')
ax.plot(t, ell0, label = 'Crude Lower Envelope (ell0)')
ax.plot(t, u0, label = 'Crude Upper Envelope (u0)')
ax.set_xlabel('Time (sec)')
ax.legend()
ax.set_title('Observation and Underlying Smooth Signal')
fig.savefig('../results/Obs1.png', format = 'png')

fig, ax = plt.subplots(figsize = fsize)  
ax.plot(t, smooth  - est2, label = 'K = 41')
ax.plot(t, smooth  - est3, label = 'K = 71')
ax.plot(t, smooth  - est4, label = 'K = 101')
ax.set_title('LTI estimation errors for different length filters')
ax.set_ylim(er_lim)
ax.legend()
fig.savefig('../results/LTIer.png', format = 'png')

fig, ax = plt.subplots(figsize = fsize)  
ax.plot(t, data, label = 'Observation')
ax.plot(t, ell, label = 'lower envelope')
ax.plot(t, u, label = 'upper enbelope')
ax.plot(t, x_smooth, label = 'Estimate of the Smooth Component')
ax.set_title('Observation, Envelopes, Estimate of the Smooth Component')
ax.set_xlim([200, 300])
ax.legend()
fig.savefig('../results/Envel_Est.png', format = 'png')

fig, ax = plt.subplots(figsize = fsize)  
ax.plot(t, data, label = 'Observation')
ax.plot(t, x_smooth, label = 'Estimate of the Smooth Component')
ax.plot(t, smooth, label = 'True Smooth Component')
ax.set_title('Estimate of the Smooth Part Obtained by the Proposed Method')
ax.set_xlim([200, 300])
ax.legend()
fig.savefig('../results/Estimate.png', format = 'png')

fig, ax = plt.subplots(figsize = fsize)  
ax.plot(t, smooth - x_smooth, label = 'Proposed')
ax.plot(t, smooth - est4, label = 'LTI, K = 101')
ax.set_title('Error for the proposed method vs one of the LTI filters')
ax.legend()
ax.set_ylim(er_lim)
fig.savefig('../results/ErrorProp.png', format = 'png')
