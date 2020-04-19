#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
# Package for implementing the minimization algorithm described in
# 'An Exploratory Method for Smooth/Transient Decomposition', 
# ilker bayram, ibayram@ieee.org, 2019, https://arxiv.org/abs/2003.02901.                   
###################################################################

import numpy as np


def Algorithm1(signal, fs, params): # 'Smooth Component Estimation' -- Algorithm in the arXiv paper

    # projector onto the constraint set
    Proj = lambda z, a0, a1 : np.minimum(a1, np.maximum(z, a0))

    N = signal.size
    
    # frequency response for the first lower/upper coarse envelopes
    # estimatted in lines 2,3 in Algorithm 1 of the arXiv paper
    H = KernelGP(N, fs, params['sigma'], params['thold'])
    
    MAX_ITER = 1000
   
    lower = (np.min(signal) - 2) * np.ones(signal.shape)
    upper = signal
    z = EnvGP(signal, lower, upper, H, params['alp'], params['lambda'], params['bet'], MAX_ITER)
    
    
    x_low = Proj(signal - z[:N]/params['lambda'], lower, upper) # this is l0 in the pseudocode
      
    # extract the upper envelope  
    upper = (np.max(signal) +2) * np.ones(signal.shape)
    lower = signal
    
    z = EnvGP(signal, lower, upper, H, params['alp'], params['lambda'], params['bet'], MAX_ITER)
    
    x_up = Proj(signal - z[:N]/params['lambda'], lower, upper)
    
    # this is the frequency response of the circulant filtering operation
    H = KernelGP(N, fs, params['sigma0'], params['thold'])
    
    MAX_ITER = 5000

    upper = signal - x_up
    lower = (np.min(signal - x_up) - 1) * np.ones(signal.shape)
    
    z = EnvGP(signal - x_up, lower, upper, H, params['alp'], params['lambda0'], params['bet'], MAX_ITER)
    
    N = signal.size
    
    x_low2 = Proj(signal - x_up - z[:N]/params['lambda0'], lower, upper) + x_up
    
    # extract the upper envelope    
    upper = (np.max(signal - x_low) + 1) * np.ones(signal.shape)
    lower = signal - x_low
    
    z = EnvGP(signal - x_low, lower, upper, H, params['alp'], params['lambda0'], params['bet'], MAX_ITER)
    
    x_up2 = Proj(signal - x_low - z[:N]/params['lambda0'], lower, upper) + x_low
    
    # estimate the smoother signal
    H = KernelGP(N, fs, params['sigma1'], params['thold'])  

    upper = x_up2
    lower = x_low2
    
    z = EnvGP(signal, lower, upper, H, params['alp'], params['lambda1'], params['bet'], MAX_ITER)
    
    return Proj(signal - z[:N]/params['lambda1'], lower, upper)

def KernelGP(N, FPS, sig, thold):
    kernel = lambda t, ell : np.exp( - 0.5 *  t**2 / ell**2 )
    
    h0 = kernel(np.arange(N) / FPS, sig)
    ind = np.argmin(np.abs(h0 - thold))
    h2 = h0[1:ind]
    h = np.zeros(N + h2.size)
    h[:ind] = h0[:ind]
    h[-(ind-1):] = h2[::-1]
    
    return np.fft.fft(h)


def EnvGP(signal, lower, upper, H, alp, lam, bet, MAX_ITER = 1000):
    
    Prox2 = lambda H, alp, z : np.real( np.fft.ifft( np.fft.fft(z) / (1 + alp * np.abs(H) ) ) )
    
    Reflect2 = lambda H, alp, z : 2 * Prox2(H, alp, z) - z
    
    Prox1_shift = lambda z, alp, a0, a1: ((z - alp * a0) * (z < (1+alp) * a0) + 
                                      (z / (1 + alp) ) * (((1+alp) * a0) <= z) * (z <= ((1+alp) * a1)) +
                                      (z - alp * a1) * (((1+alp) * a1) <  z))
    
    Prox1 = lambda z, y, alp, lam, a0, a1 : lam * ( y - Prox1_shift(y - z/lam, alp / lam, a0, a1) )
    
    Reflect1 = lambda z, y, alp, lam, a0, a1: 2 * Prox1(z, y, alp, lam, a0, a1) - z
    
    z0 = np.zeros(H.shape)

    N = signal.size

    for _ in range(MAX_ITER):
        z1 = Reflect2(H, alp, z0)
        
        z1[:N] = Reflect1(z1[:N], signal, alp, lam, lower, upper)
        z1[N:] = -z1[N:]
        
        z0 = (1-bet) * z0 + bet * z1
        
    return Prox2(H, alp, z0)