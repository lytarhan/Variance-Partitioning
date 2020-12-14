# -*- coding: utf-8 -*-
"""
Leyla Tarhan
12/2020

Code exploring cases where you might find negative shared variance, using simulations.

Main strategy: simulate 1-d data (X1 and X2) and some observations (Y). 
Vary how correlated X1 and X2 are, and assess whether you get negative 
partitioned variance at any level of correlation.

Many thanks to Dan Janini for the method I used in these simulations, and to 
Ruosi Wang for additional ideas!
"""
# %% load dependencies

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import random

# my helpers:
from Helpers import *

# %% set up simulation parameters
# always comparing 2 models

nObs = 100; # observations in the data
nVars = 1; # dimensions in each 'model' (X1 or X2)
nIters = 1000; # simulation iterations
corrRange = np.arange(-.95, 0.95, 0.01); # possible correlations between x1 & x2
errorSD = 1; # standard deviation of the error term for simulating Y

# %% run simulations

# Does it matter whether you measure variance explained using cross-validation?
# try 2 methods: cross-validated r^2 and traditional R^2
rSquaredTypes = ['cross-validated', 'R2']

# loop through the ways of measuring variance explained
for r in rSquaredTypes:
    # set up to store the results for this analysis type
    uniqueX1 = [];
    uniqueX2 = [];
    shared = [];
    corrs = [];
    
    print('\n\nvariance measurement: ' + r + '...')
    # loop through the iterations
    for i in range(nIters):   
        if (i+1)%10 == 0:
            print('iteration ' + str(i+1) + '/' + str(nIters) + '...')
        
        # generate X1 & X2 & Y
        currCorr = random.sample(list(corrRange), 1)[0]
        corrs.append(currCorr)
        corr_mat= np.array([[1.0, currCorr],
                    [currCorr, 1.0]])
        upper_chol = cholesky(corr_mat) # cholesky decomposition
        rnd = np.random.normal(0.0, 1.0, size=(nObs, 2)) # 2 random vectors drawn from a normal distribution
        rndCorr = rnd @ upper_chol # cholesky decomposition * random vectors
        X1 = rndCorr[:,0]
        X2 = rndCorr[:,1]
        # np.corrcoef(X1, X2) # to check that the correlation is as expected
        
        # Y = X1 + X2 + error
        error = np.random.normal(0.0, errorSD, size = (nObs)); # error drawn from distribution centered on 0 (manipulate the variance)
        YElements = np.stack((X1, X2, error), axis = 1)
        Y = np.sum(YElements, axis = 1)
        
        # variance partitioning based on current analysis type
        if r == 'cross-validated':
            # fit all the regressions to get cross-validated r^2 for each combo of regressors
            rSquared = predictAllModels(X1, X2, Y);
        elif r == 'R2':
            rSquared = bigRSquaredAllModels(X1, X2, Y);
        else:
            print('didn''t recognize that analysis type.')
            
        # partition the variance
        partitions = partitionVariance(rSquared);
        
        # store this iteration's results  
        uniqueX1.append(partitions['X1-unique'])
        uniqueX2.append(partitions['X2-unique'])
        shared.append(partitions['X1&X2-shared'])

        
        
    # scatterplots to visualize the results:
    f, axs = plt.subplots(1, 3, figsize = (20, 5))
    plt.setp(axs, ylim=(-0.5, 0.9))
    f.suptitle("Variance Measurement: " + r, fontsize=14)
    axs[0,].scatter(corrs, uniqueX1); axs[0,].set_xlabel('r(X1, x2)'); axs[0,].set_ylabel('unique variance - X1'); axs[0,].axhline(y=0, color = 'r')
    axs[1,].scatter(corrs, uniqueX2); axs[1,].set_xlabel('r(X1, x2)'); axs[1,].set_ylabel('unique variance - X2'); axs[1,].axhline(y=0, color = 'r')
    axs[2,].scatter(corrs, shared); axs[2,].set_xlabel('r(X1, x2)'); axs[2,].set_ylabel('shared variance'); axs[2,].axhline(y=0, color = 'r')    
    plt.show()

print('\n\nDONE!')

# regardless of how you measure variance explained, if X1 & X2 are negatively-correlated you get negative shared variance.
