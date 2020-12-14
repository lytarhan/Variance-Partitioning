# -*- coding: utf-8 -*-
"""
Leyla Tarhan
12/2020

Example showing how to call the variance partitioning functions using simulated data.

"""
# %% Dependencies

from Helpers import *
import numpy as np
import matplotlib.pyplot as plt

# %% simulate some data

# comparing 2 sets of variables, which each have 5 dimensions (e.g., 5 ways to measure air quality) and 100 observations. 
# Note: this method should work to compare any 2 sets of variables, no matter how many dimensions they have.
nObs = 100;
nVars = 5; 

# generate some random data:
X1 = np.random.rand(nObs, nVars);
X2 = np.random.rand(nObs, nVars);
Y = np.random.rand(nObs, 1);

# %% do standard variance partitioning

# predict Y using all 3 combinations of X1 & X2:
rSquared = predictAllModels(X1, X2, Y);

# partition the variance:
partitions = partitionVariance(rSquared);

# plot the results:
plotLabels = list(partitions.keys())
variances = [i*100 for i in list(partitions.values())]
plt.bar(plotLabels, variances)
plt.xlabel('type of variance')
plt.ylabel('% of variance explained')
plt.title('Example results using standard variance partitioning')
plt.show()


# %% other variance of variance partitioning

# calculate traditional R^2 instead of cross-validated r^2 to measure variance explained by each regression
bigRSquared = bigRSquaredAllModels(X1, X2, Y); 
bigRSquaredPartitions = partitionVariance(bigRSquared);
plotLabels = list(bigRSquaredPartitions.keys())
bigRSquaredVariances = [i*100 for i in list(bigRSquaredPartitions.values())]
plt.bar(plotLabels, bigRSquaredVariances)
plt.xlabel('type of variance')
plt.ylabel('% of variance explained')
plt.title('Example results using traditional R^2')
plt.show()

# use ridge regression to minimize the effects of collinearity on the combo model
rSquaredRidge = predictAllModelsRidge(X1, X2, Y); 
ridgePartitions = partitionVariance(rSquaredRidge);
plotLabels = list(ridgePartitions.keys())
ridgeVariances = [i*100 for i in list(ridgePartitions.values())]
plt.bar(plotLabels, ridgeVariances)
plt.xlabel('type of variance')
plt.ylabel('% of variance explained')
plt.title('Example results using ridge-regularization')
plt.show()

