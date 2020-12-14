# -*- coding: utf-8 -*-
"""
Leyla Tarhan
12/2020

Code exploring a case where I found negative unique variance in my own data.

"""
# %% dependencies

from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from statsmodels.stats.outliers_influence import variance_inflation_factor 

from Helpers import *

# %% import the data

# data are average neural responses to 60 actions, in the occipito-parietal cortex of the brain

dataFile = "ExampleBrainData.mat"
data = loadmat(dataFile)

# ------------
# BRAIN DATA

# this brain region's response matrix is stored as a list -- convert to an array
avgResponsePattern = np.array(data['avgOPC'])

# --------------
# RATINGS DATA
ratingsDict = data['ratings']

# get all the ratings types
models = list(ratingsDict.keys())
cleanRatingsDict = {}
for m in models:
    # get the ratings matrix (as a list of lists):
    rateLists = ratingsDict[m]
    # first level: each list = 1 video
    # 2nd level: each video = 1 list of 20 features
    
    # convert to numpy array
    ratem = np.array(rateLists) # vids x features
    
    # store it
    cleanRatingsDict[m] = ratem
    

# %% setup for variance partitioning

X1 = cleanRatingsDict[models[0]] # vids x features
X2 = cleanRatingsDict[models[1]] # vids x features
Y = avgResponsePattern # vids x 1

# %% do variance partitioning

# (1) fit all the regressions
# inputs: the ActionMaps data for a single sector (avged over voxels)
# outputs: r-squared for each regression

rSquared = predictAllModels(X1, X2, Y);

# plot it
plt.bar(list(rSquared.keys()), list(rSquared.values()))
plt.xlabel('predictors')
plt.ylabel('cross-validated r-squared')
plt.title('Example Brain Data\n X1 = body part involvement, X2 = body part visibility')
plt.show()
# the r-squared value is *lower* for the regression based on X1 & X2 than it is for the regressions based on just X1 or just X2


# (2) do variance partitioning
# inputs: the r-squared values
# outputs: variance for each partition in the 2-circle venn diagram

partitions = partitionVariance(rSquared)

# plot it
plotLabels = list(partitions.keys())
variances = [i*100 for i in list(partitions.values())]
plt.bar(plotLabels, variances)
plt.xlabel('type of variance')
plt.ylabel('% variance explained')
plt.title('Example Brain Data\n X1 = body part involvement, X2 = body part visibility')
plt.show()
# unique variance is *negative* for X1

# %% try to get rid of the negative unique variance

# (1) use traditional R^2 -- which doesn't do any prediction and therefore 
# isn't affected by over-fitting -- instead of r^2 to measure variance explained
bigRSquared = bigRSquaredAllModels(X1, X2, Y);
bigRSquaredPartitions = partitionVariance(bigRSquared);

# plot it
plotLabels = list(bigRSquaredPartitions.keys())
bigRSquaredVariances = [i*100 for i in list(bigRSquaredPartitions.values())]
plt.bar(plotLabels, bigRSquaredVariances)
plt.xlabel('type of variance')
plt.ylabel('% variance explained, based on traditional R^2')
plt.title('Example Brain Data')
plt.show()

# (2) use ridge-regularized r^2 to measure variance explained, to reduce the danger of over-fitting
rSquaredRidge = predictAllModelsRidge(X1, X2, Y);
ridgePartitions = partitionVariance(rSquaredRidge);

# plot it
plotLabels = list(partitions.keys())
ridgeVariances = [i*100 for i in list(ridgePartitions.values())];
plt.bar(plotLabels, ridgeVariances)
plt.xlabel('type of variance')
plt.ylabel('% variance explained, based on ridge r^2')
plt.title('Example Brain Data')
plt.show()




