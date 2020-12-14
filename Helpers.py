# -*- coding: utf-8 -*-
"""
Leyla Tarhan
10/2020

functions for variance partitioning analyses. Assumes you just want to compare 2 models.


"""

# %% dependencies

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# %% testing setup

# nObs = 100;
# nVars = 5; 

# # generate some random data:
# X1 = np.random.rand(nObs, nVars);
# X2 = np.random.rand(nObs, nVars);
# Y = np.random.rand(nObs, 1);

# # call it:
# rSquared = predictAllModels(X1, X2, Y);
# partitions = partitionVariance(rSquared);

# # other variants:
# bigRSquared = bigRSquaredAllModels(X1, X2, Y); # calculate traditional R^2 instead of cross-validation r^2
# partitions = partitionVariance(rSquared);

# rSquaredRidge = predictAllModelsRidge(X1, X2, Y); # use ridge regression to minimize the effects of collinearity on the combo model
# partitions = partitionVariance(rSquared);

# %%  fit all the regressions using cross-validation

# input: individual X's and Y to predict
# output: r-squared for each combo of X's (as a dictionary)

def predictAllModels(X1, X2, Y):
    
    # set up the models:
    if X1.ndim == 1 and X2.ndim == 1:
        comboModel = np.stack((X1, X2), axis = 1)
    else:
        comboModel = np.concatenate((X1, X2), axis = 1)
        
    models = [X1, X2, comboModel]
    modelNames = ['X1', 'X2', 'X1&X2']
    
    # loop through the models:
    rSquared = {};
    for m in range(len(models)):
        currModel = models[m];
        
        # use this model to predict Y:
        predictions = [];
        for o in range(len(Y)):
            # hold out this observation:
            x_train = np.delete(currModel, (o), axis = 0)
            y_train = np.delete(Y, o, 0)
            x_test = currModel[o];
            
            if currModel.ndim == 1:
                
                # fit the regression (re-shape the data first so everything's a 2-d array)
                x_train_2d = x_train.reshape(-1, 1);
                y_train_2d = y_train.reshape(-1, 1);
                reg = LinearRegression().fit(x_train_2d, y_train_2d)
                
                # predict the held-out observation:
                x_test_2d = x_test.reshape(1, -1)
                predictions.append(reg.predict(x_test_2d)[0][0])

            else:
                # fit the regression (no reshaping necessary, since already a 2d array)
                reg = LinearRegression().fit(x_train, y_train)
                
                # predict the held-out observation
                x_test_2d = x_test.reshape(1, -1)
                predictions.append(reg.predict(x_test_2d)[0])
            
            
        # correlate predicted / actual Y-values:
        predictionsArray = np.asarray(predictions)
        r = np.corrcoef(predictionsArray, Y, rowvar = False)[0][1]
        rSign = r/abs(r)
        # rSquared[modelNames[m]] = r**2 # NOT keeping the sign
        rSquared[modelNames[m]] = r**2 * rSign # keeping the sign, as in Lescroart et al. (2015)
        
    return rSquared
        
# %%  fit all the regressions using traditional R-squared

# input: individual X's and Y to explain
# output: traditional R-squared for each combo of X's (as a dictionary)

def bigRSquaredAllModels(X1, X2, Y):
    
    # set up the models:
    if X1.ndim == 1 and X2.ndim == 1:
        comboModel = np.stack((X1, X2), axis = 1)
    else:
        comboModel = np.concatenate((X1, X2), axis = 1)

    models = [X1, X2, comboModel]
    modelNames = ['X1', 'X2', 'X1&X2']       
        
    # loop through the models:
    bigRSquared = {};
    for m in range(len(models)):
        currModel = models[m];
        
        if currModel.ndim == 1:
            # re-format the data to be 2d
            currModel_2d = currModel.reshape(-1, 1)
            Y_2d = Y.reshape(-1, 1)
            
            # use this model to explain Y
            reg = LinearRegression().fit(currModel_2d, Y_2d);
            R2 = reg.score(currModel_2d, Y_2d);
        else:
            # use this model to explain Y
            reg = LinearRegression().fit(currModel, Y);
            R2 = reg.score(currModel, Y);
            
        # record the R2 for this model
        bigRSquared[modelNames[m]] = R2
    
        
    return bigRSquared
    

# %% fit all the regressions using cross-validation and ridge regularization

# input: individual X's and Y to predict
# output: r-squared for each combo of X's (as a dictionary)
# [] may need to make this more robust to 1-d inputs

def predictAllModelsRidge(X1, X2, Y):
    
    # set up the models:
    if X1.ndim == 1 and X2.ndim == 1:
        comboModel = np.stack((X1, X2), axis = 1)
    else:
        comboModel = np.concatenate((X1, X2), axis = 1)
        
    models = [X1, X2, comboModel]
    modelNames = ['X1', 'X2', 'X1&X2']
    
    # loop through the models:
    rSquaredRidge = {};
    for m in range(len(models)):
        currModel = models[m];
        
        # use this model to predict Y:
        predictions = [];
        for o in range(len(Y)):
            # hold out this observation:
            x_train = np.delete(currModel, (o), axis = 0)
            y_train = np.delete(Y, o, 0)
            x_test = currModel[o].reshape(1, -1);
            # y_test = Y[o,];
            
            # fit the regression:
            ridgeReg = Ridge(alpha = 1.0)
            ridgeReg.fit(x_train, y_train)
            
            # predict the held-out observation:
            predictions.append(ridgeReg.predict(x_test)[0])
            
            
        # correlated predicted / actual Y-values:
        predictionsArray = np.asarray(predictions)
        r = np.corrcoef(predictionsArray, np.ndarray.flatten(Y), rowvar = False)[0][1];
        rSign = r/abs(r)
        # rSquared[modelNames[m]] = r**2 # NOT keeping the sign
        rSquaredRidge[modelNames[m]] = r**2 * rSign # keeping the sign, as in Lescroart et al. (2015)
        
    return rSquaredRidge


# %% do variance partitioning

# input: r-squared values from all combo's of predictors
# output: dictionary of each partion of the variance


def partitionVariance(rSquared):
    
    partitions = {};
        
    # set up the models to expect:
    modelNames = ['X1', 'X2'];
        
    # get the unique variance for each model (X):
    for m in modelNames:
        # unique variance accounted for by *this model ALONE* = variance 
        # accounted for by the combo of both models - variance accounted for 
        # by the other model
        
        otherModel = modelNames[:];
        otherModel.remove(m);
        otherModel = otherModel[0];
        partitions[m + '-unique'] = rSquared['X1&X2'] - rSquared[otherModel]
        
    # get the variance that's shared between X1 & X2:
    # r^2 model 1 + r^2 model 2 - r^2 duo
    partitions['X1&X2-shared'] = rSquared['X1'] + rSquared['X2'] - rSquared['X1&X2'];
    
    return partitions
    