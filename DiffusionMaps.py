#****************************************************************
# Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from analytics.utils import * 
import numpy as np

class DiffusionMaps(Algorithm):
    def __init__(self):
        super(DiffusionMaps, self).__init__()
        self.parameters = ['numDim','timeStep','sigma']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Diffusion Maps'
        self.type = 'Dimension Reduction'
        self.description = 'Performs diffusion maps dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Timestep", "attrname" : "timeStep", "value" : 1, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Sigma", "attrname" : "sigma", "value" : 1, "type" : "input" , "step": 1, "max": 15, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        X = self.inputData        
        if np.size(X, axis=0) > 3000:
            numSamples = np.size(X, axis=0)
            print "Due to the large number of instances = %d, diffusion maps may run out of memory" \
            % numSamples
        # assume X being an array and normalize X    
        #X = np.double(X)
        xVector = X.flatten(1)
        xMin = np.min(xVector)
        X = X - xMin
        xVector2 = X.flatten(1)
        xMax = np.max(xVector2)
        X = X / xMax
        
        # compute Gaussian kernel matrix
        print "Compute Markov forward transition probability matrix with %d timesteps..." \
        % self.timeStep
        
        sumX = np.sum(np.power(X,2), axis=1)
        sumX = np.matrix(sumX)
        sumX = sumX.T
        sumXTranspose = sumX.T
        matrixX = np.matrix(X)
        numSamples = np.size(matrixX, axis=0)
        prodX = -2 * matrixX * (matrixX.T)
        sumXAll = np.tile(sumXTranspose, (numSamples,1))
        addX = sumXAll + prodX
        sumXAll2 = np.tile(sumX, (1,numSamples))
        addX2 = sumXAll2 + addX
        addX2 = -1 * addX2
        kernelMatrix = np.exp(addX2 / (2*np.power(self.sigma,2)))
        prob = np.sum(kernelMatrix, axis=0)
        probTranspose = prob.T
        kernelMatrix = kernelMatrix / np.power(probTranspose*prob, self.timeStep)
        prob = np.sqrt(np.sum(kernelMatrix, axis=0))
        probTranspose = prob.T
        kernelMatrix = kernelMatrix / (probTranspose*prob)
        
        print "Perform eigendecomposition..."
        U, S, V = np.linalg.svd(kernelMatrix)
        U1All = np.tile(U[:,0], (1,numSamples))
        U = U / U1All
        reducedDimData = U[:, 1:self.numDim+1]
        self.computedData = np.array(reducedDimData)
        self.results = {'matrix.csv': self.computedData}


