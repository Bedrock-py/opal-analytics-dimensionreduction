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

from bedrock.analytics.utils import * 
import numpy as np
"""
Created on Mon Oct 14 14:06:01 2013

@author: thuang38
Note: Complex data not yet tested
"""

class Gda(Algorithm):
    def __init__(self):
        super(Gda, self).__init__()
        self.parameters = ['numDim','kernelFun']
        self.param1 = 1
        self.param2 = 0
        self.inputs = ['matrix.csv','truth_labels.csv']
        self.outputs = ['matrix.csv']
        self.name ='Generalized Discriminant Analysis'
        self.type = 'Dimension Reduction'
        self.description = 'Performs generalized discriminant dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Kernel Function", "attrname" : "kernelFun", "value" : "linear", "type" : "select", "options": ['linear','poly','gauss'] }] 
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        self.truthlabels = np.genfromtxt(filepath['truth_labels.csv']['rootdir'] + 'truth_labels.csv', delimiter=',')
    
        # increment label if 0 is included as a label so the labeling starts with 1
        if np.min(self.truthlabels) == 0:
            self.truthlabels = np.array(self.truthlabels)
            self.truthlabels = self.truthlabels + 1
        numSamples = np.size(self.inputData, axis=0)
        numClass = np.max(self.truthlabels)
    
        # sort data according to labels
        sortIndex = np.argsort(self.truthlabels)
        self.truthlabels = np.sort(self.truthlabels)
        self.inputData = self.inputData[sortIndex,:]
    
        # compute kernel matrix
        gramMatrix = self.gram(self.inputData, self.inputData, self.kernelFun, self.param1, self.param2)
    
        # Perform eigenvector decomposition of kernel matrix (Kc = P * gamma * P')
        gramMatrix[np.isnan(gramMatrix)] = 0
        gramMatrix[np.isinf(gramMatrix)] = 0
    
        eigValue, eigVector = np.linalg.eig(gramMatrix)
    
        if np.size(eigVector, axis=1) < numSamples:
            print "Error: Singularities in kernel matrix prevent solution"
            #fix below to appropriate error handling
            return
    
        # sort eigenvalues and vectors in descending order     
        sortIndex = np.argsort(eigValue)
        sortIndex = sortIndex[::-1]
        eigValue = eigValue[sortIndex]
        eigVector = eigVector[:,sortIndex]
    
        # choose a subset of eigValue and eigVector to compute
        eigValue = eigValue[0:self.numDim]
        eigVector = eigVector[:,0:self.numDim]
        
        # construct diagonal block matrix W
        classSizeStore = np.zeros(numClass)
        diagonalMatrix = np.zeros((numSamples, numSamples))
        for classNumber in range(0, np.int(numClass)):
            # find the size of each class
            classSizeStore[classNumber] = len(np.where(self.truthlabels == classNumber+1)[0])
    
        countSize = 0    
        for classNumber in range(0, np.int(numClass)):
            classMatrix = np.ones((classSizeStore[classNumber], classSizeStore[classNumber])) \
            / classSizeStore[classNumber]
            diagonalMatrix[countSize:countSize+classSizeStore[classNumber], \
            countSize:countSize+classSizeStore[classNumber]] = classMatrix
            countSize = classSizeStore[classNumber] + countSize
        
        # determine target dimensionality of data     
        reducedDimOrig = self.numDim
        reducedDim = min([reducedDimOrig, numClass]) 
        if reducedDimOrig > reducedDim:
            print "Warning: Target dimensionality reduced to %d" % reducedDim
        
        # perform eigendecomposition of matrix (eigVector.T*diagonalMatrix*eigVector)     
        diagonalMatrix = np.matrix(diagonalMatrix)
        newMatrix = (eigVector.T) * diagonalMatrix * eigVector
        eigValueNew, eigVectorNew = np.linalg.eig(newMatrix)

        # compute final embedding mappedX
        eigVectorNew = np.matrix(eigVectorNew)
        mappedX = eigVector * np.matrix(np.diag(1/eigValue)) * eigVectorNew

        # normalize embedding
        for dimNumber in range(0, self.numDim):      
            denomNormalize = np.sqrt((mappedX[:,dimNumber].T)*gramMatrix*mappedX[:,dimNumber])
            mappedX[:,dimNumber] = mappedX[:,dimNumber] / denomNormalize
        
        mappedX = mappedX.T
        mappedX = mappedX * gramMatrix
        #convert datatype to float and data structure to array
        self.computedData = np.array(mappedX.T, dtype=float)
        self.results = {'matrix.csv': self.computedData}

    def gram(self, X1, X2, kernelFun="linear", param1=1, param2=3):
        # This function computes the Gram matrix
        # assume X1 and X2 are matrices
        X1 = np.matrix(X1)
        X2 = np.matrix(X2)
        # check size
        if np.size(X1, axis=1) != np.size(X2, axis=1):
            print "Error: Dimensionality of both datasets should be equal"
            return   
        if kernelFun == 'linear':
            # linear kernel
            gramMatrix = X1 * (X2.T)
        elif kernelFun == 'gauss':
            # gaussian kernel
            distance = L2_distance.L2_distance(X1.T, X2.T)
            distanceSquare = np.power(distance, 2)
            gramMatrix = np.exp(-1*(distanceSquare / (2*pow(param1,2))))
        elif kernelFun == 'poly':
            # polynomial kernel
            X1TimesX2 = X1 * (X2.T)
            gramMatrix = np.power(X1TimesX2+param1, param2)
        else:
            print "Error: Unknown kernel function"
            return
        return gramMatrix

    def L2_distance(self, X1, X2, df="False"):
        # This function computes Euclidean distance matrix

        # say X1 and X2 are matrices
        X1 = np.matrix(X1)
        X2 = np.matrix(X2)
        
        # check size
        if np.size(X1, axis=0) != np.size(X2, axis=0):
            print "X1 and X2 should be of same dimensionality"
            
        X1Vector = X1.flatten(1)
        X2Vector = X2.flatten(1)
        if (np.any(np.iscomplex(X1Vector)) or np.any(np.iscomplex(X2Vector))):
            print "Warning: running L2_distance.m with imaginary numbers. Results may be off"
            
        if np.size(X1, axis=0) == 1:
            numSamples1 = np.size(X1, axis=1)
            numSamples2 = np.size(X2, axis=1)
            addZeros1 = np.zeros((1, numSamples1))
            addZeros2 = np.zeros((1, numSamples2))
            X1 = np.bmat('X1; addZeros1')
            X2 = np.bmat('X2; addZeros2')
            
        sumX1 = sum(np.multiply(X1, X1), axis=0)
        sumX2 = sum(np.multiply(X2, X2), axis=0)
        X1TimesX2 = (X1.T) * X2
        
        partX1 = np.tile(sumX1.T, (1, np.size(sumX2,axis=1)))
        partX2 = np.tile(sumX2, (np.size(sumX1,axis=1), 1))
        distance = np.sqrt(partX1 + partX2 - (2*X1TimesX2))
            
        # make sure result is all real
        distance = np.real(distance)    
        
        # for 0 on the diagonal?
        if (df == 'True'):
            numRows = np.size(distance, axis=0)
            numCols = np.size(distance, axis=1)
            identityMatrix =  np.eye(numRows, numCols)
            identityMatrixInv = 1 - identityMatrix
            distance = np.multiply(distance, identityMatrixInv)
        
        return distance
        

