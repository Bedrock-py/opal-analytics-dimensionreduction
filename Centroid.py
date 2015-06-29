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

import time, os
import numpy as np

class Centroid(Algorithm):
    def __init__(self):
        super(Centroid, self).__init__()
        self.parameters = ['numDim']
        self.inputs = ['matrix.csv','truth_labels.csv']
        self.outputs = ['matrix.csv']
        self.name ='Centroid'
        self.type = 'Dimension Reduction'
        self.description = 'Performs centroid dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        print 'centroid started...'
        # perform Centroid method
    
        # transpose data
        X = self.inputData.T
    
        self.truthlabels = np.genfromtxt(filepath['truth_labels.csv']['rootdir'] + 'truth_labels.csv', delimiter=',')

        # increment label if 0 is included as a label so the labeling starts with 1
        if min(self.truthlabels) == 0:
            self.truthlabels = self.truthlabels + 1
            
        uniqueLabels = np.unique(self.truthlabels)
        uniqueLabelsLength = len(uniqueLabels)
        
        
        # initialize the Centroid matrix
        numVariables = np.size(X, axis=0)
        centroidMatrix = np.zeros((numVariables, uniqueLabelsLength))
        for idx in range(0, uniqueLabelsLength):
            # assume the matrix and labels are in an array
#            indexStore = np.where(self.truthlabels == idx+1)[0]
            indexStore = np.where(self.truthlabels == uniqueLabels[idx])[0]
            centroidPart = np.sum(X[:,indexStore], axis=1)
            centroidPart = centroidPart / len(indexStore)
            centroidMatrix[:, idx] = centroidPart
        reducedDimData = np.linalg.lstsq(centroidMatrix, X)
        reducedDimDataDim = np.min([uniqueLabelsLength, self.numDim])
        self.numDim = reducedDimDataDim
        #ADD ERROR CATCHING FOR USER INPUT NUMBER OF DIMENSIONS
        reducedDimData = reducedDimData[0][0:reducedDimDataDim,:]
        self.computedData = reducedDimData.T
        self.results = {'matrix.csv': self.computedData}
