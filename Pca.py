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
from sklearn.decomposition import PCA
import numpy as np


class Pca(Algorithm):
    def __init__(self):
        super(Pca, self).__init__()
        self.inputs = ['numDim']
        self.inputs = ['matrix.csv','features.txt']
        self.outputs = ['matrix.csv', 'features.txt']
        self.name ='Principal Component Analysis'
        self.type = 'Dimension Reduction'
        self.description = 'Performs PCA dimension reduction on the input dataset.'
        self.parameters = ['numDim']
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        with open(filepath['features.txt']['rootdir'] + 'features.txt') as features:
            self.features = features.read().split("\n")
            self.features.pop()
        if int(self.numDim) != len(self.features):
            self.features = [str(x + 1) for x in range(int(self.numDim))]


        print 'pca started...'
        pcaResult = PCA(n_components=int(self.numDim), copy=True, whiten=False)
        self.computedData = pcaResult.fit_transform(self.inputData)
        self.results = {'matrix.csv': self.computedData, 'features.txt': self.features}
