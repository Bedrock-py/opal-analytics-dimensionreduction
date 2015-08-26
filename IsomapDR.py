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
from sklearn.manifold import Isomap
import numpy as np

class IsomapDR(Algorithm):
    def __init__(self):
        super(IsomapDR, self).__init__()
        self.parameters = ['numDim','neighbors']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Isomap'
        self.type = 'Dimension Reduction'
        self.description = 'Performs isomap dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Neighbors", "attrname" : "neighbors", "value" : 30, "type" : "input" , "step": 1, "max": 1000, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        isomapResult = Isomap(n_neighbors=self.neighbors, n_components=self.numDim, neighbors_algorithm='auto', path_method='D')
        self.computedData = isomapResult.fit_transform(self.inputData)
        self.results = {'matrix.csv': self.computedData}




