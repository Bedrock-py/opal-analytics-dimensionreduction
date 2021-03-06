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

from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding, MDS
from bedrock.analytics.utils import * 
import numpy as np


class LocalLinear(Algorithm):
    def __init__(self):
        super(LocalLinear, self).__init__()
        self.parameters = ['numDim','neighbors']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Local Linear Embedding'
        self.type = 'Dimension Reduction'
        self.description = 'Performs local linear embedding dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Neighbors", "attrname" : "neighbors", "value" : 30, "type" : "input" , "step": 1, "max": 1000, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        #using eigen_solver='auto' causes the algorithm to crash
        localLinearEmbeddingResult = LocallyLinearEmbedding(n_neighbors=self.neighbors, n_components=self.numDim, eigen_solver='dense', method="ltsa", neighbors_algorithm='auto')
        self.computedData = localLinearEmbeddingResult.fit_transform(self.inputData)        
        self.results = {'matrix.csv': self.computedData}
