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

from analytics.utils import Algorithm 
from sklearn.manifold import SpectralEmbedding
import numpy as np

class Spectral(Algorithm):
    def __init__(self):
        super(Spectral, self).__init__()
        self.parameters=['numDim']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Spectral Embedding'
        self.type = 'Dimension Reduction'
        self.description = 'Performs spectral embedding dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1} ]

    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        spectralEmbeddingResult = SpectralEmbedding(n_components=self.numDim, affinity='rbf', eigen_solver='arpack')
        self.computedData = spectralEmbeddingResult.fit_transform(self.inputData)
        self.results = {'matrix.csv': self.computedData}

