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
from analytics.utils import Algorithm 

import time, os
import numpy as np


class Mds(Algorithm):
    def __init__(self):
        super(Mds, self).__init__()
        self.parameters = ['numDim']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Multidimensional Scaling'
        self.type = 'Dimension Reduction'
        self.description = 'Performs multidimensional scaling dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1} ]
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
            
        print 'mds started...'
        multidimensionalScalingResult = MDS(n_components=self.numDim)
        self.computedData = multidimensionalScalingResult.fit_transform(self.inputData)
        self.results = {'matrix.csv': self.computedData}
