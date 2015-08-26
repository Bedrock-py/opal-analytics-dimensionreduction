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
from sklearn.decomposition import KernelPCA
import numpy as np

class KernelPca(Algorithm):
    def __init__(self):
        super(KernelPca, self).__init__()
        self.parameters = ['numDim','function','degree','gamma','coeff']
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='Kernel PCA'
        self.type = 'Dimension Reduction'
        self.description = 'Performs Kernel PCA dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input", "step": 1, "max": 15, "min": 1 }, 
            { "name" : "Degree", "attrname" : "degree", "value" : 2, "type" : "input", "step": 1, "max": 15, "min": 1 },
            { "name" : "Gamma", "attrname" : "gamma", "value" : 0.5, "type" : "input", "step": 0.1, "max": 15, "min": 0 },
            { "name" : "Coefficient", "attrname" : "coeff", "value" : 0, "type" : "input", "step": 1, "max": 15, "min": 0 },
            { "name" : "Function", "attrname" : "function", "value" : "rbf", "type" : "select", "options": ['rbf', 'linear','poly','sigmoid','cosine','precomputed'] }] 
        
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        kernelPcaResult = KernelPCA(n_components=self.numDim, kernel=self.function, degree=self.degree, gamma=self.gamma)
        self.computedData = kernelPcaResult.fit_transform(self.inputData)
        self.results = {'matrix.csv': self.computedData}
