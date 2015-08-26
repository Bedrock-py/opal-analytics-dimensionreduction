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
from sklearn.lda import LDA
import numpy as np

class Lda(Algorithm):
    def __init__(self):
        super(Lda, self).__init__()
        self.parameters =['numDim','truthlabels']
        self.inputs = ['matrix.csv','assignments.csv']
        self.outputs = ['matrix.csv']
        self.name ='Linear Discriminant Analysis'
        self.type = 'Dimension Reduction'
        self.description = 'Performs linear discriminant dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1} ]

    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        uniqueLabels = np.unique(self.truthlabels)
        uniqueLabelsLength = len(uniqueLabels)
        if self.numDim >= uniqueLabelsLength:
            self.numDim = uniqueLabelsLength - 1
        ldaResult = LDA(n_components=self.numDim)
        self.truthlabels = np.array(self.truthlabels)
        self.computedData = ldaResult.fit_transform(self.inputData, y=self.truthlabels)
        self.results = {'matrix.csv': self.computedData}
