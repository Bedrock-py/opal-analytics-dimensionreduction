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
import numpy as np
from sklearn.decomposition import PCA

class Tsne(Algorithm):
    def __init__(self):
        super(Tsne, self).__init__()
        self.parameters = ['numDim','perplexity']
        self.initialReducedDim = 3
        self.initializeSolution = True
        self.isIter = True
        self.inputs = ['matrix.csv']
        self.outputs = ['matrix.csv']
        self.name ='T-SNE'
        self.type = 'Dimension Reduction'
        self.description = 'Performs T-SNE dimension reduction on the input dataset.'
        self.parameters_spec = [ { "name" : "Dimensions", "attrname" : "numDim", "value" : 2, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Perplexity", "attrname" : "perplexity", "value" : 15, "type" : "input" , "step": 1, "max": 30, "min": 1}  ]

    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        # Check inputs
        if self.inputData.dtype != "float64":
            print "Error: array X should have type float64."
            return -1
        (n, d) = self.inputData.shape; 
        if self.initialReducedDim > d or self.numDim > d:
            print "Error: Both initialReducedDim and reducedDim specified cannot be greater than dimension of the data"
            return
        self.inputData = self.inputData - np.amin(self.inputData)
        self.inputData = self.inputData / np.amax(self.inputData)
    
        # Initialize variables
        self.inputData = self.pca(self.inputData, self.initialReducedDim)
        max_iter = 200
        final_momentum = 0.8
        momentum = 0.5
        mom_switch_iter = 250
        eta = 500
        min_gain = 0.01
        tol = 1e-5
        if self.initializeSolution:
            Y = self.inputData[:, 0:self.numDim]
        else:
            Y = .0001 * np.random.randn(n, self.numDim)
        dY = np.zeros((n, self.numDim))
        iY = np.zeros((n, self.numDim))
        gains = np.ones((n, self.numDim))
    
        # Compute P-values
        P = self.x2p(self.inputData, tol, self.perplexity)
        #P = P + Math.transpose(P);
        P = 0.5 * (P + np.transpose(P))
        P = P / np.sum(P)
        matlabEps = 2.220446049250313e-16;                                 
        P = np.maximum(P, matlabEps)
        P = P * 4; # early exaggeration

        # Run iterations
        iter = 0;
        
        while iter<=max_iter:
          # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q, matlabEps)
            
            # Compute gradient
            PQ = 4 * (P - Q)
            for i in range(n):
                dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (self.numDim, 1)).T * (Y[i,:] - Y), 0)
                
            gains = (gains + 0.2) * (np.sign(dY) != np.sign(iY)) + \
            (gains * 0.8) * (np.sign(dY) == np.sign(iY))
            gains[gains < min_gain] = min_gain;
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
           
            # Perform the update
            if iter == mom_switch_iter:
                momentum = final_momentum
    
            # Stop lying about P-values
            if iter == 100:
                P = P / 4
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log((P+matlabEps) / (Q+matlabEps)))
                print "Iteration ", (iter + 1), ": error is ", C
                #if we are displaying the data, send a new dataset every 10 iterations
            iter += 1
        self.computedData = Y.real
        self.results = {'matrix.csv': self.computedData}

    def Hbeta(self, D, beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the 
         precision of a Gaussian distribution."""
    
        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta);
        sumP = sum(P);
        H = np.log(sumP) + beta * np.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;
    
    def x2p(self, X, tol = 1e-5, perplexity = 15.0):
        """Performs a binary search to get P-values in such a way that each 
        conditional Gaussian has the same perplexity."""
    
        # Initialize some variables
        print "Computing pairwise distances..."
        (n, d) = X.shape;
        sum_X = np.sum(np.square(X), 1);
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
        P = np.zeros((n, n));
        beta = np.ones((n, 1));
        betamin = -np.inf * np.ones((n, 1));
        betamax =  np.inf * np.ones((n, 1));
        logU = np.log(perplexity);
        
        # Loop over all datapoints
        for i in range(n):
            
            # Print progress
            if i % 500 == 0:
                print "Computing P-values for point ", i, " of ", n, "..."
    
            # Compute the Gaussian kernel and entropy for the current precision
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
            (H, thisP) = self.Hbeta(Di, beta[i]);
            
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU;
            tries = 0;
            while np.abs(Hdiff) > tol and tries < 50:
             
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin[i] = beta[i];
               
                    if betamax[i] == np.inf or betamax[i] == -np.inf:
                        beta[i] = beta[i] * 2;
                    else:
                        beta[i] = (beta[i] + betamax[i]) / 2;
                else:
                    betamax[i] = beta[i];
               
                    if betamin[i] == np.inf or betamin[i] == -np.inf:
                        beta[i] = beta[i] / 2;
                    else:
                        beta[i] = (beta[i] + betamin[i]) / 2;
                
                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i]);
                Hdiff = H - logU;
                tries = tries + 1;
            
            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;
    
        # Return final P-matrix
        print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta))
        return P;   

    def pca(self, X, no_dims = 30):
        """Runs PCA on the NxD array X in order to reduce its dimensionality to 
         no_dims dimensions."""

        X = X - np.mean(self.inputData, axis=0);

        print "Preprocessing the data using PCA..."
        (n, d) = X.shape;
#        X = X - Math.tile(Math.mean(X, 0), (n, 1));
        (l, M) = np.linalg.eig(np.dot(X.T, X));
        Y = np.dot(X, M[:,0:no_dims]);
#        tsvd = TruncatedSVD(n_components=no_dims,algorithm='arpack');
#        Y = tsvd.fit_transform(self.inputData);
        
        return Y;

