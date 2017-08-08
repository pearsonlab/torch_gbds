"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here: https://github.com/earcher/vilds/blob/master/code/GenerativeModel.py
"""
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np

class GenerativeModel(torch.nn.Module):
    """
    Interface class for generative time-series models
    """
    def __init__(self,GenerativeParams,xDim,yDim):
        super(GenerativeModel, self).__init__()

        self.xDim = xDim
        self.yDim = yDim

        self.Xsamp = Parameter(torch.zeros(xDim))

    def evaluateLogDensity(self):
        """
        Return a theano function that evaluates the density of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        """
        Return parameters of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        """
        generates joint samples
        """
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"

class LDS(GenerativeModel):
    """
    Gaussian latent LDS with (optional) NN observations:
    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1), Q * Q')
    y(t) ~ N(NN(x(t)), R * R')
    For a Kalman Filter model, choose the observation network, NN(x), to be
    a one-layer network with a linear output. The latent state has dimensionality
    n (parameter "xDim") and observations have dimensionality m (parameter "yDim").
    Inputs:
    (See GenerativeModel abstract class definition for a list of standard parameters.)
    GenerativeParams  -  Dictionary of LDS parameters
                           * A     : [n x n] linear dynamics matrix; should
                                     have eigenvalues with magnitude strictly
                                     less than 1
                           * QChol : [n x n] square root of the innovation
                                     covariance Q
                           * Q0Chol: [n x n] square root of the innitial innovation
                                     covariance
                           * RChol : [n x 1] square root of the diagonal of the
                                     observation covariance
                           * x0    : [n x 1] mean of initial latent state
                           * NN: module specifying network transforming x to
                                 the mean of y (input dim: n, output dim: m)
    """
    def __init__(self, GenerativeParams, xDim, yDim):

        super(LDS, self).__init__(GenerativeParams,xDim,yDim)

        # parameters
        if 'A' in GenerativeParams:
            self.A = Parameter(torch.Tensor(GenerativeParams['A']))
        else:
            # TBD:MAKE A BETTER WAY OF SAMPLING DEFAULT A
            self.A = Parameter(torch.eye(xDim).mul_(0.5))

        if 'QChol' in GenerativeParams:
            self.QChol = Parameter(torch.Tensor(GenerativeParams['QChol']))
        else:
            self.QChol = Parameter(torch.eye(xDim))

        if 'Q0Chol' in GenerativeParams:
            self.Q0Chol = Parameter(torch.Tensor(GenerativeParams['Q0Chol']))
        else:
            self.Q0Chol = Parameter(torch.eye(xDim))

        if 'RChol' in GenerativeParams:
            self.RChol = Parameter(torch.Tensor(GenerativeParams['RChol']))
        else:
            self.RChol = Parameter(torch.randn(yDim).div_(10))

        if 'x0' in GenerativeParams:
            self.x0 = Parameter(torch.Tensor(GenerativeParams['x0']))
        else:
            self.x0 = Parameter(torch.zeros(xDim))

        if 'NN' in GenerativeParams:
            self.add_module('NN', GenerativeParams['NN'])
        else:
            self.add_module('NN', torch.nn.Linear(xDim, yDim))


        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1/self.RChol.pow(2)
        self.Lambda = torch.inverse(torch.matmul(self.QChol, self.QChol.t()))
        self.Lambda0 = torch.inverse(torch.matmul(self.Q0Chol, self.Q0Chol.t()))

        # Call the neural network output a rate, basically to keep things consistent with the PLDS class
        self.rate = self.NN(self.Xsamp)

    def sampleX(self, N):
        """
        Sample latent state from the generative model. Return as a torch tensor.
        """
        _x0 = self.x0.data
        _Q0Chol = self.Q0Chol.data
        _QChol = self.QChol.data
        _A = self.A.data

        norm_samp = torch.normal(torch.zeros(N, self.xDim))
        x_vals = torch.zeros([N, self.xDim])

        x_vals[0] = _x0 + norm_samp[0].matmul(_Q0Chol.t())

        for ii in range(N-1):
            x_vals[ii+1] = x_vals[ii].matmul(_A.t()) + norm_samp[ii+1].matmul(_QChol.t())

        return x_vals

    def sampleY(self):
        """ Return a torch tensor sample from the generative model. """
        eps = torch.normal(torch.zeros([self.yDim]))
        return self.rate.data + torch.matmul(eps, torch.diag(self.RChol.data))

    def sampleXY(self, N):
        """ Return numpy samples from the generative model. """
        X = Variable(self.sampleX(N), requires_grad=False)
        eps = torch.randn([X.size(0), self.yDim])
        #  np.random.randn(X.shape[0],self.yDim).astype(theano.config.floatX)
        _RChol = self.RChol.data
        # Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand,np.diag(_RChol).T)
        Y = self.NN(X).data + torch.matmul(eps, torch.diag(_RChol))
        return [X.data.numpy(),Y.numpy()]

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)

    def evaluateLogDensity(self,X,Y):
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:(X.shape[0]-1)],self.A.T)
        resX0 = X[0]-self.x0

        LogDensity  = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() - (0.5*T.dot(resX.T,resX)*self.Lambda).sum() - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T)
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0))  - 0.5*(self.xDim + self.yDim)*np.log(2*np.pi)*Y.shape[0]

        return LogDensity