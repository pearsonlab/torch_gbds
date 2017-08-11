from torch_gbds.RecognitionModel import RecognitionModel, SmoothingTimeSeries
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import numpy.testing as npt

# shared testing data
prec = np.float32
npA = np.array([[1, .9], [.9, 4]]).astype(prec)
npB = (.01 * np.array([[2, 7], [7, 4]])).astype(prec)
npC = (1. * np.array([[3, 0.], [0, 1]])).astype(prec)
npD = (.01 * np.array([[7, 2], [9, 3]])).astype(prec)
npE = (.01 * np.array([[2, 0], [4, 3]])).astype(prec)
npF = (.01 * np.array([[1, 0], [2, 7]])).astype(prec)
npG = (.01 * np.array([[3, 0], [8, 1]])).astype(prec)

npZ = np.array([[0, 0], [0, 0]]).astype(prec)

npb = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(prec)

lowermat = np.bmat([[npF,   npZ,   npZ,   npZ],
                    [npB.T, npC,   npZ,   npZ],
                    [npZ,   npD.T, npE,   npZ],
                    [npZ,   npZ,   npB.T, npG]])
lowermat = np.array(lowermat)
cholmat = lowermat.dot(lowermat.T)

class ID(torch.nn.Module):
    """
    Dummy class for turning a variable into a module.
    """
    def __init__(self, V):
        super(ID, self).__init__()
        self.V = Parameter(V)

    def forward(self, data):
        return self.V


def test_recognition_model():
    rm = RecognitionModel(None, 5, 10)

    assert rm.xDim == 5
    assert rm.yDim == 10

def test_smoothing_time_series_recognition():
    Lambda = torch.Tensor(np.array([npF, npC, npE, npG]))
    LambdaX = torch.Tensor(np.array([npB.T, npD.T, npB.T]))
    mu = np.array([2, 3], dtype=prec)
    Mu = torch.Tensor(mu)
    Data = torch.zeros(10, 2)

    RecognitionParams = ({'NN_Mu': ID(Mu),
                          'NN_Lambda': ID(Lambda),
                          'NN_LambdaX': ID(LambdaX)})

    rm = SmoothingTimeSeries(RecognitionParams, Data, 2, 5)
