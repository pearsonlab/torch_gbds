from torch_gbds.GenerativeModel import GenerativeModel, LDS
import torch
from torch.autograd import Variable
import numpy as np
import numpy.testing as npt

def test_generative_model():
    gm = GenerativeModel(None, 5, 10)

    assert gm.Xsamp.size() == torch.Size([5])

def test_LDS():
    mm = LDS({}, 10, 5)

    assert {'x0', 'Q0Chol', 'A', 'RChol', 'QChol', 'Xsamp'}.issubset(dict(mm.named_parameters()).keys())

def test_LDS_sampling():
    mm = LDS({}, 10, 5)

    xx = mm.sampleX(100)
    yy = mm.sampleY()
    [x, y] = mm.sampleXY(100)
    assert xx.size() == torch.Size([100, 10])
    assert yy.size() == torch.Size([5])
    assert x.shape == (100, 10)
    assert y.shape == (100, 5)

def test_LDS_forward():
    mm = LDS({}, 10, 5)

    [x, y] = mm.sampleXY(100)

    x0 = mm.x0.data.numpy()
    yp = mm.Ypred.data.numpy()
    A = mm.A.data.numpy()
    Rinv = mm.Rinv.data.numpy()
    QChol = mm.QChol.data.numpy()
    Q0Chol = mm.Q0Chol.data.numpy()
    Lambda = mm.Lambda.data.numpy()
    Lambda0 = mm.Lambda0.data.numpy()
    N = x.shape[0]

    resy = y - yp
    resx = x[1:] - x[:-1].dot(A.T)
    resx0 = x[0] - x0

    lpdf = -(resy.T.dot(resy) * np.diag(Rinv)).sum()
    lpdf += -(resx.T.dot(resx) * Lambda).sum()
    lpdf += -(resx0.dot(Lambda0).dot(resx0))
    lpdf += N * np.log(Rinv).sum()
    lpdf += (N - 1) * np.linalg.slogdet(Lambda)[1]
    lpdf += np.linalg.slogdet(Lambda0)[1]
    lpdf += -N * (x.shape[1] + y.shape[1]) * np.log(2 * np.pi)
    lpdf *= 0.5

    t_logpdf = mm(Variable(torch.Tensor(x)), Variable(torch.Tensor(y)))
    resX0 = mm.resX0.data.numpy()
    logpdf = t_logpdf.data.numpy()[0]

    assert logpdf < 0
    npt.assert_approx_equal(logpdf, lpdf)
