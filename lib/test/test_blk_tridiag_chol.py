import numpy as np
import torch
import numpy.testing as npt

from torch.autograd import Variable
import torch_gbds.lib.blk_tridiag_chol_tools as blk

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

def test_blk_tridiag_chol():
    alist = [cholmat[i:(i+2),i:(i+2)] for i in range(0, cholmat.shape[0], 2)]
    blist = [cholmat[(i+2):(i+4),i:(i+2)].T for i in range(0, cholmat.shape[0] - 2, 2)]

    A = Variable(torch.Tensor(np.stack(alist)))
    B = Variable(torch.Tensor(np.stack(blist)))

    R = blk.blk_tridiag_chol(A, B)

    npt.assert_allclose(R[0].data.numpy(),
        np.stack([npF, npC, npE, npG]), atol=1e-4, rtol=1e-5)

    npt.assert_allclose(R[1].data.numpy(),
        np.stack([npB.T, npD.T, npB.T]), atol=1e-4, rtol=1e-5)


def test_blk_chol_inv():
    xl = np.linalg.solve(lowermat, np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(prec))
    x = np.linalg.solve(cholmat, np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(prec))

    alist = [npF, npC, npE, npG]
    blist = [npB.T, npD.T, npB.T]
    A = Variable(torch.Tensor(np.stack(alist)))
    B = Variable(torch.Tensor(np.stack(blist)))
    b = Variable(torch.Tensor(npb))

    # now solve C * x = b by inverting one Cholesky factor of C at a time
    ib = blk.blk_chol_inv(A, B, b)
    tx = blk.blk_chol_inv(A, B, ib, lower=False, transpose=True)

    npt.assert_allclose(ib.data.numpy().flatten(), xl, atol=1e-5, rtol=1e-4)
    npt.assert_allclose(tx.data.numpy().flatten(), x, atol=1e-5, rtol=3e-3)

def test_blk_chol_mtimes():
    alist = [npF, npC, npE, npG]
    blist = [npB.T, npD.T, npB.T]
    A = Variable(torch.Tensor(np.stack(alist)))
    B = Variable(torch.Tensor(np.stack(blist)))
    x = Variable(torch.Tensor(npb))

    b = lowermat.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(prec))
    bt = lowermat.T.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(prec))

    tb = blk.blk_chol_mtimes(A, B, x)
    tbt = blk.blk_chol_mtimes(A, B, x, lower=False, transpose=True)
    npt.assert_allclose(tb.data.numpy().flatten(), b, atol=1e-5)
    npt.assert_allclose(tbt.data.numpy().flatten(), bt, atol=1e-5)
