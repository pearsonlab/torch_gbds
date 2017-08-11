import numpy as np
import torch
import numpy.testing as npt

from torch.autograd import Variable
import torch_gbds.lib.sym_blk_tridiag_inv as sym

# Build a block tridiagonal matrix
prec = np.float32
npA = np.mat('1 6; 6 4', dtype=prec)
npB = np.mat('2 7; 7 4', dtype=prec)
npC = np.mat('3 9; 9 1', dtype=prec)
npD = np.mat('7 2; 9 3', dtype=prec)
npZ = np.mat('0 0; 0 0', dtype=prec)

# a 2x2 block tridiagonal matrix with 4x4 blocks
fullmat = np.bmat([[npA,     npB, npZ,   npZ],
                   [npB.T,   npC, npD,   npZ],
                   [npZ,   npD.T, npC,   npB],
                   [npZ,     npZ, npB.T, npC]])

alist = [npA, npC, npC, npC]
blist = [npB, npD, npB]
AAin = Variable(torch.Tensor(np.array(alist)))
BBin = Variable(torch.Tensor(np.array(blist)))

def test_compute_sym_blk_tridiag():
    D, OD, S = sym.compute_sym_blk_tridiag(AAin, BBin)

    invmat = np.linalg.inv(fullmat)

    benchD = [invmat[i:(i+2),i:(i+2)] for i in range(0, invmat.shape[0], 2)]
    benchOD = [invmat[(i+2):(i+4),i:(i+2)] for i in range(0, invmat.shape[0] - 2, 2)]
    npt.assert_allclose(np.array(benchD), D.data.numpy(), atol=1e-6)
    npt.assert_allclose(np.array(benchOD), OD.data.numpy(), atol=1e-6)

def test_compute_sym_blk_tridiag_inv_b():
    D, OD, S = sym.compute_sym_blk_tridiag(AAin, BBin)

    # test solving the linear sysstem Ay=b
    # now let's implement the solver (IE, we want to solve for y in Ay=b)
    npb = np.asmatrix(np.arange(4*2, dtype=prec).reshape((4,2)))
    b = Variable(torch.Tensor(npb))

    y = sym.compute_sym_blk_tridiag_inv_b(S,D,b)

    ybench = np.linalg.pinv(fullmat).dot(npb.reshape(8,1))
    npt.assert_allclose(y.data.numpy().flatten(), np.array(ybench).flatten(), atol=1e-6)

def test_compute_sym_blk_tridiag_inds():
    D, OD, S = sym.compute_sym_blk_tridiag(AAin, BBin)
    the_blocks = Variable(torch.Tensor(np.array([npA, npB, npC, npD])))

    iia = torch.from_numpy(np.array([0, 2, 2, 2]))
    iib = torch.from_numpy(np.array([1, 3, 1]))
    Dii, ODii, _ = sym.compute_sym_blk_tridiag(the_blocks, the_blocks, iia, iib)

    npt.assert_allclose(D[0].data.numpy(), Dii[0].data.numpy())
    npt.assert_allclose(OD[0].data.numpy(), ODii[0].data.numpy())
