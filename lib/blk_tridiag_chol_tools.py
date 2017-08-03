"""
Tensorflow implementation of functions to perform the Cholesky factorization
of a block tridiagonal matrix. Ported from Evan Archer's implementation
here: https://github.com/earcher/vilds/blob/master/code/lib/blk_tridiag_chol_tools.py
"""

import tensorflow as tf
import numpy as np

def _compute_chol(acc, inputs):
    """
    Compute the Cholesky decomposition of a symmetric block tridiagonal matrix.
    acc is the output of the previous loop
    inputs is a tuple of inputs
    """
    L, _ = acc
    A, B = inputs

    C = tf.transpose(tf.matrix_solve(L, B))
    D = A - tf.matmul(C, tf.transpose(C))
    L = tf.cholesky(D)

    return [L, C]

def blk_tridiag_chol(A, B):
    """
    Compute the cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block
        off-diagonal matrix
    Outputs:
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky
    """
    L = tf.cholesky(A[0])
    C = tf.zeros_like(B[0])

    R = tf.scan(_compute_chol, [A[1:], B], initializer=[L, C])
    R[0] = tf.concat([tf.expand_dims(L, 0), R[0]], 0)
    return R

if __name__ == '__main__':
    import numpy.testing as npt

    npA = np.array([[1, .9], [.9, 4]]).astype('float32')
    npB = (.01 * np.array([[2, 7], [7, 4]])).astype('float32')
    npC = np.array([[3, 0.], [0, 1]]).astype('float32')
    npD = (.01 * np.array([[7, 2], [9, 3]])).astype('float32')
    npE = (.01 * np.array([[2, 0], [4, 3]])).astype('float32')
    npF = (.01 * np.array([[1, 0], [2, 7]])).astype('float32')
    npG = (.01 * np.array([[3, 0], [8, 1]])).astype('float32')

    npZ = np.array([[0, 0], [0, 0]]).astype('float32')

    lowermat = np.bmat([[npF,   npZ,   npZ,   npZ],
                        [npB.T, npC,   npZ,   npZ],
                        [npZ,   npD.T, npE,   npZ],
                        [npZ,   npZ,   npB.T, npG]])
    cholmat = lowermat.dot(lowermat.T)

    def test_compute_chol():
        L = np.linalg.cholesky(npA)
        C = npZ
        A = npF
        B = npB

        CC = np.linalg.solve(L, B).T
        D = A - CC.dot(CC.T)
        LL = np.linalg.cholesky(D)

        LLL, CCC = _compute_chol([tf.constant(L), tf.constant(C)],
                                 [tf.constant(A), tf.constant(B)])

        with tf.Session() as sess:
            npt.assert_allclose(LLL.eval(), LL)
            npt.assert_allclose(CCC.eval(), CC, rtol=1e-5)

        return

    def np_compute_chol(acc, inputs):
        """
        Compute the Cholesky decomposition of a symmetric block tridiagonal matrix.
        acc is the output of the previous loop
        inputs is a tuple of inputs
        """
        L, _ = acc
        A, B = inputs

        C = np.linalg.solve(L, B).T
        D = A - C.dot(C.T)
        L = np.linalg.cholesky(D)

        return [L, C]

    def test_blk_tridiag_chol():
        alist = [cholmat[i:(i+2),i:(i+2)] for i in range(0, cholmat.shape[0], 2)]
        blist = [cholmat[(i+2):(i+4),i:(i+2)].T for i in range(0, cholmat.shape[0] - 2, 2)]

        theDiag = tf.stack(list(map(tf.constant, alist)))
        theOffDiag = tf.stack(list(map(tf.constant, blist)))

        R = blk_tridiag_chol(theDiag, theOffDiag)

        with tf.Session() as sess:
            R0, R1 = R[0].eval(), R[1].eval()

        for (x, y) in zip(R0, [npF, npC, npE, npG]):
            npt.assert_allclose(x, y, atol=1e-4)

        for (x, y) in zip(R1, [npB.T, npD.T, npB.T]):
            npt.assert_allclose(x, y, atol=1e-4)

        return

    test_list = [test_compute_chol, test_blk_tridiag_chol]
    for test in test_list:
        test()
