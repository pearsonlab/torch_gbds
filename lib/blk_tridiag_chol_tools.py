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

def blk_chol_inv(A, B, b, lower=True, transpose=False):
    '''
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix
    b - [T x n] tensor

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)
    Outputs:
    x - solution of Cx = b
    '''
    def _step(acc, inputs):
        x = acc
        A, B, b = inputs
        return tf.matrix_solve(A, b - tf.matmul(B, x))

    if transpose:
        A = tf.transpose(A, perm=[0, 2, 1])
        B = tf.transpose(B, perm=[0, 2, 1])
    if lower:
        x0 = tf.matrix_solve(A[0], b[0])
        X = tf.scan(_step, [A[1:], B, b[1:]], initializer=x0)
        X = tf.concat([tf.expand_dims(x0,0), X], 0)
    else:
        # xN = Tla.matrix_inverse(A[-1]).dot(b[-1])
        # def upper_step(Akm1, Bkm1, bkm1, xk):
        #     return Tla.matrix_inverse(Akm1).dot(bkm1-(Bkm1).dot(xk))
        # X = theano.scan(fn = upper_step, sequences=[A[:-1][::-1], B[::-1], b[:-1][::-1]], outputs_info=[xN])[0]
        # X = T.concatenate([T.shape_padleft(xN), X])[::-1]
        pass
    return X
