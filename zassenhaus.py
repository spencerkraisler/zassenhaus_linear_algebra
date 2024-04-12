"""Created by Spencer Kraisler in 2024."""

import numpy as np
import sympy as sp


def zassenhaus(U: np.array, V: np.array) -> np.array:
    """Zassenhaus algorithm calculates a basis for the intersection of two
    subspaces. The subspaces are represented as the span of the columnspaces of
    U and V.

    U, V: (np.ndarray) Matrices of linearly independent columns with the same 
        column dim n.

    Returns:
    basis: (np.array) Matrix of linearly independent columns with column dim n 
        and row dim equal to the dimension of the intersection subspace of 
        span(U) and span(V).

    The columns of U and V must be a basis for their columnspace, and hence must
    be linearly independent. This implies that U, V must be full rank (i.e.
    rank(U) = min(U.shape)). Otherwise, an error is raised. An error is also 
    raised if they do not have the same column dim.
    
    """
    n = U.shape[0]
    if n <= 1:
        raise ValueError("Col dim must be at least 2.")
    k = np.linalg.matrix_rank(U) + \
        np.linalg.matrix_rank(V) - \
        np.linalg.matrix_rank(np.concatenate([U, V], axis=1))
    if k == 0:
        return np.zeros((n,1))

    
    if V.shape[0] != n:
        raise ValueError("Basis matrices do not have equal column dim.")
    if np.linalg.matrix_rank(U) != min(U.shape) or \
        np.linalg.matrix_rank(V) !=  min(V.shape):
        raise ValueError("Basis matrices U and V are not full rank.")
    
    block_matrix = np.block([
        [U.T, U.T],
        [V.T, np.zeros((V.shape[1], n))]
    ])
    
    # Convert block_matrix to echelon form. 
    sp_block_matrix = sp.Matrix(block_matrix)
    sp_block_matrix_ef = sp_block_matrix.rref()[0]
    block_matrix_ef = np.array(sp_block_matrix_ef).astype('float')
    sums = np.sum(np.abs(block_matrix_ef[:,:n]), axis=1)
    first_zero_idx = np.where(np.abs(sums) <= 1e-5)[0][0]
    last_zero_idx = np.where(np.abs(sums) <= 1e-5)[0][-1]
    W = block_matrix_ef[first_zero_idx:last_zero_idx + 1, n:].T
    return W
