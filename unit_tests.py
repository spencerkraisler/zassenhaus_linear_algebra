import numpy as np
from zassenhaus import zassenhaus

def test_for_correct_output() -> None:
    U = np.array([
        [1,-1,0,1],
        [0,0,1,-1]
    ]).T
    V = np.array([
        [5,0,-3,3],
        [0,5,-3,-2]
    ]).T
    W = zassenhaus(U, V)
    assert W.shape[1] == 1, "Incorrect column dim for W."
    assert W.shape[0] == 4, "Incorrect row dim for W."
    assert np.linalg.matrix_rank(
        np.concatenate([W, np.array([[1,-1,0,1]]).T])
    ) == 1, "Incorrect basis."

    print("Pass: test_for_correct_output")

def test_for_correct_output_from_random_input() -> None:
    n = 20
    for _ in range(10):
        m1 = np.random.randint(1,n)
        m2 = np.random.randint(1,n)
        U = np.random.randn(n,m1)
        V = np.random.randn(n,m2)
        k = np.linalg.matrix_rank(U) + \
            np.linalg.matrix_rank(V) - \
            np.linalg.matrix_rank(np.concatenate([U, V], axis=1)) # correct rank
        W = zassenhaus(U, V)
        assert np.linalg.matrix_rank(W) == k, "Incorrect column dim for W."

        UW = np.concatenate([U, W], axis=1)
        VW = np.concatenate([V, W], axis=1)
        assert np.linalg.matrix_rank(UW) == m1, "W is not a subspace of U."
        assert np.linalg.matrix_rank(VW) == m2, "W is not a subspace of V."

    print("Pass: test_for_correct_output_from_random_input")

def test_for_trivial_intersection() -> None:
    U = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0]
    ]).T
    V = np.array([
        [0,0,1,0,0,0],
        [0,0,0,1,0,0]
    ]).T
    W = zassenhaus(U,V)
    assert np.linalg.norm(W) <= 1e-5, "W is non-zero."

    print("Pass: test_for_trivial_intersection")

def test_n_at_least_2() -> None:
    U = np.array([[2]])
    V = np.array([[1]])
    try:
        zassenhaus(U, V) 
    except ValueError:
        print("Pass: test_n_at_least_2")

def test_rank_error_is_thrown():
    U = np.array([
        [1,-1,0,1],
        [4,-4,0,4]
    ]).T
   
    V = np.array([
        [5,0,-3,3],
        [0,5,-3,-2]
    ]).T 
   
    try:
        zassenhaus(U, V) 
    except ValueError:
        print("Pass: test_rank_error_is_thrown")

def test_col_dim_error_is_thrown():
    U = np.array([
        [1,-1,0,1],
        [4,-4,0,0]
    ]).T
   
    V = np.array([
        [5,0,-3],
        [0,5,-3]
    ]).T

    try:
        zassenhaus(U, V) 
    except ValueError:
        print("Pass: test_col_dim_error_is_thrown") 
   
test_for_correct_output()
test_for_trivial_intersection()
test_for_correct_output_from_random_input()
test_rank_error_is_thrown()
test_col_dim_error_is_thrown()
test_n_at_least_2()