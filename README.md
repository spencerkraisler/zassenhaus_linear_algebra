# zassenhaus_linear_algebra
An implementation of Zassenhaus algorithm, in the context of linear algebra. Zassenhaus algorithm calculates a basis for the intersection of two subspaces. The subspaces are represented as the span of the columns of the matrices.

The algorithm is in zassenhaus.py. It uses numpy and sympy for computing the reduced row echelon form. Here, U and V are the basis matrices (and hence have linearly independent columns) of the subspaces you wish to intersect. Errors are raised if those conditions are not met.
