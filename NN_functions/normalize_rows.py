import numpy as np

# GRADED FUNCTION: normalizeRows
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x by its norm
    Argument:
    x -- A numpy matrix of shape (n, m)
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis =
    x_norm = np.linalg.norm(x,ord = 2, axis = 1, keepdims = True)
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###
    return x

x = np.array([
    [0,3,4],
    [1,6,4]
])
normalizeRows(x)
# print("normalizeRows(x) = " + str(normalizeRows(x)))
print("The normalization of \n" + str(x) + " is........ \n" + str(normalizeRows(x)))
