import numpy as np

def softmax(x):
    """
    Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (n, m).
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1,
    # sum of the exponential in each row of the matrix
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use n
    s = x_exp / x_sum
    ### END CODE HERE ###
    return s

x = np.array([
[9, 2, 5, 0, 0],
[7, 5, 0, 0 ,0]])

softmax(x)
print("softmax(x) = " + str(softmax(x)))
