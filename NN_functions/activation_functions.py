import numpy as np

def sigmoid(x): # same as logistic activation function
    return 1/(1+np.exp(-x))

x = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])
# print(sigmoid(1.0))

# https://www.youtube.com/watch?v=z4o0R4tMjw0
def relu(y):
    #could also be np.maximum(y, 0) OR (abs(y) + y) / 2 OR y * (y > 0)
    return np.maximum(0, y)

y = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])
print(relu(-0.5))
print(relu(0.65))

def leaky_relu(l):
    return np.maximum(0.01*l, l)

l = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])
# print(leaky_relu(l))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

z = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])

# print(tanh(z))

def softmax_numpy(x):
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

# print("softmax(x) = " + str(softmax_numpy(x)))
# print(np.sum(softmax_numpy(x)[0]))
# print(np.sum(softmax_numpy(x)[1]))

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


softmax_inputs = [2, 3, 5, 6]
# print("Softmax Function Output :: {}".format(softmax(softmax_inputs)))
# print("Total sum is :: {}".format(np.sum(softmax(softmax_inputs))))

# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6\
