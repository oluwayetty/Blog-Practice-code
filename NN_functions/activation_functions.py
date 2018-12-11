import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])
# print(sigmoid(x))

# https://www.youtube.com/watch?v=z4o0R4tMjw0
def relu(y):
    #could also be np.maximum(y, 0) OR (abs(y) + y) / 2 OR y * (y > 0)
    return np.maximum(0, y)

y = np.array([
    [-0.49986085, 0.07963809],
    [-0.29516847, 0.65896187],
    [0.48214381, 0.50296628]
])
print(relu(y))

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

# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
