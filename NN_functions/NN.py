import numpy as np
# M3WT59XD

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    x_example = np.array([[0.34651113, 0.82203687], # 2 by 2
                  [0.98299945, 0.52486512]])
    y_example = np.array([[0],[1]])

    # weights1 looks like this ROWS BY COLUMN
    w1_example = np.array([[0.94441638, 0.77813664, 0.8479552 , 0.08550717], # 2 by 4
                [0.91684602, 0.11492581, 0.35739348, 0.76254734]])

    # weights2 looks like this ROWS BY COLUMN
    w2_example = np.array([[0.51191813], # 4 by 1
                [0.84167029],
                [0.97527654],
                [0.45685174]])

    def feedforward(self):
            '''
            this looks something like dot product of 2by2 and 2by4
            which will give a 2by4 matrix.
            '''
            self.layer1 = sigmoid(np.dot(self.input, self.weights1))
            '''
            this looks something like dot product of 2by4 and 4by1
            which will give a 2by1 matrix.
            '''
            self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # d(loss)/d(weight)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        #
        # # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [0,0,0],
                  [1,1,1]])
    y = np.array([[0],[1],[0],[1],[1]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
