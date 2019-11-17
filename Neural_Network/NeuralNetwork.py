import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt


def normalize(data):
    # m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [int(255*(float(i) - mn) / (mx - mn)) for i in data]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, layers):
        self.activation = sigmoid
        self.activation_deriv = sigmoid_derivative
        self.weights = []
        self.bias = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def fit(self, x, y, learning_rate=0.2, epochs=3):
        x = np.atleast_2d(x)
        n = len(y)
        p = max(n, epochs)
        y = np.array(y)

        for k in trange(epochs * n):
            if (k+1) % p == 0:
                learning_rate *= 0.5
            a = [x[k % n]]
            # 正向传播
            for lay in range(len(self.weights)):
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))
            # 反向传播
            label = np.zeros(a[-1].shape)
            label[y[k % n]] = 1
            error = label - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            layer_num = len(a) - 2
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]

    def predict(self, x):
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))
        i = a.index(max(a))
        per = []
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per

    def hidden_img(self, x):
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):
            img = np.array(normalize(a), dtype=np.uint8)
            img = img.reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(str(lay), fontsize=24)
            plt.axis('off')
            plt.savefig('hidden_img/hidden_img' + str(lay) + '.png')
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))
        i = a.index(max(a))
        per = []
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per
