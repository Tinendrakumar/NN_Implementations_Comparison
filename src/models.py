import numpy as np
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err, err

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

        # self.weight = np.random.random((num_nodes, len(input_vals)))
        # self.weight_delta = np.zeros((num_nodes, len(input_vals)))
        # self.layer_net = np.zeros(num_nodes)
        # self.layer_out = np.zeros(num_nodes)
        # self.bias = (random.random() * 2) - 1

    def eval(self):
        # Evaluation part
        # Get input, compute the output of layer nodes.
        for i in range(self.num_nodes):
            net = self.bias + np.dot(self.input_vals, self.weight[i])
            self.layer_net[i] = net
            self.layer_out[i] = sig(net)

    def backprop(self, other):
        # Use backpropagation method to update weights
        weight = len(self.weight)
        for i in range(weight):
            for j in range(len(self.weight[i])):
                if self.layer_num == 1:
                    delta = other.weight_delta[0][i] * self.input_vals[j] * other.weight[0][i] * inv_sig(
                        self.layer_out[i])
                elif self.layer_num == 2:
                    self.weight_delta[i][j] = inv_sig(self.layer_out[i]) * inv_err(self.layer_out[i], other)
                    delta = self.weight_delta[i][j] * self.input_vals[j]

                # Update weights
                self.weight[i][j] = self.weight[i][j] - (LEARNING_RATE * delta)


class cfile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = open(filename, mode)

    def w(self, content):
        self.file.write(str(content) + '\n')  # Convert content to string before writing

    def close(self):
        self.file.close()

