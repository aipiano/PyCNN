import numpy as np
import scipy.linalg as la


class Dropout:
    def __init__(self, dim_in, dropout_rate=0.5):
        self.parameters = None
        self.gradients = None
        self.decays = None

        self.dim_out = dim_in
        self.dropout_rate = dropout_rate
        self.mask = None

    def predict(self, fan_in: np.ndarray):
        return fan_in

    def forward(self, fan_in: np.ndarray):
        self.mask = (np.random.rand(*fan_in.shape) > self.dropout_rate) / self.dropout_rate
        return fan_in * self.mask

    def backward(self, fan_out_diff: np.ndarray):
        return fan_out_diff * self.mask
