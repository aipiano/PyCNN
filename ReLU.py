import numpy as np
import scipy.linalg as la


class ReLU:
    def __init__(self, dim_in):
        self.parameters = None
        self.gradients = None
        self.decays = None
        self.dim_out = dim_in
        self.x = None

    def predict(self, fan_in: np.ndarray):
        return np.maximum(0, fan_in)

    def forward(self, fan_in: np.ndarray):
        self.x = np.copy(fan_in)
        return self.predict(fan_in)

    def backward(self, fan_out_diff: np.ndarray):
        fan_out_diff[self.x <= 0] = 0
        return fan_out_diff


