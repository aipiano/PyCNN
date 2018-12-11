import numpy as np
import scipy.linalg as la


# Linear mapping layer
class FullyConnected:
    def __init__(self, dim_in, dim_out: int):
        self.dim_out = dim_out
        self.parameters = np.zeros((dim_out, dim_in + 1), dtype=float)
        self.gradients = np.zeros_like(self.parameters)
        self.decays = self.parameters[:, :-1]
        self.w = self.parameters[:, :-1]
        self.w[:, :] = np.random.randn(dim_out, dim_in) * np.sqrt(2.0 / dim_in)
        self.b = self.parameters[:, -2: -1]  # use slice operator(:) to keep dimention unchanged
        # self.b[:, :] = 0.01
        self.dw = self.gradients[:, :-1]
        self.db = self.gradients[:, -2: -1]
        self.x = None

    def predict(self, fan_in: np.ndarray):
        # WX + b
        return self.w.dot(fan_in) + self.b

    def forward(self, fan_in: np.ndarray):
        self.x = np.copy(fan_in)
        return self.predict(fan_in)

    def backward(self, fan_out_diff: np.ndarray):
        self.dw += fan_out_diff.dot(self.x.T)  # 用增量更新梯度，以便适应于有多层与该层相连的情况
        self.db += np.sum(fan_out_diff, axis=1, keepdims=True)
        return self.w.T.dot(fan_out_diff)  # dx
