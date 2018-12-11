import numpy as np
import scipy.linalg as la


class SoftmaxOutput:
    def __init__(self, dim_in):
        self.parameters = None
        self.gradients = None
        self.decays = None
        self.dim_out = dim_in

    def predict(self, fan_in: np.ndarray):
        fan_in -= np.max(fan_in, axis=0, keepdims=True)
        np.exp(fan_in, fan_in)
        return fan_in / np.sum(fan_in, axis=0, keepdims=True)

    def forward(self, fan_in: np.ndarray):
        return self.predict(fan_in)

    def backward(self, predict_results: np.ndarray, labels: list):
        num_samples = predict_results.shape[-1]
        predict_results[labels, range(num_samples)] -= 1
        predict_results /= num_samples
        return predict_results
