import numpy as np
import scipy.linalg as la


class FullyConnected2D:
    def __init__(self, dim_in: int, dim_out: int):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = np.random.randn(dim_out, dim_in) * np.sqrt(2.0 / dim_in)
        self.b = np.zeros((dim_out, 1))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.x = None
        self.col_vecs = None

    def predict(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        self.col_vecs = np.zeros((self.dim_in, images.shape[0]), dtype=float)
        for i in range(images.shape[0]):
            self.col_vecs[:, i:i+1] = images[i, :, :, :].reshape((-1, 1))
        return self.w.dot(self.col_vecs) + self.b

    def forward(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        self.x = np.copy(images)
        return self.predict(images)

    def backward(self, diffs: np.ndarray):
        self.dw += diffs.dot(self.col_vecs.T)
        self.db += np.sum(diffs, axis=1, keepdims=True)
        dx = self.w.T.dot(diffs)
        return dx.reshape(self.x.shape)
