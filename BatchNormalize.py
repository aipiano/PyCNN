import numpy as np
import scipy.linalg as la


class BatchNormalize:
    def __init__(self, dim_in):
        self.dim_out = dim_in
        self.parameters = np.zeros((self.dim_out, 2), dtype=float)
        self.gradients = np.zeros_like(self.parameters)
        self.decays = None
        self.scale = self.parameters[:, 0:1]    # use slice operator(:) to keep dimention unchanged
        self.shift = self.parameters[:, 1:2]
        self.scale[:, :] = 1.0
        self.diff_scale = self.gradients[:, 0:1]
        self.diff_shift = self.gradients[:, 1:2]

        self.mean = np.zeros((self.dim_out, 1), dtype=float)
        self.std = np.ones((self.dim_out, 1), dtype=float)
        self.batch_mean = np.zeros((self.dim_out, 1), dtype=float)
        self.batch_var = np.ones((self.dim_out, 1), dtype=float)
        self.batch_std = np.ones((self.dim_out, 1), dtype=float)
        self.x = None
        self.nx = None

    def predict(self, fan_in: np.ndarray):
        fan_in -= self.mean
        fan_in /= self.std
        return self.scale * fan_in + self.shift

    def forward(self, fan_in: np.ndarray):
        self.x = np.copy(fan_in)
        self.batch_mean[:, :] = np.mean(fan_in, axis=1, keepdims=True)
        self.batch_var[:, :] = np.var(fan_in, axis=1, keepdims=True) + 1e-6
        self.batch_std[:, :] = np.sqrt(self.batch_var)

        # running average
        self.mean *= 0.9
        self.mean += 0.1 * self.batch_mean
        self.std *= 0.9
        self.std += 0.1 * self.batch_std

        fan_in -= self.batch_mean
        fan_in /= self.batch_std
        self.nx = np.copy(fan_in)
        return self.scale * fan_in + self.shift

    def backward(self, fan_out_diff: np.ndarray):
        self.diff_scale += np.sum(fan_out_diff * self.nx, axis=1, keepdims=True)
        self.diff_shift += np.sum(fan_out_diff, axis=1, keepdims=True)

        '''
        a straightforward implementation of papaer:
        Batch Normalization：Accelerating Deep Network Training by Reducing Internal Covariate Shift
        '''
        diff_nx = fan_out_diff * self.scale      # [KxN]
        centered_x = self.x - self.batch_mean    # [KxN]
        inv_std = 1.0 / self.batch_std           # [Kx1]
        diff_var = -0.5 * np.power(inv_std, 3) * np.sum(diff_nx * centered_x, axis=1, keepdims=True)    # [Kx1]

        inv_batch_size = 1.0 / self.x.shape[-1]
        dmean1 = diff_nx * inv_std                                    # [KxN]
        dmean2 = 2 * inv_batch_size * diff_var * centered_x           # [KxN]
        diff_mean = -np.sum(dmean1 + dmean2, axis=1, keepdims=True)   # [Kx1]

        diff_x = dmean1 + dmean2 + diff_mean * inv_batch_size         # [KxN]
        return diff_x
