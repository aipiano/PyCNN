import numpy as np
import scipy.linalg as la
from scipy.signal import convolve2d as conv2d
from scipy.signal import fftconvolve as conv3d


class Convolution:
    def __init__(self, img_size, img_channels, num_kernels=16, kernel_size=3):
        """
        仅支持stride=1
        :param img_size:
        :param img_channels:
        :param num_kernels:
        :param kernel_size:
        """
        # assert (img_size - kernel_size + 2*pad) % stride == 0
        # self.out_img_size = (img_size - kernel_size + 2*pad) / stride + 1
        self.img_size = img_size
        self.dim_in = img_channels * kernel_size * kernel_size
        self.num_kernels = num_kernels
        self.input_channels = img_channels
        # w[i, :, :, :] 表示第i个kernel, b[i]表示第i个kernel的偏置
        self.w = np.random.randn(num_kernels, img_channels, kernel_size, kernel_size) * np.sqrt(2.0 / self.dim_in)
        self.b = np.zeros(num_kernels)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.x = None

    def predict(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        out_shape = (images.shape[0], self.num_kernels, images.shape[2], images.shape[3])
        # out = np.zeros(out_shape, dtype=float)
        # for i in range(out_shape[3]):   # for all input images
        #     for c in range(self.input_channels):  # for all input channels，其实可以更紧凑的用三维卷积表示
        #         padded_image = np.pad(images[:, :, c, i], 1, mode='constant')
        #         for k in range(self.num_kernels):   # for all kernels (output channels)
        #             out[:, :, k, i] += conv2d(padded_image, self.w[:, :, c, k], mode='valid')
        out = np.zeros(out_shape, dtype=float)
        for i in range(out_shape[0]):  # for all input images
            padded_image = np.pad(images[i, :, :, :], ((0, 0), (1, 1), (1, 1)), mode='constant')
            for k in range(self.num_kernels):  # for all kernels (output channels)
                feature_map = conv3d(padded_image, self.w[k, :, :, :], mode='valid')
                out[i, k:k+1, :, :] += feature_map
        return out

    def forward(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        self.x = np.copy(images)
        return self.predict(images)

    def backward(self, diffs: np.ndarray):
        # diffs: [index, channels, rows, cols]
        backprop_diffs = np.zeros_like(self.x, dtype=float)
        # for i in range(diffs.shape[3]):  # for all diffs
        #     for c in range(self.input_channels):  # for all input channels, 可以更紧凑的用三维卷积表示
        #         padded_input = np.pad(self.x[:, :, c, i], 1, mode='constant')
        #         padded_input = np.rot90(padded_input, 2)  # rotate input diff by 180 degrees
        #         for k in range(self.num_kernels):  # for all kernels (diff channels)
        #             self.dw[:, :, c, k] += conv2d(padded_input, diffs[:, :, k, i], mode='valid')
        #             self.db[k] += np.sum(diffs[:, :, k, i])
        #             rot_w = np.rot90(self.w[:, :, c, k], 2)  # rotate kernel weights by 180 degrees
        #             padded_diff = np.pad(diffs[:, :, k, i], 1, mode='constant')
        #             backprop_diffs[:, :, c, i] += conv2d(padded_diff, rot_w, mode='valid')
        for i in range(diffs.shape[0]):  # for all diffs
            padded_input = np.pad(self.x[i, :, :, :], ((0, 0), (1, 1), (1, 1)), mode='constant')
            fliped_input = padded_input[::-1, ::-1, ::-1]
            # padded_diffs = np.pad(diffs[:, :, :, i], ((1, 1), (1, 1), (0, 0)), mode='constant')
            for k in range(self.num_kernels):  # for all kernels (diff channels)
                self.dw[k, :, :, :] += conv3d(fliped_input, diffs[i, k:k+1, :, :], mode='valid')
                self.db[k] += np.sum(diffs[i, k, :, :])
                fliped_w = self.w[k, ::-1, ::-1, ::-1]
                backprop_diffs[i, :, :, :] += conv3d(diffs[i, k:k+1, :, :], fliped_w, mode='same')
        return backprop_diffs

