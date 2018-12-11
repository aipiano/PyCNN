import numpy as np
import scipy.linalg as la


class ReLU2D:
    def __init__(self):
        self.x = None

    def predict(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        # out = np.zeros_like(images)
        for i in range(images.shape[0]):    # for all images
            for c in range(images.shape[1]):    # for all channels
                np.maximum(images[i, c, :, :], 0, images[i, c, :, :])
        return images

    def forward(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        self.x = np.copy(images)
        return self.predict(images)

    def backward(self, diffs: np.ndarray):
        # diffs: [index, channels, rows, cols]
        for i in range(diffs.shape[0]):  # for all images
            for c in range(diffs.shape[1]):  # for all channels
                diff = diffs[i, c, :, :]
                diff[self.x[i, c, :, :] < 0] = 0
        return diffs

