import numpy as np
import scipy.linalg as la


class MaxPooling:
    def __init__(self, kernel_size=2, stride=2):
        """
        仅支持kernel_size=2, stride=2
        :param kernel_size:
        :param stride:
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.masks = None

    @staticmethod
    def block_reduce(image: np.ndarray):
        rows = image.shape[0]
        cols = image.shape[1]
        out_image = np.zeros((int(rows/2), int(cols/2)), dtype=float)
        out_mask = np.zeros_like(out_image, dtype=int)
        for oy, y in enumerate(range(0, cols - 2, 2)):
            for ox, x in enumerate(range(0, rows - 2, 2)):
                out_image[oy, ox] = np.max(image[oy: oy+2, ox: ox+2])
                out_mask[oy, ox] = np.argmax(image[oy: oy+2, ox: ox+2])
        return out_image, out_mask

    @staticmethod
    def masked_upsample(image: np.ndarray, mask: np.ndarray):
        rows = image.shape[0]
        cols = image.shape[1]
        out_image = np.zeros((rows*2, cols*2), dtype=float)
        for y in range(rows):
            for x in range(cols):
                loc_id = mask[y, x]
                by = loc_id / 2
                bx = loc_id % 2
                block = out_image[y*2: y*2+2, x*2: x*2+2]
                block[by, bx] = image[y, x]
        return out_image

    def predict(self, images: np.ndarray):
        # images: [index, channels, rows, cols]
        chns = images.shape[1]
        rows = int(images.shape[2]/2)
        cols = int(images.shape[3]/2)
        self.masks = np.zeros((images.shape[0], chns, rows, cols), dtype=int)
        out = np.zeros_like(self.masks, dtype=float)
        for i in range(images.shape[0]):    # for all images
            for c in range(images.shape[1]):    # for all input channels
                out[i, c, :, :], self.masks[i, c, :, :] = MaxPooling.block_reduce(images[i, c, :, :])
        return out

    def forward(self, images: np.ndarray):
        return self.predict(images)

    def backward(self, diffs: np.ndarray):
        # diffs: [index, channels, rows, cols]
        chns = diffs.shape[1]
        rows = diffs.shape[2] * 2
        cols = diffs.shape[3] * 2
        backprop_diffs = np.zeros((diffs.shape[0], chns, rows, cols))
        for i in range(diffs.shape[0]):    # for all images
            for c in range(diffs.shape[1]):    # for all input channels
                backprop_diffs[i, c, :, :] = MaxPooling.masked_upsample(diffs[i, c, :, :], self.masks[i, c, :, :])
        return backprop_diffs



