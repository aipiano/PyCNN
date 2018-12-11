import numpy as np
import scipy.linalg as la


class InputsLayer:
    def __init__(self, input_data: np.ndarray, labels: list, dim_out: int, batch_size: int):
        self.prelayer = None
        self.dim_out = dim_out
        self.data = input_data
        self.labels = labels

        self.batch_size = batch_size
        self.batch_begin_col = 0
        self.batch_end_col = batch_size

    def fetch_batch(self):
        self.batch_end_col = min(self.batch_end_col, self.data.shape[-1])
        data_batch = self.data[:, self.batch_begin_col: self.batch_end_col]
        labels_batch = self.labels[self.batch_begin_col: self.batch_end_col]

        self.batch_begin_col = self.batch_end_col
        self.batch_end_col += self.batch_size
        if self.batch_begin_col >= self.data.shape[-1]:
            self.batch_begin_col = 0
            self.batch_end_col = self.batch_size

        return data_batch, labels_batch
