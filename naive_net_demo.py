import matplotlib.pyplot as plt
import numpy as np

import utils
from BatchNormalize import BatchNormalize
from FullyConnected import FullyConnected
from InputsLayer import InputsLayer
from ReLU import ReLU
from SoftmaxOutput import SoftmaxOutput
from Net import PlainNet


def gen_data():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X.T, list(y)


def main():
    train_dict, test_dict = utils.load_cifar_10('D:/Dataset/cifar-10-batches-py')
    train_data, train_labels = train_dict['data'], train_dict['labels']

    validation_count = int(0.1 * train_data.shape[-1])
    X = train_data[:, :-validation_count]
    y = train_labels[:-validation_count]

    val_X = train_data[:, -validation_count:]
    val_y = train_labels[-validation_count:]

    # X, y = gen_data()
    # shuffled_idx = list(range(len(y)))
    # np.random.shuffle(shuffled_idx)
    # X = X[:, shuffled_idx]
    # y = list(np.array(y)[shuffled_idx])

    net = PlainNet()
    net.append_layer(FullyConnected(3072, 100))
    net.append_layer(BatchNormalize(100))
    net.append_layer(ReLU(100))
    net.append_layer(FullyConnected(100, 10))
    net.append_layer(SoftmaxOutput(10))

    net.train(X, y, val_X, val_y,  0.1, 0.01, 1000, 1000)

    result = net.predict(X)
    result_labels = np.argmax(result, axis=0)
    matches = result_labels == y
    accurate = np.count_nonzero(matches) / len(y)
    print('Final accuracy = %f' % accurate)

    result = net.predict(test_dict['data'])
    result_labels = np.argmax(result, axis=0)
    matches = result_labels == test_dict['labels']
    accurate = np.count_nonzero(matches) / len(test_dict['labels'])
    print('Test accuracy = %f' % accurate)

main()
