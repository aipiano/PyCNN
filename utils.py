import numpy as np
import scipy.linalg as la
import pickle
import matplotlib.pyplot as plt


def center_data(data: np.ndarray):
    m = np.mean(data, axis=0)
    return data.astype(float) - m, m


def normalize_data(centered_data: np.ndarray):
    std = np.std(centered_data, axis=0)
    return centered_data.astype(float) / std, std


def homogeneous_data(data: np.ndarray):
    ones = np.ones((data.shape[0], 1), dtype=data.dtype)
    return np.hstack((data, ones))


def unpickle(file_name):
    f = open(file_name, 'rb')
    d = pickle.load(f, encoding='bytes')
    f.close()
    return d


def load_cifar_10(dir_path):
    d1 = unpickle(dir_path + '/data_batch_1')
    d2 = unpickle(dir_path + '/data_batch_2')
    d3 = unpickle(dir_path + '/data_batch_3')
    d4 = unpickle(dir_path + '/data_batch_4')
    d5 = unpickle(dir_path + '/data_batch_5')
    t1 = unpickle(dir_path + '/test_batch')

    # CIFAR-10 is already shuffled
    train_data = np.vstack((d1[b'data'], d2[b'data'], d3[b'data'], d4[b'data'], d5[b'data']))
    train_labels = d1[b'labels'] + d2[b'labels'] + d3[b'labels'] + d4[b'labels'] + d5[b'labels']
    test_data = t1[b'data']
    test_labels = t1[b'labels']

    # For images, it is not common to normalize variance, to do PCA or whitening
    train_data, train_mean = center_data(train_data)
    # train_data, train_std = normalize_data(train_data)

    test_data = test_data.astype(float)
    test_data -= train_mean
    # test_data /= train_std

    # convert to homogeneous features
    # homogeneous_data(train_data)
    # homogeneous_data(test_data)

    train = {'data': train_data.T, 'labels': train_labels}
    test = {'data': test_data.T, 'labels': test_labels}
    return train, test


def load_cifar10_as_images(dir_path):
    d1 = unpickle(dir_path + '/data_batch_1')
    d2 = unpickle(dir_path + '/data_batch_2')
    d3 = unpickle(dir_path + '/data_batch_3')
    d4 = unpickle(dir_path + '/data_batch_4')
    d5 = unpickle(dir_path + '/data_batch_5')
    t1 = unpickle(dir_path + '/test_batch')

    # CIFAR-10 is already shuffled
    train_data = np.vstack((d1[b'data'], d2[b'data'], d3[b'data'], d4[b'data'], d5[b'data']))
    train_labels = d1[b'labels'] + d2[b'labels'] + d3[b'labels'] + d4[b'labels'] + d5[b'labels']
    test_data = t1[b'data']
    test_labels = t1[b'labels']

    # For images, it is not common to normalize variance, to do PCA or whitening
    train_data, train_mean = center_data(train_data)
    train_data, train_std = normalize_data(train_data)

    test_data = test_data.astype(float)
    test_data -= train_mean
    test_data /= train_std

    # convert to homogeneous features
    # homogeneous_data(train_data)
    # homogeneous_data(test_data)

    train_data = train_data.reshape((train_data.shape[0], 3, 32, 32))
    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32))

    # t = train_data.reshape((train_data.shape[0], 32, 32, 3))
    # plt.imshow(0.5 + t[100, :, :, 0], cmap='gray')
    # plt.show()

    train = {'data': train_data, 'labels': train_labels}
    test = {'data': test_data, 'labels': test_labels}
    return train, test
