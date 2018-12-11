import matplotlib.pyplot as plt
import numpy as np
import utils

from Convolution import Convolution
from MaxPooling import MaxPooling
from ReLU2D import ReLU2D
from ReLU import ReLU
from FullyConnected2D import FullyConnected2D
from FullyConnected import FullyConnected
from SoftmaxOutput import SoftmaxOutput


def main():
    train_dict, test_dict = utils.load_cifar10_as_images('D:/Dataset/cifar-10-batches-py')
    # train_dict, test_dict = utils.load_cifar_10('D:/Dataset/cifar-10-batches-py')

    train_data, train_labels = train_dict['data'], train_dict['labels']

    validation_count = int(0.1 * train_data.shape[0])
    X = train_data[:-validation_count, :, :, :]
    y = train_labels[:-validation_count]

    # plt.imshow(0.5 + X[100, 2, :, :], cmap='gray')
    # plt.show()

    val_X = train_data[-validation_count:, :, :, :]
    val_y = train_labels[-validation_count:]

    conv1 = Convolution(32, 3, 8, 3)
    relu1 = ReLU2D()
    pool1 = MaxPooling()
    # conv2 = Convolution(16, 8, 8, 3)
    # relu2 = ReLU2D()
    fc1 = FullyConnected2D(16*16*8, 100)
    relu3 = ReLU(100)
    fc2 = FullyConnected(100, 10)
    out = SoftmaxOutput(10)

    num_iter = 0
    batch_size = 200
    batch_begin_col = 0
    batch_end_col = batch_size
    batch_losses = []
    batch_numbers = []
    learning_rate = 0.01
    while num_iter < 1000:
        num_iter += 1
        batch_end_col = min(batch_end_col, train_data.shape[0])
        batch_data = train_data[batch_begin_col: batch_end_col, :, :, :]
        batch_labels = train_labels[batch_begin_col: batch_end_col]

        # forward
        r = conv1.forward(batch_data)
        r = relu1.forward(r)
        r = pool1.forward(r)
        # r = conv2.forward(r)
        # r = relu2.forward(r)
        r = fc1.forward(r)
        r = relu3.forward(r)
        r = fc2.forward(r)
        r = out.forward(r)

        # evaluate batch loss
        num_samples = batch_data.shape[0]
        correct_logprobs = -np.log(r[batch_labels, range(num_samples)])
        batch_loss = np.sum(correct_logprobs) / num_samples
        batch_losses.append(batch_loss)
        batch_numbers.append(num_iter)
        print('batch loss: %f' % batch_loss)

        result_labels = np.argmax(r, axis=0)
        matches = result_labels == batch_labels
        accurate = np.count_nonzero(matches) / len(batch_labels)
        print('batch accuracy: %f' % accurate)

        # backward
        r = out.backward(r, batch_labels)
        r = fc2.backward(r)
        r = relu3.backward(r)
        r = fc1.backward(r)
        # r = relu2.backward(r)
        # r = conv2.backward(r)
        r = pool1.backward(r)
        r = relu1.backward(r)
        r = conv1.backward(r)

        # update
        conv1.w -= learning_rate * conv1.dw
        conv1.b -= learning_rate * conv1.db
        # conv2.w -= learning_rate * conv2.dw
        # conv2.b -= learning_rate * conv2.db
        fc1.w -= learning_rate * fc1.dw
        fc1.b -= learning_rate * fc1.db
        fc2.w -= learning_rate * fc2.dw
        fc2.b -= learning_rate * fc2.db

        # clear gradients
        conv1.dw[:, :, :, :] = 0
        conv1.db[:] = 0
        # conv2.dw[:, :, :, :] = 0
        # conv2.db[:] = 0
        fc1.dw[:, :] = 0
        fc1.db[:, :] = 0
        fc2.dw[:, :] = 0
        fc2.db[:, :] = 0

        batch_begin_col = batch_end_col
        batch_end_col += batch_size
        if batch_begin_col >= train_data.shape[0]:
            batch_begin_col = 0
            batch_end_col = batch_size

    # test
    r = conv1.forward(test_dict['data'])
    r = relu1.forward(r)
    r = pool1.forward(r)
    # r = conv2.forward(r)
    # r = relu2.forward(r)
    r = fc1.forward(r)
    r = relu3.forward(r)
    r = fc2.forward(r)
    r = out.forward(r)
    result_labels = np.argmax(r, axis=0)
    matches = result_labels == test_dict['labels']
    accurate = np.count_nonzero(matches) / len(test_dict['labels'])
    print('Test accuracy = %f' % accurate)

main()
