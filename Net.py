import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


class PlainNet:
    def __init__(self):
        self.layers = []
        self.parameters = []
        self.decays = []
        self.gradients = []

    def append_layer(self, layer):
        self.layers.append(layer)
        self.parameters.append(layer.parameters)
        self.decays.append(layer.decays)
        self.gradients.append(layer.gradients)

    def train(self, train_data: np.ndarray, train_labels: list, validate_data: np.ndarray, validate_labels: list,
              learning_rate=0.1, weight_decay=0.01, batch_size=None, num_epoch=1000):
        batch_begin_col = 0
        batch_end_col = batch_size

        # plt.ion()
        batch_losses = []
        batch_numbers = []
        validation_accuracies = []
        validation_numbers = []
        epoch = 0
        iters = 0
        print('epoch: %d' % epoch)

        best_model = [np.array([])] * len(self.parameters)
        best_validation_accuracy = 0.0

        while epoch < num_epoch:
            # plt.pause(0.001)

            batch_end_col = min(batch_end_col, train_data.shape[-1])
            batch_data = train_data[:, batch_begin_col: batch_end_col]
            batch_labels = train_labels[batch_begin_col: batch_end_col]

            # forward pass
            result = batch_data
            for layer in self.layers:
                result = layer.forward(result)

            # evaluate batch loss
            num_samples = batch_data.shape[-1]
            correct_logprobs = -np.log(result[batch_labels, range(num_samples)])
            batch_loss = np.sum(correct_logprobs) / num_samples
            reg_loss = 0
            for decay in self.decays:
                if decay is None:
                    continue
                reg_loss += 0.5 * weight_decay * np.sum(decay * decay)  # L2 regularization
            batch_losses.append(batch_loss + reg_loss)
            batch_numbers.append(iters)

            # if iters % 1000 == 0:
            #     print('batch loss: %f' % batch_loss)

            # backward pass
            result = self.layers[-1].backward(result, batch_labels)
            for i in range(len(self.layers) - 2, -1, -1):
                result = self.layers[i].backward(result)

            # update parameters
            for param, decay, grad in zip(self.parameters, self.decays, self.gradients):
                if param is not None and grad is not None:
                    param -= learning_rate * grad
                    grad[:, :] = 0.0
                if decay is not None:
                    param[:decay.shape[0], :decay.shape[1]] -= learning_rate * weight_decay * decay  # L2 regularization
            iters += 1

            batch_begin_col = batch_end_col
            batch_end_col += batch_size
            if batch_begin_col >= train_data.shape[-1]:
                batch_begin_col = 0
                batch_end_col = batch_size

                epoch += 1
                # evaluate validation accuracy
                probs = self.predict(validate_data)
                predicted_class = np.argmax(probs, axis=0)
                validation_accuracy = np.mean(predicted_class == validate_labels)
                validation_numbers.append(epoch)
                validation_accuracies.append(validation_accuracy)
                print('#epoch %d: validation accuracy = %f' % (epoch, validation_accuracy))
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    for i in range(len(self.parameters)):
                        if self.parameters[i] is None:
                            best_model[i] = None
                            continue
                        best_model[i] = np.copy(self.parameters[i])

        # reload the best model
        for i in range(len(best_model)):
            if self.parameters[i] is None:
                continue
            self.parameters[i][:, :] = best_model[i]

        plt.clf()
        plt.figure(1)
        plt.subplot(211)
        plt.plot(batch_numbers, batch_losses)
        plt.subplot(212)
        plt.plot(validation_numbers, validation_accuracies)
        plt.show()

    def predict(self, test_data: np.ndarray):
        result = test_data
        for layer in self.layers:
            result = layer.predict(result)
        return result
