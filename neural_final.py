import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plot
import pandas as ps
import math as mth


def main():
    # Hyperparameters...
    leareningRate = 0.1
    epochs = 1
    noofHiddenUnits = 100
    momentum = 0.9
    examples = 60000

    print("Reading data")
    trainfile = "mnist_train.csv"
    testfile = "mnist_test.csv"
    training_examples = np.array(ps.read_csv(trainfile, header=None), np.float)
    testing_examples = np.array(ps.read_csv(testfile, header=None), np.float)

    training_bias = np.ones((examples, 1), dtype=float);
    testing_bias = np.ones((10000, 1), dtype=float);

    training_examples[:, 1:] = (training_examples[:, 1:] / 255.0)
    testing_examples[:, 1:] = (testing_examples[:, 1:] / 255.0)

    training_examples = np.append(training_examples, training_bias, axis=1)
    testing_examples = np.append(testing_examples, testing_bias, axis=1)
    print("data read and cleaned")

    outputLayer = np.zeros(noofHiddenUnits + 1)

    outputLayer[0] = 1

    weightsI2H = np.random.uniform(-0.05, 0.05, (785, noofHiddenUnits))
    weightsH2O = np.random.uniform(-0.05, 0.05, (noofHiddenUnits + 1, 10))

    oldDeltaH = np.zeros((noofHiddenUnits, 785))
    oldDeltaK = np.zeros((noofHiddenUnits + 1, 10))

    expected_output_vector = np.zeros((examples, 10), float) + 0.1

    training_accuracy = np.zeros(epochs, float)
    testing_accuracy = np.zeros(epochs, float)

    for i in range(examples):
        expected_output_vector[i][int(training_examples[i][0])] = 0.9

    hidden_layer_activations = np.zeros(noofHiddenUnits + 1)
    hidden_layer_activations[0] = 1
    for epoch in range(epochs):
        # Reset confusion matrices
        training_confusion_matrix = np.zeros((10, 10), int)
        testing_confusion_matrix = np.zeros((10, 10), int)

        print("Epoch: "), epoch
        for i in range(examples):
            # Feed forward
            hidden_layer_activations[1:] = (1 / (1 + np.exp(-1 * np.dot(training_examples[i][1:], weightsI2H))))

            outputLayer[1:] = hidden_layer_activations[1:]
            output_layer_activations = (1 / (1 + np.exp(-1 * np.dot(outputLayer, weightsH2O))))
            output_layer_error_terms = (output_layer_activations *
                                        (1 - output_layer_activations) * (expected_output_vector[i] -
                                                                          output_layer_activations))
            hidden_layer_error_terms = (
                    hidden_layer_activations[1:] * (1 - hidden_layer_activations[1:]) * np.dot(weightsH2O[1:, :],
                                                                                               output_layer_error_terms))

            deltaK = leareningRate * (np.outer(hidden_layer_activations, output_layer_error_terms)) + (
                    momentum * oldDeltaK)
            deltaH = leareningRate * np.outer(hidden_layer_error_terms, training_examples[i][1:]) + (
                    momentum * oldDeltaH)
            weightsH2O = weightsH2O + deltaK
            oldDeltaK = deltaK
            weightsI2H = weightsI2H + deltaH.T
            oldDeltaH = deltaH

            training_confusion_matrix[int(training_examples[i][0])][int(np.argmax(output_layer_activations))] += 1

        training_accuracy[epoch] = (float((sum(training_confusion_matrix.diagonal())) / 60000.0) * 100.0)
        print('Epoch ', epoch)
        print('Training Accuracy:', training_accuracy[epoch])

        test = 0

        # Calculating for test data
        for i in range(10000):
            # Feed forward pass input to output layer through hidden layer
            hidden_layer_activations[1:] = (
                    1 / (1 + np.exp(-1 * np.dot(testing_examples[i][1:], weightsI2H))))
            # Forward propagate the activations from hidden layer to output layer
            outputLayer[1:] = hidden_layer_activations[1:]
            # calculate dot product for output layer
            # apply sigmoid function to sum of weights times inputs
            output_layer_activations = (1 / (1 + np.exp(-1 * np.dot(outputLayer, weightsH2O))))
            # print("output_layer:",output_layer_activations)
            testing_confusion_matrix[int(testing_examples[i][0])][int(np.argmax(output_layer_activations))] += 1
            test += 1
            y_true = int(testing_examples[i][0])
            y_pred = int(np.argmax(output_layer_activations))
            #print("True labels", y_true,"Predicted labels", y_pred)
            # confusion_matrix(y_true, y_pred)
        print(test)

        testing_accuracy[epoch] = ((float(sum(testing_confusion_matrix.diagonal())) / 10000.0) * 100.0)
        print("Epoch ", epoch, ": ", "Testing Accuracy: ", testing_accuracy[epoch], "%")

    np.set_printoptions(threshold=np.nan)

    print("Testing confusion matrix")
    # print(confusion_matrix)
    print(testing_confusion_matrix)

    true_pos = np.diag(testing_confusion_matrix)
    false_pos = np.sum(testing_confusion_matrix, axis=0) - true_pos
    false_neg = np.sum(testing_confusion_matrix, axis=1) - true_pos

    def precision(label, testing_confusion_matrix):
        col = testing_confusion_matrix[:, label]
        return testing_confusion_matrix[label, label] / col.sum()

    def recall(label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def precision_macro_average(testing_confusion_matrix):
        rows, columns = testing_confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += precision(label, testing_confusion_matrix)
        return sum_of_precisions / rows

    def recall_macro_average(testing_confusion_matrix):
        rows, columns = testing_confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += recall(label, testing_confusion_matrix)
        return sum_of_recalls / columns

    print("precision total:", precision_macro_average(testing_confusion_matrix))
    print("recall total:", recall_macro_average(testing_confusion_matrix))

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    print("accuracy:", accuracy(testing_confusion_matrix))


main()