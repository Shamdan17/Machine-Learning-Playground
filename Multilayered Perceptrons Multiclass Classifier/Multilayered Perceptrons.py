#!/usr/bin/env python import numpy as np
import csv
import math
import matplotlib.pyplot as pt

# To read csv files
def read_csv(filename):
    with open(filename, newline='') as fl:
        return [list(map(float, row)) for row in csv.reader(fl)]

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Softmax function
def softmax(X):
    return np.exp(X)/np.sum(np.exp(X), axis = 1, keepdims = True)

# Safelog function
def safelog(x):
    return np.log(x + 1e-100)

#gradient for V
def gradV(Y, Z):
    return np.matmul(np.concatenate((np.ones(shape = (1,500)), Z), axis=0), enc-Y)


#gradient for W
def gradW(Y, V, Z):
    return np.matmul(np.matmul(enc-Y, V[1:21,].T).T* Z * (1-Z), np.concatenate((np.ones((500,1)), training_set), axis=1)).T

# Confusion matrix calculator, input: Input image vector, correct labels
def calc_Conf(inp, realLabels):
    processed_labels = np.concatenate([list(map(int, el)) for el in realLabels])-1
    Z = sigmoid(np.matmul(np.concatenate((np.ones(shape = (500,1)), inp), axis =1), W).T)
    Y = softmax(np.matmul(V.T,np.concatenate((np.ones(shape = (1,500)),Z), axis = 0)).T)
    predicted = np.argmax(Y, axis=1)
    return confusion_Matrix(predicted, processed_labels)

# Confusion matrix input: correct labels, predicted labels
def confusion_Matrix(labels, predicted):
    #normalize indices, convert to int 
    mat = np.zeros((5,5))
    for i in range(0, len(labels)):
        mat[labels[i], predicted[i]]+=1
    return mat


def main():
    global enc, images, labels, training_set, training_out, test_out, test_set, N, K, enc, V, W, eta, eps, max_iteration, Z, Y, obj_vals

    images = np.array(read_csv("images.csv"))
    labels = np.array(read_csv("labels.csv"))

    # Split the data
    # Training/learning set
    training_set = images[0:500]
    training_out = labels[0:500]

    # test set
    test_set = images[500:1000]
    test_out = labels[500:1000]

    # Number of training images
    N = 500
    # Number of classes
    K = int(np.max(training_out))

    # One of K encoding
    enc = np.zeros((N, K))
    enc[np.arange(N), np.concatenate([list(map(int, el)) for el in training_out]) - 1] = 1

    V = np.array(read_csv("initial_V.csv"))
    W = np.array(read_csv("initial_W.csv"))

    eta = 0.0005
    eps = 1e-3
    max_iteration = 500

    obj_vals = []
    # feed forward
    Z = sigmoid(np.matmul(np.concatenate((np.ones(shape=(500, 1)), training_set), axis=1), W).T)
    Y = softmax(np.matmul(V.T, np.concatenate((np.ones(shape=(1, 500)), Z), axis=0)).T)

    obj_value = -np.sum(enc * safelog(Y))
    obj_vals.append(obj_value)

    for i in range(0, max_iteration):
        # update, back propagation
        # V = V + eta * gradV(Y, Z)
        W = W + eta * gradW(Y, V, Z)
        V = V + eta * gradV(Y, Z)

        # feed forward
        Z = sigmoid(np.matmul(np.concatenate((np.ones(shape=(500, 1)), training_set), axis=1), W).T)
        Y = softmax(np.matmul(V.T, np.concatenate((np.ones(shape=(1, 500)), Z), axis=0)).T)

        # update errors
        obj_value = -np.sum(enc * safelog(Y))
        obj_vals.append(obj_value)

        if (i != 0 and abs(obj_vals[i + 1] - obj_vals[i]) < eps):
            break

    #print the confusion matrices
    print("Training set confusion matrix")
    print(calc_Conf(training_set, training_out))

    print("Testing set confusion matrix")
    print(calc_Conf(test_set, test_out))

    #plot the graphs
    pt.plot(obj_vals)
    pt.ylabel("Error")
    pt.xlabel("Iteration")
    pt.show()

if __name__ == '__main__':
    main()