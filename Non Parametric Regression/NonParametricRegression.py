#!/usr/bin/env python
# coding: utf-8
from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def read_csv(filename):
    with open(filename, newline='') as f_input:
        reader = csv.reader(f_input)
        next(reader, None) #skip header
        return [list(map(float, row)) for row in reader]

# h value
h = 0.37

def RMSE(testData, binBounds, vals):
    sum = 0
    for el in testData:
        idx = np.digitize(el[0], binBounds)-1
        sum += (el[1]-vals[idx])*(el[1]-vals[idx])
    sum/=testData.shape[0]
    return math.sqrt(sum)

# Maxx and minn find the max and min of two elements, respectively
def maxx(a , b):
    if a > b:
        return a
    return b
def minn(a, b):
    if a <b:
        return a
    return b


#since h is 0.37 and the corresponding x values (Eruption time + 0.37/2) are not exactly in the calculated 
#values for the graph, we need to calculate them
def RMSEKernel(testData, training_set, binBounds, vals):
    sum = 0
    for el in testData:
        estval = 0
        num = 0
        den = 0
        for pt in training_set:
            tmp = (1/math.sqrt(2*math.pi))*math.exp(-0.5*(pt[0]-el[0])**2/h**2)
            num += tmp*pt[1]
            den += tmp
        estval = num/den
        sum += (estval-el[1])**2
    sum/=testData.shape[0]
    return math.sqrt(sum)


def main():
    # Load data. 
    dataset = np.array(read_csv("data_set.csv"))

    # split data to training and test sets
    trainingset = dataset[:150]
    testset = dataset[150:]
    
    
    mn = np.min(dataset[:, 0])
    mx = np.max(dataset[:, 0])

    binBounds = np.arange(1.5, mx + h, h)
    
    # regressogram
    tmp = np.zeros(((int)((mx - mn + h) / h), 2))  # to hold the counts and the sums of the buckets
    for el in trainingset:
        idx = np.digitize(el[0], binBounds) - 1
        tmp[idx, 0] += el[1]
        tmp[idx, 1] += 1
    regressogramData = tmp[:, 0] / tmp[:, 1]
    
    print("Regressogram => RMSE is", RMSE(testset, binBounds, regressogramData), "when h is ", h)
    
    # Plot the regressogram
    plt.scatter(trainingset[:, 0], trainingset[:, 1], label="Train", color='Blue')
    plt.scatter(testset[:, 0], testset[:, 1], label="Test", color='Red')
    plt.step(binBounds, np.append(regressogramData, regressogramData[len(regressogramData) - 1]), where='post',
             color='black')
    plt.ylabel('Waiting Time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title('h = {}'.format(h))
    plt.legend()
    plt.show()

    # Running mean smoother
    data_pts = np.arange(1.5, 5.2, 0.01)
    a = np.zeros((data_pts.shape[0], 2))
    maxind = data_pts.shape[0] - 1
    for el in trainingset:
        #Middle index
        idx = (int)(100 * (el[0] - 1.5))
        #Add value to sum and increment the count of right boundary
        a[maxx((int)(idx - 19), 0), 0] += el[1]
        a[maxx((int)(idx - 19), 0), 1] += 1
        #Subtract value from sum and decrement the count of left boundary
        a[minn((int)(idx + 19), maxind), 0] -= el[1]
        a[minn((int)(idx + 19), maxind), 1] -= 1
    # to not get 0 in denom
    a[maxind, 0] = 0
    a[maxind, 1] = 0
    running_mean = np.zeros((data_pts.shape[0], 1))
    cursum = 0.0
    curcnt = 0.0
    for i in range(data_pts.shape[0]):
        cursum += a[i, 0]
        curcnt += a[i, 1]
        running_mean[i] = cursum / curcnt

    # Plot the running mean smoother
    plt.scatter(trainingset[:, 0], trainingset[:, 1], label="Train", color='Blue')
    plt.scatter(testset[:, 0], testset[:, 1], label="Test", color='Red')
    plt.step(data_pts, running_mean, where='post', color='black')
    plt.ylabel('Waiting Time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title('h = {}'.format(h))
    plt.legend()
    plt.show()

    print("Running mean smoother => RMSE is", RMSE(testset, data_pts, running_mean), "when h is ", h)

    # KERNEL SMOOTHER

    kernel_smoother = np.zeros(data_pts.shape[0])
    for i in range(data_pts.shape[0]):
        num = 0
        den = 0
        for el in trainingset:
            tmp = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (el[0] - data_pts[i]) ** 2 / h ** 2)
            num += tmp * el[1]
            den += tmp
        kernel_smoother[i] = num / den

    # Plot the kernel smoother
    plt.scatter(trainingset[:, 0], trainingset[:, 1], label="Train", color='Blue')
    plt.scatter(testset[:, 0], testset[:, 1], label="Test", color='Red')
    plt.step(data_pts, kernel_smoother, where='post', color='black')
    plt.ylabel('Waiting Time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title('h = {}'.format(h))
    plt.legend()
    plt.show()

    print("Kernel Smoother => RMSE is", RMSEKernel(testset,trainingset, data_pts, kernel_smoother), "when h is ", h)


if __name__ == '__main__':
    main()
