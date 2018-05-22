
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from scipy.spatial import distance


def quad_kernel_perceptron(x, y):
    a = [0] * len(y)
    b = 0
    count = 0
    while count < len(y):
        s = 0
        for i in range(len(y)):
            s += a[i] * y[i] * (1 + np.dot(x[i], x[count])) * (1 + np.dot(x[i], x[count]))
            s += b
            if y[count] * s <= 0:
                a[count] += 1
                b += y[count]
                count = 0
            else:
                count += 1
    return a, b


def rbf_kernel_perceptron(x, y, sigma=100.0):
    a = [0] * len(y)
    b = 0
    count = 0
    while count < len(y):
        s = 0
        for i in range(len(y)):
            d = distance.euclidean(x[i], x[count])
            #d = np.linalg.norm(x[i], x[count])
            k = np.exp(-d*d/2*sigma*sigma)
            s += a[i] * y[i] * k
        s += b
        if y[count] * s <= 0:
            a[count] += 1
            b += y[count]
            count = 0
        else:
            count += 1
    return a, b


def predictt(a, b, x, y, p):
    result = []
    for i in p:
        s = 0
        for j in range(len(y)):
            s += a[j] * y[j] * (1 + np.dot(x[j], i)) * (1 + np.dot(x[j], i))
            s += b
            result.append(np.sign(s))
    return result


def predictt2(a, b, x, y, p, sigma=100.0):
    result = []
    for i in p:
        s = 0
        for j in range(len(y)):
            d = distance.euclidean(x[j], i)
            #d = np.linalg.norm(x[i], x[count])
            k = np.exp(-d*d/2*sigma*sigma)
            s += a[j] * y[j] * k
            s += b
            result.append(np.sign(s))
    return result



def plot_this(a, b, x, y, delta, x1_min, x1_max, x2_min, x2_max):
    # Create mesh for plot
    delta = 0.05
    x1_min, x1_max = 0.0, 10.5
    x2_min, x2_max = 0.0, 10.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))
    #Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = np.asarray(predictt2(a, b, x, y, np.c_[xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)
    plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.Pastel2, vmin=0, vmax=2)



    # Plot also the training points
    cols = ['ro', 'k^', 'b*', 'wD']
    x1, x2 = zip(*x)
    lx1 = []
    lx2 = []
    for i in range(len(y)):
        if y[i] == 1:
            lx1.append(x1[i])
            lx2.append(x2[i])
    plt.plot(lx1, lx2, cols[0], markersize=8)
    lx1 = []
    lx2 = []
    for i in range(len(y)):
        if y[i] == -1:
            lx1.append(x1[i])
            lx2.append(x2[i])
    plt.plot(lx1, lx2, cols[1], markersize=8)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.show() 



# read data
data1 = []
with open('data1.txt', 'r') as f:
    for line in f:
        tmp = line.split()
        tmp = [int(i) for i in tmp]
        data1.append(tmp)


data2 = []
with open('data2.txt', 'r') as f:
    for line in f:
        tmp = line.split()
        tmp = [int(i) for i in tmp]
        data2.append(tmp)


x1, x2, y = zip(*data1)
x = zip(x1, x2)
x_1 = [np.array(i) for i in x]
y_1 = list(y)


x1, x2, y = zip(*data2)
x = zip(x1, x2)
x_2 = [np.array(i) for i in x]
y_2 = list(y)

a, b = rbf_kernel_perceptron(x_1, y_1)

plot_this(a, b, x_1, y_1, 0, 0, 0, 0, 0)

