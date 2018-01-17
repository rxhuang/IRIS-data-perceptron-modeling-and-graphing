
# coding: utf-8

# In[24]:


import numpy as np

# method that classifies a single point as positive or negative
def classify(w, b, x):
    result = np.dot(w,x)+b
    if(result>=0):
        return 1
    return -1

# method that uses perceptron method on data
def perceptron(x, y):
    w = [0]*len(x[0])
    b = 0
    z = zip(x,y)
    z = np.random.permutation(z)
    counter = 0
    numUpdate = 0
    while(counter<len(z)):
        x = z[counter][0]
        y = z[counter][1]
        counter += 1
        if(classify(w,b,x)*y<0):
            w = w + np.dot(x,y)
            b = b + y
            counter = 0
            numUpdate+=1
    return (w,b,numUpdate)


from sklearn import datasets

# import the iris data and use only 2 atributes 
iris = datasets.load_iris()
x = iris.data
y = iris.target

x = x[:,1:4:2]
x = x[:100]
y = y[:100]
y[:50] = -1

# perform perceptron method on iris data
o = []
for i in range(20):
    res = perceptron(x,y)
    o.append(res[2])

a,b = x.T

# create scatter plot of iris data
import matplotlib.pyplot as plt
plt.scatter(a[:50],b[:50])
plt.scatter(a[50:],b[50:])
n = np.linspace(1,5)
plt.plot(n,-res[0][0]/res[0][1]*n-res[1]/res[0][1])
plt.xlabel('sepal width')
plt.ylabel('petal width')
plt.show()

# create histogram of iris data 
fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1, 25, 1))
plt.hist(o, 23, color='b', alpha=0.5)
plt.xlabel('number of updates')
plt.ylabel('occurence in 20 trials')
plt.show()

