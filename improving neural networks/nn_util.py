# All functions are made on basis of a 3 layer NN

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets

def sigmoid(x) : 
    return (1/(1+np.exp(-x)))

def relu(x):
    return np.maximum(0,x)


def forward_prop(X,params) :

    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    w3 = params["w3"]
    b3 = params["b3"]

    # Linear -> Relu -> Linear -> Relu -> Linear -> Sigmoid

    z1 = np.dot(w1,X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3,a2) + b3
    a3 = sigmoid(z3)

    cache = (z1,a1,w1,b1,z2,a2,w2,b2,z3,a3,w3,b3)

    return a3,cache


def backward_prop(X,Y,cache) : 

    m = X.shape[1]
    (z1,a1,w1,b1,z2,a2,w2,b2,z3,a3,w3,b3) = cache

    dz3 = 1/m * (a3 - Y)
    dw3 = np.dot(dz3,a2.T)
    db3 = np.sum(dz3,axis = 1,keepdims = True)

    da2 = np.dot(w3.T,dz3)
    dz2 = np.multiply(da2,np.int64(a2 > 0))
    dw2 = np.dot(dz2,a1.T)
    db2 = np.sum(dz2,axis = 1,keepdims = True)

    da1 = np.dot(w2.T,dz2)
    dz1 = np.multiply(da1,np.int64(a1>0))
    dw1 = np.dot(dz1,X.T)
    db1 = np.sum(dz1,axis = 1,keepdims = True)

    grads = {"dz3": dz3, "dw3": dw3, "db3": db3,
            "da2": da2, "dz2": dz2, "dw2": dw2, "db2": db2,
            "da1": da1, "dz1": dz1, "dw1": dw1, "db1": db1}

    return grads

def update_parameters(params,grads,learning_rate) :

    L = len(params)//2

    for l in range(L):
        params["w" + str(l+1)] = params["w" + str(l+1)] - learning_rate * grads["dw" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return params


def compute_cost(a3,Y) : 

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    return loss

def predict(X,Y,params) :

    m = X.shape[1]
    p = np.zeros((1,m),dtype = np.int)

    # forward prop 
    a3,caches = forward_prop(X,params)

    # convert probability to predictions

    for i in range(0,a3.shape[1]) :
        if a3[0,i] > 0.5:
            p[0,i] = 1

        else:
            p[0,i] = 0

    print("Accuracy: " + str(np.mean((p[0,:] == Y[0,:]))))

    return p

def predict_dec(params,X) :
    """
    Used to plot decision boundary
    """

    a3,cache = forward_prop(X,params)
    predictions = (a3>0.5)
    return predictions

def plot_decision_boundary(model, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()







    
