import numpy as np

def sigmoid(x): 
    return (1/(1 + np.exp(-x))

def softmax(x): 
    ex = np.exp(x - np.max(x))
    return ex/(ex.sum(axis=0))


def initialize_adam(params):
    """ 
    Initializez two python dictionaries v and s :
    with keys : "dw1","db1"...."dwL","dbL"
    with values : numpy arrays of zeros
    where v contains exponentially weighted average of gradient
    and s contains exponentially weighted average of the squared gradient
    """


    L = len(params)//2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(params["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(params["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(params["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(params["b" + str(l+1)].shape)
        

    return v,s


def update_params_with_adam(params,grads,v,s,t,learning_rate = 0.01, beat1 = 0.9, beat2 = 0.999, epsilon = 1e-8):

    L = len(params)//2

    v_corrected = {}
    s_corrected = {}


    for l in range(L):
    
        # Moving average of gradients
        v['dW' + str(l+1)] = beta1 * v['dW' + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
        v['db' + str(l+1)] = beta1 * v['db' + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]


        # Computing bias-corrected first moment estimates
        v_corrected['dW' + str(l+1)] = v['dW' + str(l+1)]/(1-beta1**t)
        v_corrected['db' + str(l+1)] = v['db' + str(l+1)]/(1-beta1**t)

        # Moving average of squared gradients
        s['dW' + str(l+1)] = beta2 * s['dW' + str(l+1)] + (1-beta2) *(grads['dW' + str(l+1)]**2)
        s['db' + str(l+1)] = beta2 * s['db' + str(l+1)] + (1-beta2) *(grads['db' + str(l+1)]**2)

        # Computing the bias-corrected second raw moment estimates
        s_corrected['dW' + str(l+1)] = s['dW' + str(l+1)]/(1-beat2**t)
        s_corrected['db' + str(l+1)] = s['db' + str(l+1)]/(1-beat2**t)


        # Updating parameters
        params['W' + str(l+1)] = params['W' + str(l+1)] - learning_rate * v_corrected['dW' + str(l+1)]/np.sqrt(s_corrected['dW' + str(l+1)] + epsilon)
        params['b' + str(l+1)] = params['b' + str(l+1)] - learning_rate * v_corrected['db' + str(l+1)]/np.sqrt(s_corrected['db' + str(l+1)] + epsilon)


        return params,v,s




        
    
