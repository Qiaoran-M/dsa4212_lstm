import jax
import jax.numpy as jnp
import numpy as np


# activation functions
def softmax(x):
    '''softmax function with normalization'''
    exp_x = jnp.exp(x - jnp.max(x))
    return exp_x / jnp.sum(exp_x)


def sigmoid(x):
    '''sigmoid activation function'''
    return 1 / (1 + jnp.exp(-x))


def tanh(x):
    '''tanh activation function'''
    return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))


# matrix weight initialization
def init_weights(dim1, dim2=None):
    if dim2 is not None:
        A = np.random.normal(0, 1, size=(dim1, dim2)) / np.sqrt(dim2)
    else:
        A = np.random.normal(0, 1, size=(dim1, 1))
    return A


# mse calculation
def mse(Y_true, Y_pred):
    squared_diff = (Y_pred - Y_true) ** 2
    err = np.mean(squared_diff, axis=1)  # N*1
    return np.mean(err)  # scalar





