from functools import partial
from model.helper import *
import jax


# the forward and loss function are located outside the Lstm class to avoid jax conflicts
# LSTM forwarding
@jax.jit
def forward(params, states, X):
    Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by = params
    def step(prev_state, x):
        h_prev, c_prev = prev_state
        input = jnp.concatenate((h_prev, x))
        F = sigmoid(Wf @ input + bf)
        I = sigmoid(Wi @ input + bi)
        C = jnp.tanh(Wc @ input + bc)
        O = sigmoid(Wo @ input + bo)
        c = F * c_prev + I * C
        h = O * jnp.tanh(c)
        output = Wy @ h + by
        return (h, c), output
    states, outputs = jax.lax.scan(step, states, X[:, :, jnp.newaxis])
    # return states, outputs[:, :, 0]
    return states, outputs[-1, :, 0]
forward_batch = jax.vmap(forward, in_axes=(None, 0, 0))


# Mean Squared Error (MSE) loss function
@jax.jit
def mse_loss(params, states, X_batch, Y_batch):
    states, Y_batch_pred = forward_batch(params, states, X_batch)
    squared_diff = (Y_batch_pred - Y_batch) ** 2
    err = jnp.mean(squared_diff, axis=1) 
    loss_value = jnp.mean(err)
    return loss_value, states


class Lstm():
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_epoches, learning_rate, batch_size):
        
        # initialize hyperparameters
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimensions = [input_dim, hidden_dim, output_dim]
        
        # initialize forget, input, candidate, and output gate
        self.params = []
        for i in range(4):
            W = init_weights(hidden_dim, input_dim)
            b = jnp.zeros((hidden_dim, 1))
            self.params.extend([W, b])
        # initialize final gate
        self.params.append(init_weights(output_dim, hidden_dim))
        self.params.append(np.zeros((output_dim, 1)))
    
    
    def train_epoch(self, X, Y, num_batches, mode='TRAIN'):
        '''
        Training function for each epoch. 
        The parameter is only updated when model is in TRAINING mode.
        '''
        assert mode in ['TRAIN', 'EVAL'], f"{mode} mode is not recognized (use TRAIN / EVAL)!"
        epoch_losses = []
        # initialize state for each epoch
        states = (jnp.zeros((self.batch_size, self.dimensions[1], 1)), jnp.zeros((self.batch_size, self.dimensions[1], 1)))
        # training
        for batch in range(num_batches):
            # select batch training data 
            X_batch = X[batch * self.batch_size : (batch + 1) * self.batch_size]
            Y_batch = Y[batch * self.batch_size : (batch + 1) * self.batch_size]
            (loss, states), grads = jax.value_and_grad(mse_loss, has_aux=True)(self.params, states, X_batch, Y_batch)
            # update parameters
            if mode == 'TRAIN':
                self.params = [param - self.learning_rate * np.clip(grad, -1, 1) 
                               for param, grad in zip(self.params, grads)]
            epoch_losses.append(loss)
        return np.mean(epoch_losses)     

    def train(self, X_train, Y_train, X_val, Y_val):
        '''Overall training function'''
        print('Start training ...')
        N_train, N_val = X_train.shape[0], X_val.shape[0]
        num_train_batches, num_val_batches = N_train // self.batch_size, N_val // self.batch_size 
        # iterate epoches
        for i in range(self.num_epoches):
            train_loss = self.train_epoch(X_train, Y_train, num_train_batches, mode='TRAIN')
            val_loss = self.train_epoch(X_val, Y_val, num_val_batches, mode='EVAL')
            # display loss every 10 epoch
            if i % 10 == 0:
                print(f"Epoch {i+1}/{self.num_epoches}: training loss = {train_loss} | validation loss = {val_loss}")
        print('Training finished!')
    
    def predict(self, X_test):
        '''Predict the data at next time step given time series data'''
        # initialize state for each epoch
        states = (jnp.zeros((self.batch_size, self.dimensions[1], 1)), jnp.zeros((self.batch_size, self.dimensions[1], 1)))
        # batch training
        outputs = []
        for batch in range(X_test.shape[0] // self.batch_size):
            X_batch = X_test[batch * self.batch_size : (batch + 1) * self.batch_size]
            states, batch_outputs = forward_batch(self.params, states, X_batch)
            outputs.append(batch_outputs)
        return jnp.concatenate(outputs, axis=0)