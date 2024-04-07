from model.helper import *

class Lstm():
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_epoches, learning_rate, batch_size):
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimensions = [input_dim, hidden_dim, output_dim]
        
        # initialize forget, input, candidate, output gate (same dimension)
        self.params = []
        for i in range(4):
            W = init_weights(input_dim, hidden_dim)
            b = np.zeros((hidden_dim, 1))
            self.params.extend([W, b])
        # initialize final gate
        self.append(init_weights(hidden_dim, output_dim))
        self.append(np.zeros((hidden_dim, 1)))
    
    @staticmethod
    def forward(params, hidden_dim, output_dim, X_batch):
        Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by = params
        N = X_batch.shape[0]
        hidden_states = {-1: np.zeros((hidden_dim, 1))}
        cell_states = {-1: np.zeros((hidden_dim, 1))}
        outputs = np.empty((N, output_dim))
        
        for i in range():
            input = np.concatenate((hidden_states[i - 1], X_batch[i]))
            # TODO: double check dimension!
            F = sigmoid(Wf @ input + bf)
            I = sigmoid(Wi @ input + bi)
            C = tanh(Wc @ input + bc)
            O = sigmoid(Wo @ input + bo)
            cell_states[i] = F * cell_states[i - 1] + I * C
            hidden_states[i] = O * tanh(cell_states[i])
            outputs[i] = Wy @ hidden_states[i] + by  # check need reshape or not
        return outputs

    @staticmethod
    @jax.jit
    def mse_loss(params, hidden_dim, output_dim, X_batch, Y_batch):
        Y_batch_pred = Lstm.forward(params, hidden_dim, output_dim, X_batch)
        return mse(Y_batch, Y_batch_pred)
    
    @staticmethod
    @jax.jit
    def loss_value_and_grad(params, hidden_dim, output_dim, X_batch, Y_batch):
        return jax.value_and_grad(Lstm.mse_loss)(params, hidden_dim, output_dim, X_batch, Y_batch)
    
    def train_batch(self, X, Y, num_batches, mode='TRAIN'):
        assert mode in ['TRAIN', 'EVAL'], f"{mode} mode is not recognized (use TRAIN / EVAL)!"
        epoch_losses = []
        # batch training
        for batch in range(num_batches):
            # select batch training data 
            X_batch = X[batch * self.batch_size : (batch + 1) * self.batch_size]
            Y_batch = Y[batch * self.batch_size : (batch + 1) * self.batch_size]
            # compute loss & gradient
            loss_value, grads = self.loss_value_and_grad(self.params, 
                                                        self.dimensions[1], 
                                                        self.dimensions[2], 
                                                        X_batch, 
                                                        Y_batch)
            # update parameters
            if mode == 'TRAIN':
                self.params = [param - self.learning_rate * grad[0] for param, grad in zip(self.params, grads)]
            epoch_losses.append(loss_value)
        return np.mean(epoch_losses)     

    def train(self, X_train, Y_train, X_val, Y_val):
        print('Start training ...')
        N_train, N_val = X_train.shape[0], X_val[0]
        num_train_batches, num_val_batches = N_train // self.batch_size + 1, N_val // self.batch_size + 1
        # iterate epoches
        for i in range(self.num_epoches):
            train_loss = self.train_batch(X_train, Y_train, num_train_batches, mode='TRAIN')
            val_loss = self.train_batch(X_val, Y_val, num_val_batches, mode='EVAL')
            # display loss every 10 epoch
            if i % 10 == 0:
                print(f"Epoch {i+1}/{self.num_epoches}: training loss = {train_loss} | validation loss = {val_loss}")
        print('Training finished!')
    
    def predict(self, X_test):
        return self.forward(self.params, self.dimensions[1], self.dimensions[2], X_test)