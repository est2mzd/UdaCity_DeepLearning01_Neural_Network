import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # np.random.normal( loc=mean, scale=std, size=output_size)
        
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        # Hint : lambda 引数: 返り値
        self.activation_function = lambda x : 1/(1+np.exp(- np.clip(x,-10,10) ))
       #self.activation_function = lambda x : 1/(1+np.exp(-x))
        self.activation_output   = lambda x : x
        
        self.h1 = 0.0
        self.h2 = 0.0

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs  = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_output(final_inputs) # signals from final output layer

        return final_outputs, hidden_outputs

    # 8-5 p.78
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = (y- final_outputs) # Output layer error is the difference between desired target and actual output.

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error   # * final_outputs *( 1 - final_outputs )
        hidden_error      = np.dot( self.weights_hidden_to_output, output_error_term )
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        #
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        #
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        update_type = 1
        #
        if update_type == 1:
            # SGD
            self.weights_hidden_to_output += (self.lr/n_records) * delta_weights_h_o
            self.weights_input_to_hidden  += (self.lr/n_records) * delta_weights_i_h
        else:
            # RMSprop
            alpha   = 0.1
            self.h1 = self.h1 * alpha + delta_weights_h_o*delta_weights_h_o*(1-alpha)
            self.h2 = self.h2 * alpha + delta_weights_i_h*delta_weights_i_h*(1-alpha)
            #
            self.weights_hidden_to_output += (self.lr/n_records) * delta_weights_h_o / (np.sqrt(self.h1)+1e-7)
            self.weights_input_to_hidden  += (self.lr/n_records) * delta_weights_i_h / (np.sqrt(self.h2)+1e-7)

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        final_outputs, hidden_outputs = self.forward_pass_train(features)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
#iterations    = 2000
#learning_rate = 0.1
#hidden_nodes  = 10
#output_nodes  = 1

#iterations    = 2000
#learning_rate = 0.02
#hidden_nodes  = 5
#output_nodes  = 1

#iterations    = 3000
#learning_rate = 0.05
#hidden_nodes  = 10
#output_nodes  = 1

#iterations    = 2000
#learning_rate = 0.05
#hidden_nodes  = 5
#output_nodes  = 1

#------------------------- Submit Version
# iterations    = 2000
# learning_rate = 0.02
# hidden_nodes  = 5
# output_nodes  = 1

# Training loss: 0.521 ... Validation loss: 0.636
# iterations    = 7000
# learning_rate = 0.07
# hidden_nodes  = 5
# output_nodes  = 1

# Progress: 100.0% ... Training loss: 0.872 ... Validation loss: 1.328
# iterations    = 8000
# learning_rate = 0.07
# hidden_nodes  = 7
# output_nodes  = 1


# iterations    = 2000
# learning_rate = 0.7/2
# hidden_nodes  = 28
# output_nodes  = 1

# iterations    = 3000
# learning_rate = 0.05
# hidden_nodes  = 5
# output_nodes  = 1

#------------------------- Submit Version 2
iterations    = 7000
learning_rate = 0.02*128
hidden_nodes  = 5
output_nodes  = 1

# iterations    = 5104
# learning_rate = 0.05
# hidden_nodes  = 5
# output_nodes  = 1

















