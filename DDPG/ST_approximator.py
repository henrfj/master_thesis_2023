"""
State transition (ST) approximator class

- Uses an aritifical neural network to estimate the residuals in the state transition model
"""
# ANN
import tensorflow as tf
from tensorflow import keras as ker
# Data
import pandas as pd
import numpy as np

class ST_apprixmator():

    def __init__(self, max_size, input_shape, layer_dimensions, learning_rate=0.1) -> None:
        
        self.mem_size = max_size
        self.mem_cntr = 0
        self.full_memory = False
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layer_dimensions = layer_dimensions
        self.n_layers = len(self.layer_dimensions)
        # INPUT
        self.input_memory = np.zeros((self.mem_size, *input_shape))
        # OUTPUT
        self.output_memory = np.zeros((self.mem_size, *input_shape))


    def create_model(self):
        # Sequential model.
        model = ker.Sequential()
       
        # Input layer.
        model.add(ker.Input(shape=(self.input_size, )))
        
        # A number of hidden layers
        for i in range(self.n_layers):
            # Adds dense layers of different dimensions.
            model.add(ker.layers.Dense(self.layer_dimensions[i], activation=tf.nn.relu))
        
        # Output layer.
        # Activation "tanh" works best. 
        model.add(ker.layers.Dense(self.input_shape, activation=tf.nn.relu))

        # Compiling the model.
        model.compile(optimizer=ker.optimizers.Adam(learning_rate=self.learning_rate), loss=ker.losses.MeanSquaredError(), metrics=['accuracy'])
        
    def store_datapair(self, input, output):
        # store datapairs in 
        index = self.mem_cntr % self.mem_size
        # Is the memory full?
        if not self.full_memory and self.mem_cntr > self.mem_size:
            self.full_memory = True 
        #
        self.input_memory[index] = input
        self.output_memory[index] = output
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Pich a batch sized sample from buffer
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        inputs = self.input_memory[batch]
        outputs = self.output_memory[batch]
        return inputs, outputs
    

