import numpy as np
import scipy.special as ssp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


class neuralNetwork():
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: ssp.expit(x)
    
    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        #calculate 
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # calculate the errors for the output and for the hidden_nodes
        outputs_error = targets - final_outputs
        hidden_error = np.dot(self.who.T, outputs_error)

        # backpropagation
        self.who += self.lr * np.dot(outputs_error*final_outputs*(1-final_outputs), hidden_outputs.T)
        self.wih += self.lr * np.dot(hidden_error*hidden_outputs*(1-hidden_outputs), inputs.T)
    
    def query(self, input_list):        
        inputs = np.array(input_list, ndmin=2).T
        #calculate 
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        

if __name__=='__main__':
    
    input_nodes = 784
    hidden_nodes = 800
    output_nodes = 10
    learning_rate = 0.05
    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    path = r'/home/anton/Schreibtisch/DataScienceTraining/01_basics/mnist_data/mnist_train.csv'
    
    # training the neural network
    training_data_file = open(path, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    epoch = 5
    for e in range(epoch):/home/anton/Schreibtisch/DataScienceTraining/01_basics/pagseoTF/mnist_data
        for record in training_data_list:
            all_values = record.split(',')
            inputs = np.asfarray(all_values[1:])/255*0.99+0.01
            targets = np.zeros(output_nodes) +0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)
#            inputs_plus10 = rotate(inputs.reshape((28,28)), 10, cval=0.01, reshape=False)
#            nn.train(inputs_plus10.reshape(784), targets)
#            inputs_minus10 = rotate(inputs.reshape((28,28)), -10, cval=0.01, reshape=False)
#            nn.train(inputs_minus10.reshape(784), targets)            
        print('epoch: {} done.'.format(e))
    
    #%%
    with open(r'/home/anton/Schreibtisch/DataScienceTraining/01_basics/mnist_data/mnist_test.csv', 'r') as test_data_file:
        test_data_list = test_data_file.readlines()
        accuracy = []
        for record in test_data_list:
            all_values = record.split(',')
            inputs = np.asfarray(all_values[1:])/255*0.99+0.01
            targets = all_values[0]
            
            
            output = nn.query(inputs)
            probability = output.max()/output.sum()
            if int(targets) == np.argmax(output):
                accuracy.append(1)
            else: 
                accuracy.append(0)
#            print('correct, network, probability = {}, {}, {}'.format(targets,np.argmax(output),probability))
    print('Test error rate: {}'.format(100*(1-np.sum(accuracy)/len(accuracy))))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    