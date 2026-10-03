import cascade_nn,matplotlib as plt,math.mean as mean

nn_tree = [cascade_nn.NN_model(32,4)]

class cascade_algorithm:
    def __init__(self,start_nn,relative_error_derivitive_threshold):
        self.nn_current = start_nn
        self.nn_layer_added = None#created when a new layer is added to the current neural network
        self.nn_neuron_added = None#created when a new neuron is added to the current neural network
        self.relative_error_derivitive_relative_threshold = relative_error_derivitive_threshold#used to determine when to add a new neuron or layer
        self.min_error = 0.01#used to determine when to stop training the neural network

        self.error_array = []#used when ploting the error over time for the cascade algorithm
        self.total_nuerons_array = []#used when ploting the number of neurons over time for the cascade algorithm
        self.total_layers_array = []#used when ploting the number of layers over time for the cascade algorithm
        self.total_nuerons = 1#used when ploting the number of neurons over time for the cascade algorithm
        self.total_layers = 1#used when ploting the number of layers over time for the cascade algorithm

    def train(self, input_array, expected_output,relative_step_size):

        #get initial error
        total_error_array = 0
        next_total_error = 0
        for _input in input_array:
            self.nn_current.predict(_input)
            total_error += cascade_nn.forward_pass_error(expected_output,self.nn_current.predict(_input))

        total_error /= len(input_array)
        self.error_array.append(total_error)
        self.total_nuerons_array.append(self.total_nuerons)
        self.total_layers_array.append(self.total_layers)

        #make change to the neural network and get new error
        self.nn_current.change_hidden_layer_weight(0,0,total_error*relative_step_size)

        #get error derivitive
        while next_total_error < self.min_error:
            for _input in input_array:
                self.nn_current.predict(_input)
                next_total_error += cascade_nn.forward_pass_error(expected_output,self.nn_current.predict(_input))

            next_total_error /= len(input_array)
            delta_error = next_total_error - total_error
            relative_error_derivitive_relative = delta_error / total_error


            #if the relative error derivitive is below the threshold, add a new neuron and layer to the neural network. then compare the new error of the neuroal network that had a neuron added to the neural network that had a layer added.
            #whichever has the lowest error will be the new current neural network. if both have the same error, then the neural network that had a neuron added will be the new current neural network
            if relative_error_derivitive_relative < self.relative_error_derivitive_relative_threshold:
                self.nn_current.change_hidden_layer_weight(0,0,next_total_error*relative_step_size)
                total_error = next_total_error
                self.error_array.append(total_error)
                self.total_nuerons_array.append(self.total_nuerons)
                self.total_layers_array.append(self.total_layers)

    def plot(self):
        # Implement plotting logic for the cascade algorithm
        pass