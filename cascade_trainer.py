import cascade_nn,matplotlib as plt

nn_tree = [cascade_nn.NN_model(32,4)]

class cascade_algorithm:
    def __init__(self,start_nn,relative_error_delta_relative_threshold):
        self.nn_current = start_nn
        self.nn_layer_added = None#created when a new layer is added to the current neural network
        self.nn_neuron_added = None#created when a new neuron is added to the current neural network
        self.relative_error_delta_relative_threshold = relative_error_delta_relative_threshold#used to determine when to add a new neuron or layer
        self.min_error = 0.01#used to determine when to stop training the neural network

        self.error_array = []#used when ploting the error over time for the cascade algorithm
        self.total_nuerons_array = []#used when ploting the number of neurons over time for the cascade algorithm
        self.total_layers_array = []#used when ploting the number of layers over time for the cascade algorithm
        self.total_nuerons = 1#used when ploting the number of neurons over time for the cascade algorithm
        self.total_layers = 1#used when ploting the number of layers over time for the cascade algorithm

    def train(self, input_array, expected_output_array,relative_step_size,epoch):

        if len(input_array) != len(expected_output_array):
            raise ValueError("Input array and expected output array must be the same length")

        if isinstance(input_array[0],list) == False: raise ValueError("Input array must be a list of lists")
        elif isinstance(expected_output_array[0],list) == False: raise ValueError("Expected output array must be a list of lists")

        #get initial error
        mean_error = 0
        next_mean_error = 0
        adjustment_direction = 1
        
        def sub_train():
            #get the output of the neural network for the n inputs
            output_array = []
            for _ in range(n):
                nn_output = self.nn_current.predict(input_array[n])
                output_array.append(nn_output)
            return output_array
            
        for n in range(len(input_array)):
            for branck in [self.nn_layer_added, self.nn_neuron_added]:
                for l in range(len(self.nn_current.hidden_layers_array)):
                    for neuron in range(self.nn_current.hidden_layers_array[l].neurons):
                        for x in range(self.nn_current.num_token_attributes):
                            for y in range(self.nn_current.num_token_attributes):

                                #get the output of the neural network for the current input
                                mean_error = cascade_nn.forward_pass_error(sub_train(),expected_output_array[:,n])/n
                                self.error_array.append(mean_error)
                                self.total_nuerons_array.append(self.total_nuerons)
                                self.total_layers_array.append(self.total_layers)

                                #make change to the neural network and get new error
                                self.nn_current.change_hidden_layer_weight(x,y,mean_error*relative_step_size)

                                #get error derivitive and direction
                                next_mean_error = cascade_nn.forward_pass_error(sub_train(),expected_output_array[:,n])/n

                                if next_mean_error > mean_error:
                                    adjustment_direction = -adjustment_direction#triggers UNARY_NEGATIVE, which is the most optimised way to swich signs
                                    #undo the change to the neural network if the error increased
                                    self.nn_current.change_hidden_layer_weight(x,y,-mean_error*relative_step_size)
                                    #reset the mean error to the previous value
                                    next_mean_error = 0

                                delta_error = next_mean_error - mean_error
                                relative_error_delta = delta_error / mean_error
                                current_epoch = 0

                                #continue to make changes to the neural network until the error is no longer decreasing
                                while next_mean_error < mean_error and current_epoch < epoch:

                                    self.nn_current.change_hidden_layer_weight(x,y,mean_error*relative_step_size)
                                    next_mean_error = cascade_nn.forward_pass_error(sub_train(),expected_output_array[:,n])/n
                                    delta_error = next_mean_error - mean_error
                                    relative_error_delta = delta_error / mean_error
                                    current_epoch += 1

            #if the relative error delta is below the threshold, add a new neuron and layer to the neural network. then compare the new error of the neuroal network that had a neuron added to the neural network that had a layer added.
            #whichever has the lowest error will be the new current neural network. if both have the same error, then the neural network that had a neuron added will be the new current neural network
            if relative_error_delta < self.relative_error_delta_relative_threshold:
                #copy the current neural network
                self.nn_layer_added = self.nn_current
                self.nn_neuron_added = self.nn_current

                #add a new layer to the neural network
                self.nn_layer_added.add_layer()

                #add a new neuron to the neural network
                self.nn_neuron_added.add_neuron()

                #track the number of neurons and layers in the neural network relative to the error for ploting
                self.error_array.append(mean_error)
                self.total_nuerons_array.append(self.total_nuerons)
                self.total_layers_array.append(self.total_layers)


        if mean_error < self.min_error:
            print(f"Training complete. Mean error: {mean_error}. Minimum error reached.")
            return
        else:
            print(f"Training complete. Mean error: {mean_error}. Minimum error not reached.")
            return

    def plot(self):
        # Implement plotting logic for the cascade algorithm
        pass