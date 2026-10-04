import numpy as np

#libary description
#

class NN_model:
    def __init__(self,context_len,num_token_attributes):
        #input layer is the first layer of the hidden layer

        self.context_len = context_len#only used to initialize the first hidden layer, after that the hidden layers are determined by the cascade algorithm
        self.num_token_attributes = num_token_attributes
        self.hidden_layer_array = [hidden_layer(num_token_attributes)]
        self.output_layer_weights = np.ones((num_token_attributes,num_token_attributes),order='f')
        self.output_layer_bias = 1

    def add_layer(self):
        self.hidden_layer_array.append(hidden_layer(self.context_len))

    def add_neuron(self,layer_num):
        self.hidden_layer_array[layer_num].add_neuron()

    def change_output_layer_bias(self,value):
        self.output_layer_bias += value

    def change_output_layer_weights(self,attribute_row_index,attribute_col_index,value):
        self.output_layer_weights[attribute_row_index, attribute_col_index] += value

    def predict(self,input_array):

        if isinstance(input_array, np.ndarray) and np.shape(input_array)[1] == np.shape(self.num_token_attributes)[0]:

            hiddenput = self.hidden_layer_array[0]._pass(input_array)#default order c which is prefered for perf

            for layer in self.hidden_layer_array[1:-1]:
                hiddenput = layer._pass(hiddenput)

            output = np.matmul(hiddenput,self.output_layer_weights)#expected shape (num_token_attributes,num_token_attributes)

            return np.sum(output) + self.output_layer_bias#output a scalar value which is the sum of all outputs plus the bias

        else: raise TypeError(f'input array is not able to used. must be numpy array and num cols must be {self.num_token_attributes} not {input_array.shape[1]}')

    def change_hidden_layer_weight(self,layer_num,attribute_row_index,attribute_col_index,value):
        self.hidden_layer_array[layer_num].change_weight(attribute_row_index,attribute_col_index,value)

    def change_hidden_layer_bias(self,layer_num,method,value):
        self.hidden_layer_array[layer_num].change_bias_value(value)

class hidden_layer:
    def __init__(self,layer_len):#hidden layer is always square so output shape is identical to input shape

        if isinstance(layer_len, int):
            self.neurons = [np.ones((layer_len,layer_len),order='f')]#order f for better matmul perf

        else: raise TypeError('First arg was not a numpy array')

        self.bias = 1

    def _pass(self,input_array):

        layer_output = np.matmul(input_array,self.layer)
        layer_output += self.bias
        return layer_output

    def change_weight(self,attribute_row_index,attribute_col_index,method,value):

        self.layer[attribute_row_index, attribute_col_index] += value

        else: raise ValueError('not supported input for arg 2')
            

    def change_bias_value(self, value):

        if isinstance(self.bias,np.ndarray):

            self.bias += value

            else: raise ValueError('not supported input for arg 2')

    def add_neuron(self):
        self.layer.append(np.ones((np.shape(self.layer)[0],1),order='f'))#add a new row to the layer

    def remove_neuron(self,index):
        if isinstance(index,int) == False: raise TypeError('arg 1 must be int')
        if index < 0 or index >= np.shape(self.layer)[0]: raise ValueError('index out of bounds')
        self.layer = np.delete(self.layer,index,axis=0)#remove the row at the specified index

def forward_error(output_array,expected_output_array):
    """
    Calculates the forward error of a neural network model given an input array and expected output. 
    The function first checks if the input array is a numpy array and has the correct number of columns. 
    It then passes the input through the hidden layers of the model and calculates the output using the output layer weights and bias. 
    Finally, it returns the difference between the calculated output and expected output.
    """
    if len(output_array) != len(expected_output_array): raise ValueError("Input array and expected output array must be the same length")

    #check if the input array is a numpy array and has the correct number of columns
    for output in output_array:
        if isinstance(output, np.ndarray) == False or np.shape(output)[1] != np.shape(nn_model.num_token_attributes)[0]:
            raise TypeError(f'input array is not able to used. must be numpy array and num cols must be {nn_model.num_token_attributes} not {output.shape[1]}')

    dif = np.subtract(output_array, expected_output_array)
    return (dif**2)**(0.5)#return the root mean square error
