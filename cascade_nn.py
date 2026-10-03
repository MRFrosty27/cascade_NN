import numpy as np

#libary description
#

class NN_model:
    def __init__(self,context_len,num_token_attributes):

        self.context_len = context_len
        self.num_token_attributes = num_token_attributes
        self.hidden_layer_array = [hidden_layer(num_token_attributes)]
        self.output_layer_weights = np.ones((1,num_token_attributes),order='f')
        self.output_layer_bias = 1

    def add_layer(self):
        self.hidden_layer_array.append(hidden_layer(self.context_len))

    def change_output_layer_bias(self,method,value):
        if isinstance(method,str) == False: raise TypeError('arg 2 must be str')

        if method == 'add': self.output_layer_bias += value

        elif method == 'set': self.output_layer_bias = value

        else: raise ValueError('not supported input for arg 2')

    def predict(self,input_array):

        if isinstance(input_array, np.ndarray) and np.shape(input_array)[1] == np.shape(self.num_token_attributes)[0]:

            for layer in self.hidden_layer_array:
                hiddenput = layer._pass(input_array)#default order c which is prefered for perf

            output = np.matmul(hiddenput,self.output_layer_weights)#expected shape (context len,1)
            return np.sum(output) + self.output_layer_bias

        else: raise TypeError(f'input array is not able to used. must be numpy array and num cols must be {self.num_token_attributes} not {input_array.shape[1]}')


class hidden_layer:
    def __init__(self,layer_len):#hidden layer is always square so output shape is identical to input shape

        if isinstance(layer_len, int):
            self.layer = np.ones((layer_len,layer_len),order='f')#order f for better matmul perf

        else: raise TypeError('First arg was not a numpy array')

        self.bias = 1

    def _pass(self,input_array):

        layer_output = np.matmul(input_array,self.layer)
        layer_output += self.bias#supported datatypes are int,float and ndarray
        return layer_output

    def change_weight(self,index,method,value):

        if isinstance(method,str) == False: raise TypeError('arg 2 must be str')

        elif len(index) != 2: raise ValueError('index must contain x and y value')

        if method == 'add': self.layer[index] += value

        elif method == 'set': self.layer[index] = value

        else: raise ValueError('not supported input for arg 2')
            

    def change_bias_value(self,index, method, value):

        if isinstance(self.bias,np.ndarray):

            if isinstance(method,str) == False: raise TypeError('arg 2 must be str')

            elif len(index) != 1: raise ValueError('index must contain only a single value')

            if method == 'add': self.bias[index] += value

            elif method == 'set': self.bias[index] = value

            else: raise ValueError('not supported input for arg 2')

        else:

            if isinstance(method,str) == False: raise TypeError('arg 2 must be str')

            elif len(index) != 1: raise ValueError('index must contain only a single value')

            if method == 'add': self.bias += value

            elif method == 'set': self.bias = value

            else: raise ValueError('not supported input for arg 2')

    def change_bias_datatype(self):

        if isinstance(self.bias,np.ndarray): self.bias = np.mean(self.bias)

        else: self.bias = np.full((self.layer.shape[0],1),self.bias)#shape (n,1)