from numpy import exp, array, random, dot, tanh, cos
from utils import *

class NeuralLayer():
    def __init__(self, neuron_config_values):
        self.synaptic_weights = 2 * random.random((neuron_config_values[1], neuron_config_values[0])) - 1
        self.config_values = neuron_config_values

class ArtificialNeuralNetwork():
    def __init__(self, ann_layer_config, act_func, train_input, train_output):
        self.layers = []
        for index in range(len(ann_layer_config)):
            print('layer configs :', ann_layer_config[index])
            self.layers.append(NeuralLayer(ann_layer_config[index]))
        self.ann_layer_configs = ann_layer_config
        self.layerOutputs = []
        self.activ_function = act_func
        self.training_inputs = train_input
        self.training_outputs = train_output
        self.ann_error = []

    # sigmoid function to normalise them between 0 and 1.
    def activation_sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def activation_null(self, x):
        return 0

    def activation_hyper_tangent(self, x):
        return tanh(x)

    def activation_cosine(self, x):
        return cos(x)

    def activation_gaussian(self, x):
        return (exp(-((x**2)/2)))

    def activation_function_call(self, x):
        if (self.activ_function == 0):
            return self.activation_null(x)
        elif (self.activ_function == 1):
            return self.activation_sigmoid(x)
        elif (self.activ_function == 2):
            return self.activation_hyper_tangent(x)
        elif (self.activ_function == 3):
            return self.activation_cosine(x)
        elif (self.activ_function == 4):
            return self.activation_gaussian(x)

    #convert vector position values in to layer weight arrays
    def set_weights_from_position(self, position):
        position1 = position
        all_layers_weights = []
        #break the list in to layer specific 2D array
        all_layers_weights = []
        k = 0
        for layer_config in self.ann_layer_configs:
            position_count = dot(layer_config[0], layer_config[1])
            layer_weights = []
            for j in range(layer_config[1]):
                row = []
                for i in range(layer_config[0]):
                    row.append(position1[k])
                    k += 1
                layer_weights.append(row)
            all_layers_weights.append(layer_weights)

        #Assign the 2D weights to all layers
        for index1 in range(len(self.layers)):
            self.layers[index1].synaptic_weights = all_layers_weights[index1]


    # Training neural network by adjusting the weights
    def forward_inside_ann(self):
        layersCount = len(self.layers)
        layerOutputValues = []
        layerOutputValues = self.process()
        last_layer_error = self.training_outputs - layerOutputValues[layersCount-1]
        self.ann_error = last_layer_error
        print ("last layer error")
        print (last_layer_error)

    def process(self):
        layerOutputs = []
        inputs = self.training_inputs
        for index1 in range(len(self.layers)):
            layerOutput = []
            layerOutput = self.activation_function_call(dot(inputs, self.layers[index1].synaptic_weights))
            inputs = layerOutput
            layerOutputs.append(layerOutput)
        return layerOutputs

    def print_layer_weights(self):
        for index2 in range(len(self.layers)):
            print ( 'ANN Layer Number:', format(index2+1))
            print ( 'ANN Layer Details: %d Neurons %d Inputs' % (self.layers[index2].config_values[0], self.layers[index2].config_values[1]))
            print (self.layers[index2].synaptic_weights)

# if __name__ == "__main__":
#
#     random.seed(1)
#     #ann_layer_config = array([[4, 3], [3, 4], [2, 3], [1, 2]])
#     ann_layer_config = [[4, 3], [1, 4]]
#     # Activation functions
#     # Null -> 0
#     # Sigmoid -> 1
#     # Hyperbloic Tan -> 2
#     # Cosine -> 3
#     # Gaussian -> 4
#
#     activation_function = 1
#     data = getDataFromFile('xyz')
#     position =  []
#     for i in range(16):
#         position.append(random.uniform(0,1))
#     print (position)
#     position1 = position
#     # Create neural network
#     ann =  ArtificialNeuralNetwork(ann_layer_config, activation_function, data[0], data[1])
#     ann.set_weights_from_position(position)
#     ann.print_layer_weights()
#
#     # train the network
#     ann.forward_inside_ann()







    #ann.print_layer_weights()
    # Test the neural network with a new input.
    #op = ann.process(array([1, 1, 0]))
    #print ("Output with the trained ANN [1, 1, 0] -> : %f" % op[-1])
