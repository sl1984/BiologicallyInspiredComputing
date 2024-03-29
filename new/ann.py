from numpy import exp, array, random, dot, tanh, cos
from utils import *

class NeuralLayer():
    def __init__(self, neuron_config_values):
        self.synaptic_weights = []
        self.config_values = neuron_config_values

class ArtificialNeuralNetwork():
    def __init__(self, ann_layer_config, act_func, train_input, train_output):
        self.layers = []
        for index in range(len(ann_layer_config)):
            self.layers.append(NeuralLayer(ann_layer_config[index]))
        self.ann_layer_configs = ann_layer_config
        self.layerOutputs = []
        self.activ_function = act_func
        self.training_inputs = train_input
        self.training_outputs = train_output
        self.ann_error = []
        self.ann_output = []

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
        self.ann_output = layerOutputValues[layersCount-1]
        self.mean_square_value()

    def mean_square_value(self):
        sum = 0
        sample_count = len(self.ann_output)
        for i in range(sample_count):
            sum = sum + ((self.training_outputs[i] - self.ann_output[i])**2)
        self.mse = sum / sample_count
        #print (self.mse)

    def process(self):
        layerOutputs = []
        inputs = self.training_inputs
        for index1 in range(len(self.layers)):
            #layerOutput = []
            layerOutput = self.activation_function_call(dot(inputs, self.layers[index1].synaptic_weights))
            inputs = layerOutput
            layerOutputs.append(layerOutput)
        return layerOutputs

    def print_layer_weights(self):
        for index2 in range(len(self.layers)):
            print ( 'ANN Layer Number:', format(index2+1))
            print ( 'ANN Layer Details: %d Neurons %d Inputs' % (self.layers[index2].config_values[0], self.layers[index2].config_values[1]))
            print (self.layers[index2].synaptic_weights)

    def set_input_values(self, t_input):
        self.training_inputs = t_input
