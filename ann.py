from numpy import exp, array, random, dot, tanh, cos

class NeuralLayer():
    def __init__(self, neuron_config_values):
        self.synaptic_weights = 2 * random.random((neuron_config_values[1], neuron_config_values[0])) - 1
        self.config_values = neuron_config_values

class ArtificialNeuralNetwork():
    def __init__(self, layers, act_func):
        self.layers = layers
        self.layerOutputs = []
        self.activ_function = act_func

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

    # derivative of the Sigmoid function to check confident on existing weights
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Training neural network by adjusting the weights
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        layersCount = len(self.layers)
        for iteration in range(number_of_training_iterations):
            layerOutputValues = []
            layerOutputValues = self.process(training_set_inputs)
            layer_delta_array = []
            last_layer_error = training_set_outputs - layerOutputValues[layersCount-1]
            for index3 in range(layersCount, 0, -1):
                if index3 < layersCount :
                    last_layer_error = layer_delta.dot(self.layers[index3].synaptic_weights.T)
                layer_delta = last_layer_error * self.sigmoid_derivative(layerOutputValues[index3-1])
                layer_delta_array.append(layer_delta)

            layer_delta_array.reverse()
            for index4 in range(layersCount):
                if index4 == 0:
                    layer_adjustment = training_set_inputs.T.dot(layer_delta_array[index4])
                else:
                    layer_adjustment = layerOutputValues[index4-1].T.dot(layer_delta_array[index4])

                # Adjust the weights.
                self.layers[index4].synaptic_weights += layer_adjustment

    def process(self, inputs):
        layerOutputs = []
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

if __name__ == "__main__":

    random.seed(1)

    # Provide multiple layer details
    layersValues = array([[4,3],[3,4],[2,3],[1,2]])
    layers = []
    for index in range(len(layersValues)):
        print ('layer configs :', layersValues[index])
        layers.append(NeuralLayer(layersValues[index]))

    # Activation functions
    # Null -> 0
    # Sigmoid -> 1
    # Hyperbloic Tan -> 2
    # Cosine -> 3
    # Gaussian -> 4

    activation_function = 1

    # Create neural network
    ann = ArtificialNeuralNetwork(layers, activation_function)
    ann.print_layer_weights()

    # training set input values
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # training set output values
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # train the network
    ann.train(training_set_inputs, training_set_outputs, 80000)

    #ann.print_layer_weights()
    # Test the neural network with a new input.
    op = ann.process(array([1, 1, 0]))
    print ("Output with the trained ANN [1, 1, 0] -> : %f" % op[-1])
