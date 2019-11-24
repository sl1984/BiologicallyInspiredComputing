from numpy import array

def getDataFromFile(filename):
    # training set input values
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # training set output values
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
    return training_set_inputs, training_set_outputs

