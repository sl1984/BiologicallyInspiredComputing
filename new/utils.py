from numpy import array, asarray

def getDataFromFile(data_set_name):

    if (data_set_name == 0):
        file_name = '1in_cubic.txt'
    elif (data_set_name == 1):
        file_name = '1in_linear.txt'
    elif (data_set_name == 2):
        file_name = '1in_sine.txt'
    elif (data_set_name == 3):
        file_name = '1in_tanh.txt'
    elif (data_set_name == 4):
        file_name = '2in_complex.txt'
    elif (data_set_name == 5):
        file_name = '2in_xor.txt';
    else:
        filename = '1in_cubic.txt'

    training_set_inputs = []
    training_set_outputs = []
    f = open("Data/" + file_name, "r")
    for x in f:
        y = x.split()
        if(len(y) == 2):
            training_set_inputs.append([float(y[0])])
            training_set_outputs.append(float(y[1]))
        elif(len(y) == 3):
            ip_value = []
            ip_value.append(float(y[0]))
            ip_value.append(float(y[1]))
            training_set_inputs.append(ip_value)
            training_set_outputs.append(float(y[2]))

    tsi = array(training_set_inputs)
    tso = (array([training_set_outputs])).T
    f.close()
    return tsi, tso
    # training set input values
    #training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # training set output values
    #training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
    #return training_set_inputs, training_set_outputs
