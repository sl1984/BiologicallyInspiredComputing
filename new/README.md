
#********************************************************
# Update the configuration in the start of pso.py files.
#********************************************************

# ANN & PSO (Particle Settings)

Note : provide ANN layer based on input.
For example for data set 0,1,2,3 the training input will 1 so set the ANN layers like => array([[2,1],[2,2],[1,2]])
For data set 4 & 5 training input value will be 2. So set the ANN layers like => array([[2,2],[3,2],[1,3]])

ann_layer_config = array([[2,2],[2,2],[1,2]])

# Choose the Activation functions Null -> 0 , Sigmoid -> 1, Hyperbloic Tan -> 2, Cosine -> 3, Gaussian -> 4
activation_function = 4

# Choose the Data set file cubic -> 0 , linear -> 1, sine -> 2, tanh -> 3, complex -> 4, xor -> 5
data_set = 4

# Hyper Parameters
# Set velocity proportion
vp = 0.5

# Set personal best proportion
pbp = 0.2 

#Set global best proportion
gbp = 0.1 

#Set velocity jump size
jp = 0.05 

# Set the PSO iteration
pso_iterations = 100

# Set the number of PSO particles
pso_particles = 40

#**********************************************************

# To run the code, run below from command line

$python3.6 pso.py

#**********************************************************

Reference:

https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
https://towardsdatascience.com/particle-swarm-optimisation-in-machine-learning-b01b1d2ad8a8
