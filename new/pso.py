from numpy import exp, array, dot, tanh, cos, asarray
from ann import ArtificialNeuralNetwork, NeuralLayer
from utils import *
import random
from sys import exit

vp = 0.5 #velocity proportion
pbp = 0.2 #personal best proportion
gbp = 0.1 # global best proportion
jp = 0.05 # velocity jump size

pso_iterations = 100
target_error = 1e-6
pso_particles = 40

#ANN (Particle Settings)
# Note : provide ANN layer based on input.
# For eg for data set 0,1,2,3 give like array([[2,1],[2,2],[1,2]])
# For data set 4 & 5 give like array([[2,2],[3,2],[1,3]])
ann_layer_config = array([[2,2],[2,2],[1,2]])
# Activation functions Null -> 0 , Sigmoid -> 1, Hyperbloic Tan -> 2, Cosine -> 3, Gaussian -> 4
activation_function = 4
# Data set file cubic -> 0 , linear -> 1, sine -> 2, tanh -> 3, complex -> 4, xor -> 5
data_set = 4

class Particle():
    def __init__(self):
        self.ann_layer_config = ann_layer_config
        self.position = array([0, 0])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        # Create the ANN for this particle
        self.create_ann()

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position = self.position + dot(jp,self.velocity)
        self.set_ann_weights()

    def create_ann(self):
        #TO-DO read the file and convert as array
        data = getDataFromFile(data_set)
        #print (data[0])
        self.ann = ArtificialNeuralNetwork(self.ann_layer_config, activation_function, data[0], data[1] )
        self.set_initial_position()
        self.set_ann_weights()

    def set_ann_weights(self):
        #pass self position values to
        self.ann.set_weights_from_position(self.position)

    # Create array of random values
    def set_initial_position(self):
        init_position = []
        init_velocity = []
        for layer_config in self.ann_layer_config:
            for i in range(dot(layer_config[0],layer_config[1])):
                init_position.append(self.create_random())
                init_velocity.append(0)
        self.position = asarray(init_position)
        self.velocity = asarray(init_velocity)
        self.pbest_position = self.position
    def create_random(self):
        return random.uniform(-100,100)

class PSO_Space():

    def __init__(self, target, target_error, pso_particles):
        self.ann_layer_config = ann_layer_config
        self.target = target
        self.target_error = target_error
        self.pso_particles = pso_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = array([0,0])
        self.set_initial_gbest_position()

    def set_initial_gbest_position(self):
        init_gbest_position = []
        for layer_config in self.ann_layer_config:
            for i in range(dot(layer_config[0],layer_config[1])):
                init_gbest_position.append(self.create_random())
        self.gbest_position = asarray(init_gbest_position)

    def create_random(self):
        return random.uniform(-100,100)

    def print_particles(self):
        for particle in self.particles:
             particle.__str__()

    def fitness_func(self, particle):
        return particle.ann.mse

    def set_personal_best(self):
        for particle in self.particles:
            fitness_value = self.fitness_func(particle)
            if (particle.pbest_value > fitness_value):
                particle.pbest_value = fitness_value
                particle.pbest_position = particle.position

    def set_global_best(self):
        for particle in self.particles:
            best_fitness_value = self.fitness_func(particle)
            if (self.gbest_value > best_fitness_value):
                self.gbest_value = best_fitness_value
                self.gbest_position = particle.position

    def forwardfeed_particles(self):
        for particle in self.particles:
            particle.ann.forward_inside_ann()

    def move_particles(self):
        for particle in self.particles:
            new_velocity = (vp * particle.velocity) + (random.uniform(0.0,pbp)) * (
                        particle.pbest_position - particle.position) + \
                           (random.uniform(0.0, gbp)) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


pso = PSO_Space(1, target_error, pso_particles)
pso.particles = [Particle() for _ in range(pso.pso_particles)]
#pso.print_particles()

iteration = 0
while (iteration < pso_iterations):
    pso.forwardfeed_particles()
    pso.set_personal_best()
    pso.set_global_best()

    if (abs(pso.gbest_value - pso.target) <= pso.target_error):
        break

    pso.move_particles()
    iteration += 1
    if(iteration == 1):
        print("Initial MSE", pso.particles[0].ann.mse, " and weights for first particle", pso.particles[0].position)

print("Final MSE", pso.particles[0].ann.mse, " and weights for first particle", pso.particles[0].position)

#Testing the output with the sample input
#pso.particles[0].ann.set_weights_from_position(pso.gbest_position)
#pso.particles[0].ann.set_input_values(array([0, 0, 1]))
#pso.particles[0].ann.forward_inside_ann()
#print (pso.particles[0].ann.ann_output)