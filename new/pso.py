from numpy import exp, array, dot, tanh, cos, asarray
from ann import ArtificialNeuralNetwork, NeuralLayer
from utils import *
import random
from sys import exit

W = 0.2
c1 = 0.5
c2 = 0.2

n_iterations = 50
target_error = 1e-6
n_particles = 30

class Particle():
    def __init__(self):
        self.ann_layer_config = array([[4, 3], [1, 4]])
        self.position = array([0, 0])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.create_ann()

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position = self.position + self.velocity
        self.set_ann_weights()

    def create_ann(self):
        # Activation functions Null -> 0 , Sigmoid -> 1, Hyperbloic Tan -> 2, Cosine -> 3, Gaussian -> 4
        activation_function = 1
        #TO-DO read the file and convert as array
        data = getDataFromFile('xyz')
        self.ann = ArtificialNeuralNetwork(self.ann_layer_config, activation_function, data[0], data[1] )
        self.set_initial_position()
        self.set_ann_weights()

    def set_ann_weights(self):
        #pass self position values to
        self.ann.set_weights_from_position(self.position)

    # Create array of random values
    def set_initial_position(self):
        init_position = []
        for layer_config in self.ann_layer_config:
            for i in range(dot(layer_config[0],layer_config[1])):
                init_position.append(self.create_random())
        self.position = asarray(init_position)
        self.pbest_position = self.position
    def create_random(self):
        return random.random()

class PSO_Space():

    def __init__(self, target, target_error, n_particles):
        self.ann_layer_config = array([[4, 3], [1, 4]])
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
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
        return random.random()

    def print_particles(self):
        for particle in self.particles:
             particle.__str__()

    def fitness(self, particle):
        return particle.ann.mse

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if (particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle)
            if (self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def feed_ann_particles(self):
        for particle in self.particles:
            particle.ann.forward_inside_ann()

    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = (W * particle.velocity) + (c1 * random.random()) * (
                        particle.pbest_position - particle.position) + \
                           (random.random() * c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


pso = PSO_Space(1, target_error, n_particles)
particles_vector = [Particle() for _ in range(pso.n_particles)]
pso.particles = particles_vector
pso.print_particles()

iteration = 0
while (iteration < n_iterations):
    pso.feed_ann_particles()
    pso.set_pbest()
    pso.set_gbest()

    if (abs(pso.gbest_value - pso.target) <= pso.target_error):
        break

    pso.move_particles()
    iteration += 1

print("The best solution is: ", pso.gbest_position, " in n_iterations: ", iteration)

pso.particles[0].ann.set_weights_from_position(pso.gbest_position)
pso.particles[0].ann.set_input_values(array([1, 1, 0]))
pso.particles[0].ann.forward_inside_ann()
print (pso.particles[0].ann.ann_output)