from numpy import exp, array, dot, tanh, cos, asarray
from ann import ArtificialNeuralNetwork, NeuralLayer
from utils import *
import random

W = 0.5
c1 = 0.8
c2 = 0.9

n_iterations = 50
target_error = 1e-6
n_particles = 30

class Particle():
    def __init__(self):
        self.ann_layer_config = array([[4, 3], [1, 4]])
        self.position = array([0, 0])
        print (self.position)
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = array([0, 0])
        self.create_ann()

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position = self.position + self.velocity

    def create_ann(self):
        # Activation functions Null -> 0 , Sigmoid -> 1, Hyperbloic Tan -> 2, Cosine -> 3, Gaussian -> 4
        activation_function = 1
        #read the file and convert as array
        data = getDataFromFile('xyz')
        self.ann = ArtificialNeuralNetwork(self.ann_layer_config, activation_function, data[0], data[1] )
        self.set_initial_position()
        print (len(self.position))
        self.set_ann_weights()

    def set_ann_weights(self):
        #pass self position values to
        self.ann.set_weights_from_position(self.position)

    def set_initial_position(self):
        init_position = []
        for layer_config in self.ann_layer_config:
            for i in range(dot(layer_config[0],layer_config[1])):
                init_position.append(self.create_random())
        self.position = asarray(init_position)
    def create_random(self):
        return (-1) ** (bool(random.getrandbits(1))) * random.random() * 50

class Space():

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
        return random.random() * 50

    def print_particles(self):
        for particle in self.particles:
             particle.__str__()

    def fitness(self, particle):
        #to-do mean square
        return particle.position[0] ** 2 + particle.position[1] ** 2 + 1

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

    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = (W * particle.velocity) + (c1 * random.random()) * (
                        particle.pbest_position - particle.position) + \
                           (random.random() * c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


search_space = Space(1, target_error, n_particles)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.print_particles()

iteration = 0
while (iteration < n_iterations):
    search_space.set_pbest()
    search_space.set_gbest()

    if (abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break

    search_space.move_particles()
    iteration += 1

print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)
