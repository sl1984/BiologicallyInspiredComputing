import numpy as np


def assess_fitness(position):
    return position[0] ** 2 + position[1] ** 2 + np.random.rand() * 2


def pso(swarmsize, iterations):
    inertia_weight = 0.5  # proportion of velocity to be retained - Inertia Weight
    cognitive_constant = 0.5  # proportion of personal best to be retained - cognitive constant
    social_constant = 0.9  # proportion of the informants’ best to be retained - social constant

    # An array of random location for each particle
    particles_location = np.array([np.array([np.random.rand() * 50, np.random.rand() * 50]) for _ in range(swarmsize)])

    print("Random Locations for each particle in swarm size ", swarmsize, "is ", particles_location)

    # Inititalise
    particle_fittest_location = particles_location  # fittest location assigned to default location
    particle_fitness_value = np.array([float('inf') for _ in range(swarmsize)])  # for swarm size 2, [inf inf]
    global_fitness_value = float('inf')  # inf
    global_fittest_location = np.array([float('inf'), float('inf')])  # [inf inf]

    velocity = ([np.array([0, 0]) for _ in range(swarmsize)])  # for a swarm size 2, [array([0,0]), array([0,0]))]


    for iteration in range (iterations):
        for i in range(swarmsize):
            fitness_value = assess_fitness(particles_location[i])

            if (particle_fitness_value[i] > fitness_value):
                particle_fitness_value[i] = fitness_value
                particle_fittest_location[i] = particles_location[i]
                print("particle_fittest_location[i]= ", particle_fittest_location[i], " in iteration:", iteration, "for particle:",i)

            if (global_fitness_value > fitness_value):
                global_fitness_value = fitness_value
                global_fittest_location = particles_location[i]
                print("global_fittest_location= ", global_fittest_location, " in iteration:", iteration)

        for i in range(swarmsize):
            # Update particle(i) velocity
            calculated_velocity = (inertia_weight * velocity[i]) + (cognitive_constant * np.random.rand()) * (
                        particle_fittest_location[i] - particles_location[i]) + (social_constant * np.random.rand()) * (
                                       global_fittest_location - particles_location[i])


            # Update position of particle(i) to the new position based on calculated velocity
            new_position = calculated_velocity + particles_location[i]
            particles_location[i] = new_position


    print("The best position achieved is ", global_fittest_location)

pso(10,8)