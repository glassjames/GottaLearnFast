import retro
import numpy
import cv2
import neat
import pickle

import os
import sys

# initialize environment with game and level
env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')
# env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

# current environment view
view = []

#begin evaluating genomes
def eval_genomes(genomes, config):
    # for each genome in the pool
    for genome_id, genome in genomes:
        # create a new environment and set related variables
        observation = env.reset()
        action = env.action_space.sample()
        environment_x, environment_y, environment_color = env.observation_space.shape

        # scale down the environment lengths
        environment_x = int(environment_x / 8)
        environment_y = int(environment_y / 8)

        # create recurrent network with genome and config
        neural_network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # used for keeping track of fitness
        max_fitness = 0
        fitness = 0

        # used to tell if Sonic is getting stuck
        decay = 0

        done = False

        max_x = 0

        # cv2.namedWindow("Sonic.py", cv2.WINDOW_NORMAL)

        # while the current Sonic's done conditions aren't met
        while not done:
            # render the environment
            env.render()
            # resize, current to black and white, and reshape the observation
            observation = cv2.resize(observation, (environment_x, environment_y))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = numpy.reshape(observation, (environment_x, environment_y))

            # add each pixel into the view array
            for i in observation:
                for j in i:
                    view.append(j)

            # view cv image
            # img = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(scaledimg, (environment_y, environment_x))
            # cv2.imshow("Sonic.py", img)
            # cv2.waitKey(1)

            # feed the neural network the view and retrieve action
            output = neural_network.activate(view)

            # perform action during step and retrieve info
            observation, reward, done, info = env.step(output)

            view.clear()

            fitness += reward

            x = info['x']

            if max_x < x:
                fitness += 1.0
                max_x = x

            if max_fitness < fitness:
                max_fitness = fitness
                if decay > 0:
                    decay -= 1
            else:
                decay += 1

            if decay == 200:
                done = True
                print("Genome: ", genome_id, "Fitness: ", fitness)

            if x >= info['screen_x_end'] and x != 0:
                print("X: ", x, "End: ", info['screen_x_end'])
                fitness += 100000
                done = True
                print("Genome: ", genome_id, "Fitness: ", fitness)

            genome.fitness = fitness

def pb():
    print("Starting playback of personal best...")
    with open('sonic-pb.pkl', 'rb') as input:
        genome = pickle.load(input)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config')
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    observation = env.reset()
    action = env.action_space.sample()
    environment_x, environment_y, environment_color = env.observation_space.shape

    # scale down the environment lengths
    environment_x = int(environment_x / 8)
    environment_y = int(environment_y / 8)

    # create recurrent network with genome and config
    neural_network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

    done = False

    # while the current Sonic's done conditions aren't met
    while not done:
        # render the environment
        env.render()
        # resize, current to black and white, and reshape the observation
        observation = cv2.resize(observation, (environment_x, environment_y))
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = numpy.reshape(observation, (environment_x, environment_y))

        # add each pixel into the view array
        for i in observation:
            for j in i:
                view.append(j)

        # feed the neural network the view and retrieve action
        output = neural_network.activate(view)

        # perform action during step and retrieve info
        observation, reward, done, info = env.step(output)

        view.clear()


def train():
    prefix = "sonic-network-"
    save = neat.Checkpointer(1, 300, prefix);
    file = [filename for filename in os.listdir('.') if filename.startswith(prefix)]
    print(file)

    if file:
        print("Opening network restore point: " + file[-1])
        pop = save.restore_checkpoint(file[-1])

    else :
        print("Starting new network...")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config')
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(save)

    # parallel = neat.ParallelEvaluator(6, eval_genomes)
    # winner = pop.run(parallel.evaluate)
    winner = pop.run(eval_genomes)

    #neat.save_checkpoint(config, pop, winner, p.generation)
    with open('sonic-pb.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

# main
def main(arg):
    if arg == "pb":
        print("Running personal best...")
        pb()
    elif arg == "train":
        print("Beginning training...")
        train()
    else:
        print("Invalid command line argument.")

if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except:
        print("\nExiting...")