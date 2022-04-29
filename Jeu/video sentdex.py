import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from game import Game
from main import get_data, launch_game, move

LR = 1e-3
goal_steps = 2
score_requirement = 2
initial_games = 10




def initial_population():
    training_data= []
    scores = []
    accepted_scores= []
    launch_game()
    for _ in range(initial_games):
        #lancer partie?
        score = 0 
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            #renvoyer l'action choisie aléatoirement au jeu et récupérer l'état suivant (changer env.step(action) par game.step(action)) 
            #observation ça doit etre les données 
            observation, score, done = env.(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation

            #On break si la partie est finie ou si le bot arrive à 500pas 
            if done:
                break 
        #Si le score est assez haut on let dans les training data 
        if score >= score_requirement:
            accepted_scores.append(score)
            #one-hot
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])
        #clean l'environnement? réinitialiser jeu?
        #env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)

    #inutile non?
    np.save('saved.npy', training_data_save)

    print('average accepted score:', mean(accepted_scores))
    print('median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

initial_population()


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    #jsp a quoi sert le id_model...
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            print(np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1)) [0]))
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1)) [0])
        choices.append(action)

        #même truc qu'avant
        new_observation, reward, done = env.step(action)
        prev_obs = new_observation 
        game_memory.append([new_observation, action])
        if done:
            break 
    scores.append(score)
print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))

training_data = initial_population()

model = train_model(training_data)


def rungame():
    game = Game()
    running = True



    