import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from game import Game
from main import get_data, launch_game, drive_move, one_game_run_game_memory
import pygame

# générer fenêtre jeu
pygame.display.set_caption("comet fall Game")
screen = pygame.display.set_mode((540, 360))

# importer arriere plan du jeu
background = pygame.image.load('assets/bg.jpg')
background = pygame.transform.scale(background, (550,360))





LR = 1e-3
score_requirement = 500
initial_games = 100




def initial_population():
    training_data= []
    scores = []
    accepted_scores= []
    for _ in range(initial_games):
        #lancer partie?
        score = 0 
        game_memory = one_game_run_game_memory()
        score = game_memory[-1][-1]

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
    print(scores)
    #print(training_data)
    return(training_data)

#initial_population()

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
a = 0
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    a += 1
    # lancement jeu
    game = Game()
    running = True
    game.start()

    while running:

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)




        # appliquer arriere plan du jeu
        screen.blit(background, (0, 0))

        # déclencher les instructions de la partie
        new_observation = get_data(game)
        score = game.score
        print(new_observation , drive_move(action), score) 
        game.update(screen)
        
         # mettre a jour l'ecran
        pygame.display.flip()

        prev_obs = new_observation
        game_memory.append([new_observation, action])

        if game.player.health < 1 :
                 # jeux en mode lancé
                running = False
 
    scores.append(score)

print('Average Score:',sum(scores)/len(scores), 'a=', a)
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
