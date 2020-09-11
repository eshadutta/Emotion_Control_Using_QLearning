# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:38:24 2019

@author: Ananya
"""

import numpy as np
import pandas as pd
from SongCategorization import SongCategorization
from RewardTablePopulation import populateRewardTable
from NextStateTablePopulation import populateNextStateTable

def QTablePopulation(RewardShaping = True, no_of_emotions=16,starting_emotion='Envy'):
    song_categorization = SongCategorization()
    no_of_songs = len(song_categorization)
    
    qTable = np.zeros((no_of_emotions,no_of_songs),dtype=float)
    
    emotionIndices = {'Happy' : 2, 'Satisfaction' : 1, 'Elation' : 3, 'Pride' : 4,
                      'Contempt' : 6, 'Anger' : 5, 'Disgust' : 7, 'Envy' : 8,
                      'Guilt' : 9, 'Shame' : 10, 'Fear' : 11, 'Sadness' : 12,
                      'Relief' : 0, 'Hope' : 15, 'Interest' : 14, 'Surprise' : 13}
    emotionWithPositions = {'Happy' : 45 - 22.5/2, 'Satisfaction' : 22.5/2, 'Elation' : 67.5 - 22.5/2, 'Pride' : 90 - 22.5/2,
                            'Contempt' : 135 - 22.5/2, 'Anger' : 112.5 - 22.5/2, 'Disgust' : 157.5 - 22.5/2, 'Envy' : 180 - 22.5/2,
                            'Guilt' : 180+22.5/2, 'Shame' : 202.5+22.5/2, 'Fear' : 225+22.5/2, 'Sadness' : 247.5+22.5/5,
                            'Relief' : 337.5+22.5/2, 'Hope' : 315+22.5/2, 'Interest' : 292.5+22.5/2, 'Surprise' : 270+22.5/2}
    
    inv_emotion_indices = {v: k for k, v in emotionIndices.items()}
    ##Hyperparamaters
    #alphas = [0.1,0.3,0.5]
    #gammas = [0.4,0.6,0.8]
    #epsilons = [0,0.05,0.1]
    
    alphas = [0.1]
    gammas = [0.6]
    epsilons = [0.1]
    
    ##Reward table and Next state table
    reward_table = populateRewardTable()
    next_state_table = populateNextStateTable()

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for i in range(1, 10001):
                    state = starting_emotion
                
                    epochs, penalties, reward, = 0, 0, 0
                    
                    for epochs in range(1000):
                        if np.random.uniform(0,1) < epsilon:
                            action = np.random.choice(no_of_songs) # Explore action space
                        else:
                            action = np.argmax(qTable[emotionIndices[state]]) # Exploit learned values
                 
                        next_state_index = next_state_table[emotionIndices[state]][action]
                        next_state = inv_emotion_indices[next_state_index]
                        reward     = reward_table[emotionIndices[state]][action]
                        
                        old_value = qTable[emotionIndices[state]][action]
                        next_max = np.max(qTable[next_state_index])
                        
                        angle_between_new_state_and_happy = np.abs(emotionWithPositions[next_state]-emotionWithPositions['Happy'])
                        if angle_between_new_state_and_happy > 180:
                            angle_between_new_state_and_happy = 360-angle_between_new_state_and_happy
                        angle_between_state_and_happy = np.abs(emotionWithPositions[state]-emotionWithPositions['Happy'])
                        if angle_between_state_and_happy > 180:
                            angle_between_state_and_happy = 360-angle_between_state_and_happy
                       
                        reward_shaping = 0
                        if (RewardShaping):
                            reward_shaping = 10
                            if angle_between_new_state_and_happy > angle_between_state_and_happy:
                                reward_shaping = -100
                            elif angle_between_new_state_and_happy == angle_between_state_and_happy:
                                reward_shaping = -200
                        
                        new_value = (1 - alpha) * old_value + alpha * (reward + reward_shaping + gamma * next_max)
                        qTable[emotionIndices[state]][action] = new_value
                
                        if reward == -10:
                            penalties += 1
                
                        state = next_state
                        
                    if i % 1000 == 0:
                        print(f"Episode: {i}")
                
                print("Training finished.\n")
                Y = pd.DataFrame(qTable)
                Y.index = Y.index.map(str)
                for i in range(16):
                    Y.index.values[i] = inv_emotion_indices[i]
                fname = "qTable_envy.csv"
                Y.to_csv(fname)
    
    