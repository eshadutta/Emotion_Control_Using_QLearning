# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:20:54 2019

@author: Ananya
"""

import numpy as np
import pandas as pd
import pickle
import math
from matplotlib import pyplot as plt
from QTablePopulation import QTablePopulation
from SongCategorization import SongCategorization
from NextStateTablePopulation import populateNextStateTable
from RewardTablePopulation import populateRewardTable

def ResultValidationWithVsWithoutRewardShaping(starting_emotion='Sadness'):
    action_sequence_from_table = []
    next_state_emotion_sequence_from_table = []
    
    emotionIndices = {'Happy' : 2, 'Satisfaction' : 1, 'Elation' : 3, 'Pride' : 4,
                      'Contempt' : 6, 'Anger' : 5, 'Disgust' : 7, 'Envy' : 8,
                      'Guilt' : 9, 'Shame' : 10, 'Fear' : 11, 'Sadness' : 12,
                      'Relief' : 0, 'Hope' : 15, 'Interest' : 14, 'Surprise' : 13}
    
    inv_emotion_indices = {v: k for k, v in emotionIndices.items()}
    
    emotionWithPositions = {'Happy' : 45 - 22.5/2, 'Satisfaction' : 22.5/2, 'Elation' : 67.5 - 22.5/2, 'Pride' : 90 - 22.5/2,
                            'Contempt' : 135 - 22.5/2, 'Anger' : 112.5 - 22.5/2, 'Disgust' : 157.5 - 22.5/2, 'Envy' : 180 - 22.5/2,
                            'Guilt' : 180+22.5/2, 'Shame' : 202.5+22.5/2, 'Fear' : 225+22.5/2, 'Sadness' : 247.5+22.5/5,
                            'Relief' : 337.5+22.5/2, 'Hope' : 315+22.5/2, 'Interest' : 292.5+22.5/2, 'Surprise' : 270+22.5/2}
    
    next_state_table = populateNextStateTable()
    
    video_data = pd.read_excel('Metadata xls/video_list.xls')
    video_data = video_data.dropna(subset=['Experiment_id']) ##Data corresponding to which every subject is experimented    
    videos = video_data['Online_id'].tolist()
    
    qTable = pd.read_csv('qTable_without_reward_shaping.csv',index_col=0,header=None)
      
    ##Modified Qtable corresponding to the relevant actions
    modified_qTable = qTable[videos]
    song_ids = modified_qTable.columns.values
    modified_qTable = np.asarray(modified_qTable)[1:] 
    
    emotion = starting_emotion
    print("From Q-learning model without reward shaping:")
    
    for i in range(20):
        if emotion == 'Happy':
            print("Model Converged in %s iterations" %i)
            break
        emotion_id = emotionIndices[emotion]
        song_best_id = np.argmax(modified_qTable[emotion_id])
        song_best = song_ids[song_best_id]
        
        next_state = next_state_table[emotion_id][song_best-1]
        
        action_sequence_from_table.append(song_best)
        next_state_emotion_sequence_from_table.append(inv_emotion_indices[next_state])
        
        emotion = inv_emotion_indices[next_state]
      
    print(action_sequence_from_table)
    print(next_state_emotion_sequence_from_table)
    
    iterations = len(next_state_emotion_sequence_from_table)
    iteration_no = []
    actual_difference = [np.abs(emotionWithPositions['Sadness']-emotionWithPositions['Happy'])]
    for i in range(iterations):
        iteration_no.append(i)
        angle_between_actual_state_and_happy = np.abs(emotionWithPositions[next_state_emotion_sequence_from_table[i]]-emotionWithPositions['Happy'])
        if angle_between_actual_state_and_happy > 180:
           angle_between_actual_state_and_happy = 360- angle_between_actual_state_and_happy
        actual_difference.append(angle_between_actual_state_and_happy)
    iteration_no.append((i+1))
        
        
    fig = plt.figure(figsize=(9,7))        
    plt.plot(iteration_no,actual_difference,label="Without reward shaping",color='red')
    
    ### ---- Repeat the same process with Q-table populated with reward shaping-------###
    
    action_sequence_from_table = []
    next_state_emotion_sequence_from_table = []
    
    qTable = pd.read_csv('qTable_6.csv',index_col=0,header=None)
    
    ##Modified Qtable corresponding to the relevant actions
    modified_qTable = qTable[videos]
    song_ids = modified_qTable.columns.values
    modified_qTable = np.asarray(modified_qTable)[1:] 
    
    emotion = starting_emotion
    
    print("From Q-learning model with reward shaping:")
    for i in range(20):
        if emotion == 'Happy':
            print("Model Converged in %s iterations" %i)
            break
        emotion_id = emotionIndices[emotion]
        song_best_id = np.argmax(modified_qTable[emotion_id])
        song_best = song_ids[song_best_id]
        
        next_state = next_state_table[emotion_id][song_best-1]
        
        action_sequence_from_table.append(song_best)
        next_state_emotion_sequence_from_table.append(inv_emotion_indices[next_state])
        
        emotion = inv_emotion_indices[next_state]
      
    
    print(action_sequence_from_table)
    print(next_state_emotion_sequence_from_table)
    
    iterations = len(next_state_emotion_sequence_from_table)
    iteration_no = []
    actual_difference = [np.abs(emotionWithPositions['Sadness']-emotionWithPositions['Happy'])]
    for i in range(iterations):
        iteration_no.append(i)
        angle_between_actual_state_and_happy = np.abs(emotionWithPositions[next_state_emotion_sequence_from_table[i]]-emotionWithPositions['Happy'])
        if angle_between_actual_state_and_happy > 180:
           angle_between_actual_state_and_happy = 360- angle_between_actual_state_and_happy
        actual_difference.append(angle_between_actual_state_and_happy)
        
    iteration_no.append((i+1))
    #plt.plot(iteration_no,actual_difference,linestyle="dashdot",label='With reward shaping')
    plt.xlim((0,7))
    plt.xlabel("Number of iterations",fontsize=18)
    plt.ylabel("Angular Error in Emotion Transition",fontsize=18)
    plt.title("Angular Error vs Iterations",fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig('ErrorWithVsWithoutRewardShaping.pdf')
    
    
    
        
    
        
        
        
    
    
    
    
    
    
    
    
    