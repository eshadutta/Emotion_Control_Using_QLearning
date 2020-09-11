# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:30:16 2019

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

def ResultValidation(starting_emotion='Sadness'):
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
    
    qTable = pd.read_csv('qTable_6.csv',index_col=0,header=None)
    
    video_data = pd.read_excel('Metadata xls/video_list.xls')
    video_data = video_data.dropna(subset=['Experiment_id']) ##Data corresponding to which every subject is experimented
    
    videos = video_data['Online_id'].tolist()
    
    ##Modified Qtable corresponding to the relevant actions
    modified_qTable = qTable[videos]
    song_ids = modified_qTable.columns.values
    modified_qTable = np.asarray(modified_qTable)[1:] 
    
    emotion = starting_emotion
    
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
      
    print("From Q-learning model:")
    print(action_sequence_from_table)
    print(next_state_emotion_sequence_from_table)
    
    online_id_to_exp = pd.Series(video_data.Experiment_id.values,index=video_data.Online_id).to_dict()
    next_state_emotion_from_EEG_data = []
    
    tan_angle_of_sadness = math.tan(math.radians(emotionWithPositions['Sadness']))
    emotion_vector_sadness = np.asarray([1,tan_angle_of_sadness])
    
     
    subjects = []
    for i in range(32):  #nUser #4, 40, 32, 40, 8064 4 labels, 40 sample for each user, 32 such user, 40 electrode, 8064*40 features
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "deap_data/s"+str(name)+".dat"     
        subjects.append(fname)
     
    
    ## Experimenting on some randomly selected subjects    
    #subjects = ["deap_data/s29.dat","deap_data/s20.dat","deap_data/s13.dat","deap_data/s19.dat","deap_data/s02.dat"]
    #subjects = ["deap_data/s24.dat"]
    emotion_output = pd.DataFrame()
    for i in range(len(subjects)):
        next_state_emotion_from_EEG_data_for_subject = []
        for action in action_sequence_from_table:
            f = open(subjects[i], 'rb')                 
            x = pickle.load(f, encoding='latin1')
            exp_id = online_id_to_exp[action]
            valence = x['labels'][int(exp_id)-1][0]-4.5
            arousal = x['labels'][int(exp_id)-1][1]-4.5
            
            next_state = np.asarray([valence,arousal])
            
            next_state_with_correction = emotion_vector_sadness+next_state
            angle = math.degrees(math.atan2(next_state_with_correction[1],next_state_with_correction[0]))
            
            if 0 <= angle < 22.5:
                next_state_emotion_from_EEG_data_for_subject.append('Satisfaction')
            elif 22.5 <= angle < 45:
                next_state_emotion_from_EEG_data_for_subject.append('Happy')
            elif 45 <= angle < 67.5:
                next_state_emotion_from_EEG_data_for_subject.append('Elation')
            elif 67.5 <= angle < 90:
                next_state_emotion_from_EEG_data_for_subject.append('Pride')
            elif 90 <= angle < 112.5:
                next_state_emotion_from_EEG_data_for_subject.append('Anger')
            elif 112.5 <= angle < 135:
                next_state_emotion_from_EEG_data_for_subject.append('Contempt')
            elif 135 <= angle < 157.5:
                next_state_emotion_from_EEG_data_for_subject.append('Disgust')
            elif 157.5 <= angle < 180:
                next_state_emotion_from_EEG_data_for_subject.append('Envy')
            elif -22.5 <= angle < 0:
                next_state_emotion_from_EEG_data_for_subject.append('Relief')
            elif -45 <= angle < -22.5:
                next_state_emotion_from_EEG_data_for_subject.append('Hope')
            elif -67.5 <= angle < -45:
                next_state_emotion_from_EEG_data_for_subject.append('Interest')
            elif -90 <= angle < -67.5:
                next_state_emotion_from_EEG_data_for_subject.append('Surprise')
            elif -112.5 <= angle < -90:
                next_state_emotion_from_EEG_data_for_subject.append('Sadness')
            elif -135 <= angle < -112.5:
                next_state_emotion_from_EEG_data_for_subject.append('Fear')
            elif -157.5 <= angle < -135:
                next_state_emotion_from_EEG_data_for_subject.append('Shame')
            else:
                next_state_emotion_from_EEG_data_for_subject.append('Guilt')
            
        
        next_state_emotion_from_EEG_data.append(next_state_emotion_from_EEG_data_for_subject)
        print("From EEG data for %s" %subjects[i])
        print(next_state_emotion_from_EEG_data_for_subject)
        emotion_output = pd.concat([emotion_output,pd.DataFrame(next_state_emotion_from_EEG_data_for_subject)],axis=1)
        
        
    iterations = len(next_state_emotion_from_EEG_data[0])
    
    iteration_no = []
    actual_difference = []
    predicted_difference = []
    actual_regret = []
    predicted_regret = []
    
    for i in range(len(next_state_emotion_from_EEG_data)):
        angle_iteration = [np.abs(emotionWithPositions['Sadness']-emotionWithPositions['Happy'])]
        for j in range(iterations):
            angle_between_predicted_state_and_happy = np.abs(emotionWithPositions[next_state_emotion_from_EEG_data[i][j]]-emotionWithPositions['Happy'])
            if angle_between_predicted_state_and_happy > 180:
               angle_between_predicted_state_and_happy = 360- angle_between_predicted_state_and_happy
            angle_iteration.append(angle_between_predicted_state_and_happy)
        predicted_difference.append(angle_iteration)
            
    actual_regret = [0]
    for i in range(iterations):
        iteration_no.append(i)
        angle_between_actual_state_and_happy = np.abs(emotionWithPositions[next_state_emotion_sequence_from_table[i]]-emotionWithPositions['Happy'])
        if angle_between_actual_state_and_happy > 180:
           angle_between_actual_state_and_happy = 360- angle_between_actual_state_and_happy
        actual_difference.append(angle_between_actual_state_and_happy)
        regret = actual_regret[len(actual_regret)-1] + angle_between_actual_state_and_happy
        actual_regret.append(regret)
        
        
        
    plt.plot(iteration_no,actual_difference,label="Q-table prediction")
    iteration_no.append((i+1))
    for i in range(len(predicted_difference)):
        name = "Subject" + " " + str(i+1)
        plt.plot(iteration_no,predicted_difference[i],label=name,linestyle='dashdot')
        
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Angular Error in Emotion Transition")
    plt.title("Angular Error vs Iterations")
    plt.savefig('file24.pdf')
    return emotion_output,actual_difference,predicted_difference
    
    
    
        
    
        
        
        
    
    
    
    
    
    
    
    
    