
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:49:18 2020

@author: Ananya
"""

import numpy as np
import pandas as pd
import math
import pickle
from SongValArousalFromEEGTraining import SongValArousalFromEEGTraining,getSubjectToExperimentIDsDict
from populateRewardTableNew import populateRewardTableNew
from NextStateTablePopulationNew import populateNextStateTableNew
import matplotlib.pyplot as plt

def computeEmotion(angle):
    if 0 <= angle < 22.5:
        return "Satisfaction"
    elif 22.5 <= angle < 45:
        return "Happy"
    elif 45 <= angle < 67.5:
        return "Elation"
    elif 67.5 <= angle < 90:
        return "Pride"
    elif 90 <= angle < 112.5:
        return "Anger"
    elif 112.5 <= angle < 135:
        return "Contempt"
    elif 135 <= angle < 157.5:
        return "Disgust"
    elif 157.5 <= angle < 180:
        return "Envy"
    elif -22.5 <= angle < 0:
        return "Relief"
    elif -45 <= angle < -22.5:
        return "Hope"
    elif -67.5 <= angle < -45:
        return "Interest"
    elif -90 <= angle < -67.5:
        return "Surprise"
    elif -112.5 <= angle < -90:
        return "Sadness"
    elif -135 <= angle < -112.5:
        return "Fear"
    elif -157.5 <= angle < -135:
        return "Shame"
    else:
        return "Guilt"
    
def QTableTesting(training,testing,file,starting_emotion='Sadness'):
    subjectToExp_ = getSubjectToExperimentIDsDict()
    emotionIndices = {'Happy' : 2, 'Satisfaction' : 1, 'Elation' : 3, 'Pride' : 4,
                      'Contempt' : 6, 'Anger' : 5, 'Disgust' : 7, 'Envy' : 8,
                      'Guilt' : 9, 'Shame' : 10, 'Fear' : 11, 'Sadness' : 12,
                      'Relief' : 0, 'Hope' : 15, 'Interest' : 14, 'Surprise' : 13}
    
    inv_emotion_indices = {v: k for k, v in emotionIndices.items()}
    
    emotionWithPositions = {'Happy' : 45 - 22.5/2, 'Satisfaction' : 22.5/2, 'Elation' : 67.5 - 22.5/2, 'Pride' : 90 - 22.5/2,
                            'Contempt' : 135 - 22.5/2, 'Anger' : 112.5 - 22.5/2, 'Disgust' : 157.5 - 22.5/2, 'Envy' : 180 - 22.5/2,
                            'Guilt' : 180+22.5/2, 'Shame' : 202.5+22.5/2, 'Fear' : 225+22.5/2, 'Sadness' : 247.5+22.5/5,
                            'Relief' : 337.5+22.5/2, 'Hope' : 315+22.5/2, 'Interest' : 292.5+22.5/2, 'Surprise' : 270+22.5/2}
    
    next_state_table = populateNextStateTableNew(training)
    
    qTable = pd.read_csv(file,index_col=0,header=None)
    
    song_ids = qTable.columns.values
    modified_qTable = np.asarray(qTable)[1:] 
    
    
    
    tan_angle_of_sadness = math.tan(math.radians(emotionWithPositions['Sadness']))
    emotion_vector_sadness = np.asarray([1,tan_angle_of_sadness])
    
    for subject in testing:
        if(subject%1 == 0):
            if subject < 10:
                name = '%0*d' % (2,subject)
            else:
                name = subject
        fname = "deap_data/s"+str(name)+".dat" 
        f = open(fname,'rb')
        x = pickle.load(f, encoding='latin1')
        X = pd.concat([pd.DataFrame(subjectToExp_[subject]),pd.DataFrame(x['labels'])],axis=1)
        X.columns = ['index','v','a','d','l']
        X = X.sort_values(by='index')
        emotion = starting_emotion
        next_state_emotion_from_EEG_data_for_subject = []
        action_sequence_from_table = []
        for i in range(20):
            if emotion == 'Happy':
                print("Model Converged in %s iterations" %i)
                break
            emotion_id = emotionIndices[emotion]
            song_best_id = np.argmax(modified_qTable[emotion_id])
            song_best = song_ids[song_best_id]
            
            #next_state_id = next_state_table[emotion_id][song_best-1]
            #next_state = inv_emotion_indices[next_state_id]
           
            music_effect_valence = X.iloc[song_best-1,1]-4.5
            music_effect_arousal = X.iloc[song_best-1,2]-4.5
            
            #next_state_valence = X.iloc[song_best-1,1]-4.5
            #next_state_arousal = X.iloc[song_best-1,2]-4.5
            
            tan_angle_of_current_emotion = math.tan(math.radians(emotionWithPositions[emotion]))
            emotion_vector_current = np.asarray([1,tan_angle_of_current_emotion])
            
            next_state_valence = music_effect_valence + emotion_vector_current[0]
            next_state_arousal = music_effect_arousal + emotion_vector_current[1]
            
            next_state_angle = math.degrees(math.atan2(next_state_arousal,next_state_valence))
            next_state_emotion = computeEmotion(next_state_angle)
            next_state = next_state_emotion
            #next_state = next_state_table[emotion_id][song_best-1]
            
            action_sequence_from_table.append(song_best)
            next_state_emotion_from_EEG_data_for_subject.append(next_state)
            
            emotion = next_state
            
        print("Subject ",subject)
        print(action_sequence_from_table)
        print(next_state_emotion_from_EEG_data_for_subject)
     
        iteration_no = []
        actual_difference = []
        iterations = len(next_state_emotion_from_EEG_data_for_subject)
        for i in range(iterations):
            iteration_no.append(i)
            angle_between_actual_state_and_happy = np.abs(emotionWithPositions[next_state_emotion_from_EEG_data_for_subject[i]]-emotionWithPositions['Happy'])
            if angle_between_actual_state_and_happy > 180:
               angle_between_actual_state_and_happy = 360- angle_between_actual_state_and_happy
            actual_difference.append(angle_between_actual_state_and_happy)
            
        subject_name = "Subject" + " " + str(subject)
        print("Angular error")
        print(actual_difference)
        plt.plot(iteration_no,actual_difference,label=subject_name)
        plt.legend()
        

