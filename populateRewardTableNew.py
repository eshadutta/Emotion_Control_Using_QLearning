# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:02:09 2020

@author: Ananya
"""

from SongValArousalFromEEGTraining import SongValArousalFromEEGTraining
import numpy as np
import math

def populateRewardTableNew(training, next_state_table, no_of_emotions=16,desired_emotion='Happy'):
    song_categorization,songdict = SongValArousalFromEEGTraining(training)
    no_of_songs = len(song_categorization)
    
    reward_table = np.zeros((no_of_emotions,no_of_songs))
    
    emotionWithPositions = {'Happy' : 45 - 22.5/2, 'Satisfaction' : 22.5/2, 'Elation' : 67.5 - 22.5/2, 'Pride' : 90 - 22.5/2,
                            'Contempt' : 135 - 22.5/2, 'Anger' : 112.5 - 22.5/2, 'Disgust' : 157.5 - 22.5/2, 'Envy' : 180 - 22.5/2,
                            'Guilt' : 180+22.5/2, 'Shame' : 202.5+22.5/2, 'Fear' : 225+22.5/2, 'Sadness' : 247.5+22.5/5,
                            'Relief' : 337.5+22.5/2, 'Hope' : 315+22.5/2, 'Interest' : 292.5+22.5/2, 'Surprise' : 270+22.5/2}
    
    emotionIndices = {'Happy' : 2, 'Satisfaction' : 1, 'Elation' : 3, 'Pride' : 4,
                      'Contempt' : 6, 'Anger' : 5, 'Disgust' : 7, 'Envy' : 8,
                      'Guilt' : 9, 'Shame' : 10, 'Fear' : 11, 'Sadness' : 12,
                      'Relief' : 0, 'Hope' : 15, 'Interest' : 14, 'Surprise' : 13}
    
    inv_emotion_indices = {v: k for k, v in emotionIndices.items()}
    best_emotion = ['Happy','Elation','Satisfaction']
    
    for emotion in emotionIndices.keys():
        for song_id in songdict.keys():
            
            '''
            angle_between_current_and_dest = np.abs(emotionWithPositions[emotion] - emotionWithPositions[desired_emotion])
            if angle_between_current_and_dest > 180:
                angle_between_current_and_dest = 360 - angle_between_current_and_dest
            next_state = next_state_table[emotionIndices[emotion]][song_id]
            angle_between_next_state_and_dest = np.abs(emotionWithPositions[inv_emotion_indices[next_state]] - emotionWithPositions[desired_emotion])
            if angle_between_next_state_and_dest > 180:
                angle_between_next_state_and_dest = 360 - angle_between_next_state_and_dest
            difference_angle = angle_between_next_state_and_dest + angle_between_current_and_dest
            if difference_angle <= 180:
                reward_table[emotionIndices[emotion]][song_id] = 1
            '''
            next_state = next_state_table[emotionIndices[emotion]][song_id]
            next_state_emotion = inv_emotion_indices[next_state]
            if next_state_emotion in best_emotion:
                reward_table[emotionIndices[emotion]][song_id] = 1
                
            
            
    
    return reward_table
    