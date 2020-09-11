# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:55:10 2019

@author: Ananya
"""

import numpy as np
import pandas as pd
import math
from SongCategorization import SongCategorization

def populateNextStateTable(no_of_emotions=16):
    
    emotionWithPositions = {'Happy' : 45 - 22.5/2, 'Satisfaction' : 22.5/2, 'Elation' : 67.5 - 22.5/2, 'Pride' : 90 - 22.5/2,
                            'Contempt' : 135 - 22.5/2, 'Anger' : 112.5 - 22.5/2, 'Disgust' : 157.5 - 22.5/2, 'Envy' : 180 - 22.5/2,
                            'Guilt' : 180+22.5/2, 'Shame' : 202.5+22.5/2, 'Fear' : 225+22.5/2, 'Sadness' : 247.5+22.5/5,
                            'Relief' : 337.5+22.5/2, 'Hope' : 315+22.5/2, 'Interest' : 292.5+22.5/2, 'Surprise' : 270+22.5/2}
    
    emotionIndices = {'Happy' : 2, 'Satisfaction' : 1, 'Elation' : 3, 'Pride' : 4,
                      'Contempt' : 6, 'Anger' : 5, 'Disgust' : 7, 'Envy' : 8,
                      'Guilt' : 9, 'Shame' : 10, 'Fear' : 11, 'Sadness' : 12,
                      'Relief' : 0, 'Hope' : 15, 'Interest' : 14, 'Surprise' : 13}
    
    song_categorization = SongCategorization()
    no_of_songs = len(song_categorization)
    
    df = pd.read_excel('Metadata xls/video_list.xls')
    valences = df['AVG_Valence']
    arousals = df['AVG_Arousal']
    
    next_state_table = np.ones((no_of_emotions,no_of_songs),dtype=int)*-1
    
    
    for emotion in emotionIndices.keys():
        
        tan_angle_of_emotion = math.tan(math.radians(emotionWithPositions[emotion]))
        emotion_vector = np.asarray([1,tan_angle_of_emotion])
        
        for song_id in song_categorization.keys():
            valence = 4.5 - valences.iloc[song_id]
            arousal = 4.5 - arousals.iloc[song_id]
            song = np.asarray([valence,arousal])
            
            next_state = emotion_vector+song
            angle = math.degrees(math.atan2(next_state[1],next_state[0]))
            
            if 0 <= angle < 22.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Satisfaction']
            elif 22.5 <= angle < 45:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Happy']
            elif 45 <= angle < 67.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Elation']
            elif 67.5 <= angle < 90:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Pride']
            elif 90 <= angle < 112.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Anger']
            elif 112.5 <= angle < 135:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Contempt']
            elif 135 <= angle < 157.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Disgust']
            elif 157.5 <= angle < 180:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Envy']
            elif -22.5 <= angle < 0:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Relief']
            elif -45 <= angle < -22.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Hope']
            elif -67.5 <= angle < -45:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Interest']
            elif -90 <= angle < -67.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Surprise']
            elif -112.5 <= angle < -90:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Sadness']
            elif -135 <= angle < -112.5:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Fear']
            elif -157.5 <= angle < -135:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Shame']
            else:
                next_state_table[emotionIndices[emotion]][song_id] = emotionIndices['Guilt']
                
        
    return next_state_table

    
    
    
    