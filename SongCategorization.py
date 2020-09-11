# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:27:11 2019

@author: Ananya
"""

import pandas as pd
import math

def SongCategorization():
    df = pd.read_excel('Metadata xls/video_list.xls')
    valences = df['AVG_Valence']
    arousals = df['AVG_Arousal']
    songCategorizationDict_ = dict()
    emotion_count = dict() ##{Happy--> 2, Sad --> 3} /// This keeps count of the instances of the emotions
    emotion_count = {'Happy' : 0, 'Satisfaction' : 0, 'Elation' : 0, 'Pride' : 0,
                     'Contempt' : 0, 'Anger' : 0, 'Disgust' : 0, 'Envy' : 0,
                     'Guilt' : 0, 'Shame' : 0, 'Fear' : 0, 'Sadness' : 0,
                     'Relief' : 0, 'Hope' : 0, 'Interest' : 0, 'Surprise' : 0}
    
    for i in range(len(df)):
        valence = valences.iloc[i]-4.5
        arousal = arousals.iloc[i]-4.5
        angle = math.degrees(math.atan2(arousal,valence))
        if 0 <= angle < 22.5:
            songCategorizationDict_[i] = "Satisfaction"
            emotion_count['Satisfaction'] += 1
        elif 22.5 <= angle < 45:
            songCategorizationDict_[i] = "Happy"
            emotion_count['Happy'] += 1
        elif 45 <= angle < 67.5:
            songCategorizationDict_[i] = "Elation"
            emotion_count['Elation'] += 1
        elif 67.5 <= angle < 90:
            songCategorizationDict_[i] = "Pride"
            emotion_count['Pride'] += 1
        elif 90 <= angle < 112.5:
            songCategorizationDict_[i] = "Anger"
            emotion_count['Anger'] += 1
        elif 112.5 <= angle < 135:
            songCategorizationDict_[i] = "Contempt"
            emotion_count['Contempt'] += 1
        elif 135 <= angle < 157.5:
            songCategorizationDict_[i] = "Disgust"
            emotion_count['Disgust'] += 1
        elif 157.5 <= angle < 180:
            songCategorizationDict_[i] = "Envy"
            emotion_count['Envy'] += 1
        elif -22.5 <= angle < 0:
            songCategorizationDict_[i] = "Relief"
            emotion_count['Relief'] += 1
        elif -45 <= angle < -22.5:
            songCategorizationDict_[i] = "Hope"
            emotion_count['Hope'] += 1
        elif -67.5 <= angle < -45:
            songCategorizationDict_[i] = "Interest"
            emotion_count['Interest'] += 1
        elif -90 <= angle < -67.5:
            songCategorizationDict_[i] = "Surprise"
            emotion_count['Surprise'] += 1
        elif -112.5 <= angle < -90:
            songCategorizationDict_[i] = "Sadness"
            emotion_count['Sadness'] += 1
        elif -135 <= angle < -112.5:
            songCategorizationDict_[i] = "Fear"
            emotion_count['Fear'] += 1
        elif -157.5 <= angle < -135:
            songCategorizationDict_[i] = "Shame"
            emotion_count['Shame'] += 1
        else:
            songCategorizationDict_[i] = "Guilt"
            emotion_count['Guilt'] += 1
        
                
    
    
    
    return songCategorizationDict_
    
        