# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 02:15:43 2019

@author: Ananya
"""

from SongCategorization import SongCategorization
import numpy as np

def populateRewardTable(no_of_emotions=16,desired_emotion='Happy'):
    song_categorization = SongCategorization()
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
    
    for emotion in emotionIndices.keys():
        for song_id in song_categorization.keys():
            angle_between_action_and_dest = np.abs(emotionWithPositions[song_categorization[song_id]] - emotionWithPositions[desired_emotion])
            angle_between_action_and_source = np.abs(emotionWithPositions[song_categorization[song_id]] - emotionWithPositions[emotion])
            difference_angle = angle_between_action_and_dest + angle_between_action_and_source
            if difference_angle <= 180:
                reward_table[emotionIndices[emotion]][song_id] = 1
    
    return reward_table
    