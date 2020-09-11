# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:22:50 2020

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

def SongValArousalFromEEGTraining(training,starting_emotion='Sadness'):
    #subjects = list(range(1,33))
    #training = np.random.choice(32,22)
    #testing = [x for x in subjects if x not in training]
    
    songCategorizationDict_ = dict()
    emotion_count = dict() ##{Happy--> 2, Sad --> 3} /// This keeps count of the instances of the emotions
    emotion_count = {'Happy' : 0, 'Satisfaction' : 0, 'Elation' : 0, 'Pride' : 0,
                     'Contempt' : 0, 'Anger' : 0, 'Disgust' : 0, 'Envy' : 0,
                     'Guilt' : 0, 'Shame' : 0, 'Fear' : 0, 'Sadness' : 0,
                     'Relief' : 0, 'Hope' : 0, 'Interest' : 0, 'Surprise' : 0}
    
    ## Populating valence arousal of the music videos
    for i in training:  #nUser #4, 40, 32, 40, 8064 4 labels, 40 sample for each user, 32 such user, 40 electrode, 8064*40 features
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "deap_data/s"+str(name)+".dat" 
        f = open(fname,'rb')
        x = pickle.load(f, encoding='latin1')
        
        for j in range(40):
            if j in songCategorizationDict_.keys():
                songCategorizationDict_[j] = ((songCategorizationDict_[j][0]+x['labels'][j][0]),(songCategorizationDict_[j][1]+x['labels'][j][1]))
            else:
                songCategorizationDict_[j] = (x['labels'][j][0],x['labels'][j][1])
                
    
    for key in songCategorizationDict_.keys():
        songCategorizationDict_[key] = (songCategorizationDict_[key][0]/32,songCategorizationDict_[key][1]/32)
       
    
    print(songCategorizationDict_)
    return songCategorizationDict_
        