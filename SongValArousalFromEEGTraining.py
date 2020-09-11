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

def getSubjectToExperimentIDsDict():
    video_data = pd.read_excel('Metadata xls/participant_ratings.xls')
    vd = list(video_data['Participant_id'])
    ex = list(video_data['Experiment_id'])
    subjectToExp_ = dict()
    
    for i in range(len(vd)):
        if vd[i] in subjectToExp_.keys():
            subjectToExp_[vd[i]].append(ex[i])
        else:
            subjectToExp_[vd[i]] = [ex[i]]
        
    return subjectToExp_
    
    
    
    

def SongValArousalFromEEGTraining(training,starting_emotion='Sadness'):
    #subjects = list(range(1,33))
    #training = np.random.choice(32,22)
    #testing = [x for x in subjects if x not in training]
    subjectToExp_ = getSubjectToExperimentIDsDict()
    
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
                name = '%0*d' % (2,i)
            else:
                name = i
        fname = "deap_data/s"+str(name)+".dat" 
        f = open(fname,'rb')
        x = pickle.load(f, encoding='latin1')
        X = pd.concat([pd.DataFrame(subjectToExp_[i]),pd.DataFrame(x['labels'])],axis=1)
        X.columns = ['index','v','a','d','l']
        X = X.sort_values(by='index')
        
        
        for j in range(40):
            if j in songCategorizationDict_.keys():
                songCategorizationDict_[j] = ((songCategorizationDict_[j][0]+X.iloc[j,1]),(songCategorizationDict_[j][1]+X.iloc[j,2]))
            else:
                songCategorizationDict_[j] = (X.iloc[j,1],X.iloc[j,2])
                
    
    for key in songCategorizationDict_.keys():
        songCategorizationDict_[key] = (songCategorizationDict_[key][0]/len(training),songCategorizationDict_[key][1]/len(training))
       
    
    
    songdict = dict()
    for i in range(len(songCategorizationDict_)):
        valence =  songCategorizationDict_[i][0]-4.5
        arousal =  songCategorizationDict_[i][1]-4.5
        angle = math.degrees(math.atan2(arousal,valence))
        if 0 <= angle < 22.5:
            songdict[i] = "Satisfaction"
            emotion_count['Satisfaction'] += 1
        elif 22.5 <= angle < 45:
            songdict[i] = "Happy"
            emotion_count['Happy'] += 1
        elif 45 <= angle < 67.5:
            songdict[i] = "Elation"
            emotion_count['Elation'] += 1
        elif 67.5 <= angle < 90:
            songdict[i] = "Pride"
            emotion_count['Pride'] += 1
        elif 90 <= angle < 112.5:
            songdict[i] = "Anger"
            emotion_count['Anger'] += 1
        elif 112.5 <= angle < 135:
            songdict[i] = "Contempt"
            emotion_count['Contempt'] += 1
        elif 135 <= angle < 157.5:
            songdict[i] = "Disgust"
            emotion_count['Disgust'] += 1
        elif 157.5 <= angle < 180:
            songdict[i] = "Envy"
            emotion_count['Envy'] += 1
        elif -22.5 <= angle < 0:
            songdict[i] = "Relief"
            emotion_count['Relief'] += 1
        elif -45 <= angle < -22.5:
            songdict[i] = "Hope"
            emotion_count['Hope'] += 1
        elif -67.5 <= angle < -45:
            songdict[i] = "Interest"
            emotion_count['Interest'] += 1
        elif -90 <= angle < -67.5:
            songdict[i] = "Surprise"
            emotion_count['Surprise'] += 1
        elif -112.5 <= angle < -90:
            songdict[i] = "Sadness"
            emotion_count['Sadness'] += 1
        elif -135 <= angle < -112.5:
            songdict[i] = "Fear"
            emotion_count['Fear'] += 1
        elif -157.5 <= angle < -135:
            songdict[i] = "Shame"
            emotion_count['Shame'] += 1
        else:
            songdict[i] = "Guilt"
            emotion_count['Guilt'] += 1
    return songCategorizationDict_,songdict
        