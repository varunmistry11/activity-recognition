# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:23:23 2017

@author: Dishang
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('../pose_bodyparts.json') as json_file:
        bodypoints_index=json.load(json_file)

cols=[]
for i in range(0,19):
    cols.append(bodypoints_index[str(i)])
    
def load_frames(path_to_json):
    
# =============================================================================
#     Returns raw list as follows
#     All : frame -> person(s) ->18 points
#     Each element will be a list
#     
# =============================================================================
    
    
    json_files = [pos_json for pos_json in os.listdir(path_to_json) \
                     if pos_json.endswith('.json')]
    
    frames=[]
        
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            people=[]
            for j in range(0,len(json_text['people'])):
                person=json_text['people'][j]['pose_keypoints']
                xyc=[ [person[i*3],person[i*3+1],person[i*3+2]] \
                     for i in range(0,18)]
                people.append(xyc)
            frames.append(people)
#            try:
#                row=json_text['people'][0]['pose_keypoints']
#                xyc=[ [row[i*3],row[i*3+1],row[i*3+2]] for i in range(0,18)]
#                frames.append(xyc)
#            except IndexError:
#                continue
#            try:
#                row=json_text['people'][1]['pose_keypoints']
#                xyc=[ [row[i*3],row[i*3+1],row[i*3+2]] for i in range(0,18)]
#                frames2.append(xyc)
#            except IndexError:
#                continue
    #        for c in range(0,19):
    #            people.loc[index]=pd.Series(xyc.tolist(),cols[0:len(xyc)])
    
#    frames=np.array(frames)
#    frames2=np.array(frames2)
    return frames
    
def load_frames_as_DF(path_to_json):
    
# =============================================================================
#     Returns a list with the first index denoting frame number 
#     The second index will contain a list of dataframes of the persons
#     Each Dataframe has 18 rows and 3 columns :x, y & c
#     Eg. frames[32][0] will give dataframe of 0th person in 32nd frame
#     and frames[32][1] will give dataframe of 1st person in 32nd frame
#     if they are present
# =============================================================================
    
    json_files = [pos_json for pos_json in os.listdir(path_to_json) \
                     if pos_json.endswith('.json')]
    
    frames=[]
        
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            people=[]
            for j in range(0,len(json_text['people'])):
                person=json_text['people'][j]['pose_keypoints']
                xyc=pd.DataFrame({'x': person[i*3],\
                                  'y':person[i*3+1],\
                                  'c':person[i*3+2]} \
                     for i in range(0,18))
                people.append(xyc)
            frames.append(people)
    return frames
    '''
def plot_frames(frames):
    colors = cm.rainbow(np.linspace(0, 1, 18))
    a=1
#    b=1
        


    for i in range(0,18):
        if a==1:
            plt.scatter(frames[:,i,0],frames[:,i,1],c=colors[i],label=cols[i])
#        if b==1:    
#            plt.scatter(frames2[:,i,0],frames2[:,i,1],c=colors[i],label=cols[i])
    plt.gca().invert_yaxis()
    plt.show()
    '''

#for i in range(0,len(frames)):
#    if a==1:
#        for j in range(0,18):
#            plt.scatter(frames[i,j,0],frames[i,j,1],c=colors[j],label=cols[j])
#        plt.xlim(-20,1280)
#        plt.ylim(-20,720)
#        plt.gca().invert_yaxis()
#        plt.show()        
#    
#    if b==1:    
#        for j in range(0,18):
#            plt.scatter(frames2[i,j,0],frames2[i,j,1],c=colors[j],label=cols[j])
#        plt.xlim(-20,1280)
#        plt.ylim(-20,720)
#        plt.gca().invert_yaxis()
#        plt.show()
#    print(i)

#plt.legend(loc='best')
if __name__ == '__main__':
    path_to_json = ''
    frames=load_frames_as_DF(path_to_json)
#    plot_frames(frames)