# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 23:02:34 2018

@author: Dishang
"""
import math

def getDissimilarity(keyPoints1, keyPoints2):
    dissimilarity = float(0)
    for i in range(len(keyPoints1)):
        if i % 3 != 2 and (keyPoints1[i] != 0 and keyPoints2[i] != 0):
            dissimilarity += (keyPoints1[i] - keyPoints2[i]) ** 2
    dissimilarity = math.sqrt(dissimilarity)
    return dissimilarity

def withinThreshold(d1,d2):
#    left=0,right=1,top=2,bottom=3
    d1_box=[]
    d1_box.append(d1[d1>0].dropna()[['x']].min().values[0])  #left
    d1_box.append(d1[d1>0].dropna()[['x']].max().values[0])  #right
    d1_box.append(d1[d1>0].dropna()[['y']].max().values[0])  #top  
    d1_box.append(d1[d1>0].dropna()[['y']].min().values[0])  #bottom 
    
    d1_width=d1_box[1]-d1_box[0]    #right-left
    d1_height=d1_box[2]-d1_box[3]   #top-bottom
    
    d2_box=[]
    d2_box.append(d2[d2>0].dropna()[['x']].min().values[0])  #left
    d2_box.append(d2[d2>0].dropna()[['x']].max().values[0])  #right
    d2_box.append(d2[d2>0].dropna()[['y']].max().values[0])  #top  
    d2_box.append(d2[d2>0].dropna()[['y']].min().values[0])  #bottom
    
    threshold_box=[]
    threshold_box.append(d1_box[0]-d1_width)
    threshold_box.append(d1_box[1]+d1_width)
    threshold_box.append(d1_box[2]+d1_height)
    threshold_box.append(d1_box[3]-d1_height)
    
    if d2_box[0]<threshold_box[1]:
        if d2_box[2]>threshold_box[3] or d2_box[3]<threshold_box[2]:
            return True
        else: return False
        
    if d2_box[1]>threshold_box[0]:
        if d2_box[2]>threshold_box[3] or d2_box[3]<threshold_box[2]:
            return True
    return False

def getSimilarPersons(df,start,frame,end,current_person):

    d1=df.iloc[frame][current_person]

    same_persons=[]

    for frame_ in range(start,end):

        if frame_==frame:
            same_persons.append(current_person)

        else:
            
# =============================================================================
#            for each person in frame:
#               calculate dissimilarity
#               if dissimilarity<min_dissimilarity:
#                   if person is within threshold:
#                       update min dissimilarity 
# =============================================================================
            
            min_dissimilarity=float('inf')
            
            # Technically index should be -1, but indexing with -1 means 
            # accessing the last element, hence using infinity
            min_dissimilarity_index=float('inf')
            
            # iterate over each person (as i)
            for i in range(0,len(df.iloc[frame_])):
                if df.iloc[frame_][i] is not None:
                    d2=df.iloc[frame_][i]
                    dissimilarity=getDissimilarity(
                                    d1[['x','y','c']].values.flatten(),
                                    d2[['x','y','c']].values.flatten())
                    if min_dissimilarity>dissimilarity:
                        if withinThreshold(d1,d2):
                            min_dissimilarity=dissimilarity
                            min_dissimilarity_index=i
                else:
#                    print('DEBUG getSimilarPersons',frame_,i)
                    break
            same_persons.append(min_dissimilarity_index)
    return same_persons

    