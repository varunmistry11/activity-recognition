# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:03:37 2018

@author: Dishang
"""

# =============================================================================
# two previous frames=pf
# two next frames=nf
# current frame=cf
# CASES:
#     1. if pf and nf dont have missing values for a missing point in cf (say mp)
#             then mp = avg of pf and nf 
#     2. if either one or both pf and nf have missing values for corresponding 
#         missing values of cf 
#             then do nothing
#     3. if cf is starting or ending frame 
#             then do nothing
# 
# =============================================================================

#import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import reading_jsons as rjsons
import numpy as np
import glob
import json
from math import ceil as math_ceil,floor as math_floor

def missing_points_count(person,mode=1):
# =============================================================================
#     Returns number of missing points in person
#     mode indicates which input to use.
#     If using output of missing_points as input then mode=1
#     If using person array then mode=0
# =============================================================================
    if mode==0:
        try:
            return ((person==0).astype(int).sum(axis=1)==3).astype(int
                   ).value_counts()[1]
        except KeyError:
            return 0
    else:
        try:
            return person.value_counts()[1]
        except KeyError:
            return 0
    
def missing_points(person):
# =============================================================================
#     Returns an 18 point array with array[i]=1 if person[i] has a missing
#     point , i.e. person[i] contains zeroes.
# =============================================================================
    try:
        return ((person==0).astype(int).sum(axis=1)==3).astype(int)
    except KeyError:
        return 0

def get_mp_mpc(df,start,end):
# =============================================================================
#     For each frame of df between start and end, returns:
#     number of missing points in each person as mpc[frame][person] and
#     the missing points as mp[frame][person]
# =============================================================================
    
    mp =[]
    mpc=[]
    for i in range(start,end):
        try:
            mp_=[]
            mpc_=[]
            for person in range(0,len(df.iloc[i])):
                if df.iloc[i][person] is not None:
                    a=missing_points(df.iloc[i][person])
                    mpc_.append(missing_points_count(a,1))
                    a=a[a!=0]
                    mp_.append(a)
                else: 
                    pass
#                    mp_.append([])
#                    mpc_.append([])
            mp.append(mp_)
            mpc.append(mpc_)
        except IndexError:            
            print(i,start,'\n FRAMES NOT FOUND. INDEX ERROR')
    return mp,mpc
    
def get_sumx_sumy_sumc(df,start,end,keypoint,persons,filt=1):
    sumx=[]
    sumy=[]
    sumc=[]
    for i in range(start,end): #[frame-2,frame-1,frame+1,frame+2]
        person=persons[i-start]
        try:
            if df.iloc[i][person] is not None:
                if(df.iloc[i][person]['c'].iloc[keypoint]!=0):
                    sumx.append(df.iloc[i][person]['x'].iloc[keypoint])
                    sumy.append(df.iloc[i][person]['y'].iloc[keypoint])
                    sumc.append(df.iloc[i][person]['c'].iloc[keypoint])
            else: print('Person array not found at frame',i,',index',person)
        except TypeError: #means same person doesnt exist
            print('DEBUG get sums',i,keypoint)
            continue
    
    if filt:
        a=np.array(sumx)
        b=np.array(sumy)
        s=pow(a*a+b*b,0.5)
        mean=np.mean(s)
        std=np.std(s)
        zscore=np.absolute((s-mean)/std)
        z=(zscore>1.5)
#        print(z)
        for i in range(0,len(z)):
            if z[i]:
                sumx.remove(sumx[i])
                sumy.remove(sumy[i])
                sumc.remove(sumc[i])
                break
    return sumx,sumy,sumc

def update_json(json_path,number,frame,p,keypoint,persons,x,y,c=0):
    padded_fr="%012d" %frame
    re=number+'_'+padded_fr+'_keypoints.json'

    for filename in glob.glob(json_path+'/'+re):
        with open(json_path+'/'+'missing_value.log','a') as log:
            with open(filename, 'r+') as f:
                json_data = json.load(f)
                j=json_data
                j['people'][p]['pose_keypoints'][keypoint*3]=x
                j['people'][p]['pose_keypoints'][keypoint*3+1]=y
                j['people'][p]['pose_keypoints'][keypoint*3+2]=c
                f.seek(0)
                f.write(json.dumps(json_data,indent=4))
                f.truncate()
            log.write(filename+' POINT:'+str(keypoint)+\
                      ' VALUES:'+str(x)+', '+str(y)+str(c)+
                      'Persons:'+str(persons)+'\n')

if __name__ == '__main__':
   mp,mpc=get_mp_mpc(df,341,365)