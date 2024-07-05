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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import reading_jsons as rjsons
import glob
import json

def missing_points_count(person):
    try:
        return (person==0).astype(int).sum(axis=1).value_counts()[3]
    except KeyError:
        return 0
    
def missing_points(person):
    try:
        return ((person==0).astype(int).sum(axis=1)==3).astype(int)
    except KeyError:
        return 0

def get_mp_mpc(df,start,end,p=0):
    mp =[]
    mpc=[]
    for i in range(start,end):
        try:
            if df.iloc[i][p] is not None:
                mpc.append(missing_points_count(df.iloc[i][p]))
                a=missing_points(df.iloc[i][p])
                a=a[a!=0]
                mp.append(a)
            else: 
                mp.append([])
                mpc.append([])
        except IndexError:            
            print(i,start,'AA\n FRAMES LOST')
    return mp,mpc
    
def get_sumx_sumy_sumc(df,l,x,p=0,filt=1):
    sumx=[]
    sumy=[]
    sumc=[]
    for k in [l-2,l-1,l+1,l+2]:
        if df.iloc[k][p] is not None:
            sumx.append(df.iloc[k][p]['x'].iloc[x])
            sumy.append(df.iloc[k][p]['y'].iloc[x])
            sumc.append(df.iloc[k][p]['c'].iloc[x])
        else: print('aa')
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

def update_json(json_path,number,frame,keypoint,p,x,y,c=0):
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
                      ' VALUES:'+str(x)+', '+str(y)+str(c)+'\n')

if __name__ == '__main__':
    
    root='../Openpose output/'
    csv_root='../Csv/'
    activity='eating'
    plot=1   
    
    
    for number in range(2,3):
        number=str(number)
        path_to_json = root+activity+'/'+number+'_output'
        csv =csv_root+activity+'/'+number+'.csv'
        csv =pd.read_csv(csv)
        
        frames=rjsons.load_frames_as_DF(path_to_json)
        
        colors = cm.rainbow(np.linspace(0, 1, 5))
    
        df=pd.DataFrame(frames)
        
        result=[]
        p=0
        for csv_i in range(0,len(csv)):
            start=csv.iloc[csv_i]['start']
            end=csv.iloc[csv_i]['end']
    
            mp,mpc=get_mp_mpc(df,start,end,p)
    
            for i in range(2,end-start-3):
                a=mp[i]
                for j in range(0,len(a)):
                    x=a.index[j]
                    count=0
                    for k in range(i-2,i+3):
                        try:
                            if mp[k][x]>0:
                                count=count+1
                        except KeyError:
                            continue
                        except IndexError:
                            continue
    
                    if (count==1):
                        l=start+i
                        
                        sumx,sumy,sumc=get_sumx_sumy_sumc(df,l,x,p,1)
                        try:
                            x_coor=sum(sumx)/len(sumx) 
                            y_coor=sum(sumy)/len(sumy)
                            df.iloc[l][p]['x'].iloc[x]=x_coor
                            df.iloc[l][p]['y'].iloc[x]=y_coor
                            result.append([l,x_coor,y_coor])
                            print(i+start,x,count)
                            print(x_coor,y_coor)

                            if plot:
                                plt.scatter(sumx[:],sumy[:],c=colors)
                                plt.scatter(x_coor,y_coor,c='black')
                                plt.xlim(-20,1280)
                                plt.ylim(-20,720)
                                plt.gca().invert_yaxis()
                                plt.show()
                        
#                            update_json(path_to_json,number,l,x,p,x_coor,y_coor,0)
                        except ZeroDivisionError:
#                            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA\n DIV BY ZERO')
                            continue
    '''
        result.sort()
        import cv2
        import time
        cap = cv2.VideoCapture(path_to_json+'/'+number+'_output.mp4')
        k=0
        i=0
        while(k<len(df)):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame=cv2.putText(frame,str(k),(100,100),
                              cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            
            frame=cv2.circle(frame,(result[i][1].astype(int),
                                result[i][2].astype(int)),10,(255,255,255),-1)
            if i<len(result) and k==result[i][0]:
                i=i+1
                time.sleep(1)
                cv2.imshow('frame',frame)
                time.sleep(1)
            else:
                cv2.imshow('frame',frame)
            
            k=k+1
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        '''