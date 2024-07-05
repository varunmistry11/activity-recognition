# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:00:25 2018

@author: Dishang
"""
import MissingValuesHelper as mvh
import reading_jsons as rjsons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import ceil as math_ceil,floor as math_floor
from MyCollections import Roots,Paths

if __name__ == '__main__':
    
    root='../Openpose output/'
    csv_root='../Csv/'
    activity='eating'
    plot=1
    window=5
    
    if plot:
        colors=cm.rainbow(np.linspace(0, 1, window))
    
    floor=math_floor(window/2)
    ceil =math_ceil(window/2)
    
    for number in range(2,3):
        
        number=str(number)
        path_to_json = root+activity+'/'+number+'_output'
        csv =csv_root+activity+'/'+number+'.csv'
        csv =pd.read_csv(csv)
        
        frames=rjsons.load_frames_as_DF(path_to_json)
        df=pd.DataFrame(frames)
        
        result=[]
        person=0
        
        for csv_i in range(0,len(csv)):
            start=csv.iloc[csv_i]['start']
            end=csv.iloc[csv_i]['end']
            
            mp,mpc=mvh.get_mp_mpc(df,start,end,person)
            
            #iterate over frames
            for i in range(floor,end-start-ceil):
                a=mp[i]     #a contains list of missing points for a person
                
                #iterate over all points of a frame
                for j in range(0,len(a)):
                    x=a.index[j]
                    count=0
                    for frame_ in range(i-floor,i+ceil):
                        try:
                            if mp[frame_][x]>0:
                                count=count+1
                        except KeyError:
                            continue
                        except IndexError:
                            continue
    
                    if (count==1):
                        l=start+i
                        
                        sumx,sumy,sumc=mvh.get_sumx_sumy_sumc(df,l,x,person,1,
                                                              window)
                        try:
                            x_coor=sum(sumx)/len(sumx) 
                            y_coor=sum(sumy)/len(sumy)
                            df.iloc[l][person]['x'].iloc[x]=x_coor
                            df.iloc[l][person]['y'].iloc[x]=y_coor
                            result.append([l,x_coor,y_coor])
                            print('Frame:\t',i+start,
                                  '\nBody Keypoint:\t',x,
                                  '\nNumMissingPoints:',count)
                            print('X,Y :',x_coor,',',y_coor)

                            if plot:
                                plt.scatter(sumx[:],sumy[:],c=colors)
                                plt.scatter(x_coor,y_coor,c='black')
                                plt.xlim(-20,1280)
                                plt.ylim(-20,720)
                                plt.gca().invert_yaxis()
                                plt.show()
                        
#                            mvh.update_json(path_to_json,number,l,x,p,x_coor,y_coor,0)
                        except ZeroDivisionError:
#                            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA\n DIV BY ZERO')
                            continue