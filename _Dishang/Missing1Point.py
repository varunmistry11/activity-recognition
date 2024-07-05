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
from dissimilarity import getSimilarPersons


if __name__ == '__main__':
    root=Roots('../Csv/','../Json/','../Raw video/')
    
    actionLabels = ['eating',
                'kicking',
                'pushing2',
                'running',
                'snatching2',
                'vaulting',
                'walking',
                'pushing',
                'snatching'
                ]
    videosPerActionLabel = {'eating' : 5,
                            'kicking' : 6,
                            'pushing' : 8,
                            'pushing2' : 4,
                            'running' : 8,
                            'snatching' : 8,
                            'snatching2' : 1,
                            'vaulting' : 8,
                            'walking' : 16}
    plot=0
    window=5
    
    if plot:
        colors=cm.rainbow(np.linspace(0, 1, window))
    
    floor=math_floor(window/2)
    ceil =math_ceil(window/2)
    
    for activity in actionLabels:
        for number in range(1,videosPerActionLabel[activity]):        
            number=str(number)
            
            path=Paths().fromRoots(root,activity,number)
    
            csv =pd.read_csv(path.csv)
            frames=rjsons.load_frames_as_DF(path.json)        
            df=pd.DataFrame(frames)    
            
            result=[]
            
            for csv_i in range(0,len(csv)):
                start=csv.iloc[csv_i]['start']
                end=csv.iloc[csv_i]['end']
                
                mp,mpc=mvh.get_mp_mpc(df,start,end)
                
                #iterate over frames
                #i is relative frame number
                for i in range(floor,end-start-ceil):
                    mp_=mp[i]
                    mpc_=mpc[i]
                    
                    #for each frame iterate over number of persons
                    #j is person number per frame
                    for j in range(0,len(mpc_)):
                        mpc_person=mpc_[j]
                        
                        #for each person iterate over missing points
                        #x is missing body-keypoint
                        for x in mp_[j].index:
    
                            #count the corresponding missing points in window
                            count=0
                            
                            persons=getSimilarPersons(df,start+i-floor,start+i,
                                                     start+i+ceil,j)
                            for frame_ in range(i-floor,i+ceil):
                                try:
                                    if mp[frame_][persons[frame_-i+floor]][x]>0:
                                        count=count+1
                                except KeyError: # means it has valid point
                                    continue
                                except IndexError: # means frame_ is out of range
                                    continue
                                except TypeError: # means same person doesnt exist
                                    print('DEBUG counting',frame_,i,x)
                                    continue
            
                            if (count==1):
                                l=start+i #actual frame number
                                print(persons)
                                
                                sumx,sumy,sumc=mvh.get_sumx_sumy_sumc(
                                        df,l-floor,l+ceil,x,persons,1)
                                try:
                                    x_coor=sum(sumx)/len(sumx) 
                                    y_coor=sum(sumy)/len(sumy)
                                    df.iloc[l][j]['x'].iloc[x]=x_coor
                                    df.iloc[l][j]['y'].iloc[x]=y_coor
                                    result.append([l,x_coor,y_coor])
                                    print(path.json+'\n',
                                            'Frame:\t',i+start,
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
                                
#                                    mvh.update_json(path.json,number,l,j,x,
#                                                    persons,x_coor,y_coor,0)
                                except ZeroDivisionError:
        #                            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA\n DIV BY ZERO')
                                    continue