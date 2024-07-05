# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:07:33 2018

@author: Dishang
"""

class Roots():
    def __init__(self,csv ,json ,video ):
        self.csv   = csv
        self.json  = json
        self.video = video


class Paths():
    def __init__(self,csv='' ,json='' ,video='' ):
        self.csv   = csv
        self.json  = json
        self.video = video
        
    def fromRoots(self,Roots,activity,number):
        self.csv   = Roots.csv  +activity+'/'+number+'.csv'
        self.json  = Roots.json +activity+'/'+number+'_output'
        self.video = Roots.video+activity+'/'+number+'.mp4'
        return self
    
    