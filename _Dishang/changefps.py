# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:52:26 2018

@author: Dishang
"""
import subprocess
import os

root='examples/videos/'
for x in os.listdir(root):
    # x is activity folder
    print(x)
    current_folder=root+x

    #    if(x=='Snatching'):
    #        print(1)
    #    else: continue
    for file in os.listdir(current_folder):
        print(file)

        video_input_path=current_folder+'/'+file

        video_output_path=current_folder+'/'+os.path.splitext(file)[0]+ \
                        '_12fps'+os.path.splitext(file)[1]
                        
        c = 'ffmpeg -y -i ' \
            + video_input_path+\
            ' -r 12  -c:v libx264 -b:v 3M -strict -2 -movflags faststart ' \
            + video_output_path
        subprocess.call(c, shell=True)
    
        print(video_output_path)
