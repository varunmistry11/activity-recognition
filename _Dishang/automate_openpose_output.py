# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:46:53 2018

@author: Dishang
"""
import subprocess
import os

# =============================================================================
# 1.Put this file in your openpose demo folder.
# 2.Put your videos folder in the path provided by root variable (relative path).
# 3.The videos folder will have folders for different activities which contain 
#     the respective videos
#     
# Command that runs is:
#
# bin\OpenPoseDemo.exe --video input_file --write_video output_file 
#     --write_json output_dir
# 
# =============================================================================

root='examples/ForTraining/'

for x in os.listdir(root):
#    print(x)
    current_folder=root+x
    print(current_folder)

    for file in os.listdir(current_folder):
        if file.endswith('.mp4'):

            input_file=current_folder +'/'+file

            output_dir=current_folder+'/'+os.path.splitext(file)[0]+'_output'

            output_file=os.path.splitext(file)[0]+'_output'+\
                        os.path.splitext(file)[1]
#            print(output_dir)
            os.mkdir(output_dir)

            subprocess.call(('bin\OpenPoseDemo.exe','--video',input_file, \
                             '--write_video',output_file,'--write_json',\
                             output_dir))
            print(output_file+' DONE')