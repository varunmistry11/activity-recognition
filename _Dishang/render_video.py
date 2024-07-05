# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:47:28 2018

@author: Dishang
"""

import cv2
import json
import matplotlib.pyplot as plt

def draw_skeleton(img,jsson_persons):
    for i in range(len(json_persons)):
        skeleton = json_persons[i]["pose_keypoints"]

#        print(re)
        # Draw a RGB line joining m to n
        #for key point m, the three points in array are at indices 3*m,
        #3*m+1,3*m+2
#        img = cv2.line(img, (int(skeleton[3*m)], int(skeleton[3*m)+1]),
#                            (int(skeleton[3*n)], int(skeleton[3*n)
#                            +1]), (r, g, b), 5)

        #Draw a blue line joining 0 to 1

        if int(skeleton[2] *100)!= 0 and int(skeleton[5] *100)!= 0:
            img = cv2.line(img, (int(skeleton[0]), int(skeleton[1])),
                                     (int(skeleton[3]), int(skeleton[4])),
                                     (0, 0, 153), 5)
        # Draw a  line joining 1 to 2
        if int(skeleton[5] *100)!= 0 and int(skeleton[8] *100)!= 0:
            img = cv2.line(img, (int(skeleton[3]), int(skeleton[4])),
                                     (int(skeleton[6]), int(skeleton[7])),
                                     (153, 0, 0), 5)
        # Draw a  line joining 2 to 3
        if int(skeleton[8] *100)!= 0 and int(skeleton[11]*100) != 0:
            img = cv2.line(img, (int(skeleton[6]), int(skeleton[7])),
                                     (int(skeleton[9]), int(skeleton[10])),
                                     (153, 102, 0), 5)
        # Draw a  line joining 3 to 4
        if int(skeleton[11]*100) != 0 and int(skeleton[14]*100) != 0:
            img = cv2.line(img, (int(skeleton[9]), int(skeleton[10])),
                                     (int(skeleton[12]), int(skeleton[13]))
                                     , (153, 153, 0), 5)
        # Draw a  line joining 1 to 5
        if int(skeleton[5] *100)!= 0 and int(skeleton[17]*100) != 0:
            img = cv2.line(img, (int(skeleton[3]), int(skeleton[4])),
                                     (int(skeleton[15]), int(skeleton[16]))
                                     , (153, 51, 0), 5)
        # Draw a  line joining 5 to 6
        if int(skeleton[17]*100) != 0 and int(skeleton[20]*100) != 0:
            img = cv2.line(img, (int(skeleton[15]), int(skeleton[16])),
                                     (int(skeleton[18]), int(skeleton[19]))
                                     , (102, 153, 0), 5)
        # Draw a  line joining 6 to 7
        if int(skeleton[20]*100) != 0 and int(skeleton[23]*100) != 0:
            img = cv2.line(img, (int(skeleton[18]), int(skeleton[19])),
                                     (int(skeleton[21]), int(skeleton[22]))
                                     , (51, 153, 0), 5)

        # Draw a  line joining 1 to 8
        if int(skeleton[5] *100)!= 0 and int(skeleton[26]*100) != 0:
            img = cv2.line(img, (int(skeleton[3]), int(skeleton[4])),
                                     (int(skeleton[24]), int(skeleton[25]))
                                     , (0, 153, 0), 5)


        # Draw a  line joining 8 to 9
        if int(skeleton[26]*100) != 0 and int(skeleton[29]*100) != 0:
            img = cv2.line(img, (int(skeleton[24]), int(skeleton[25])),
                                     (int(skeleton[27]), int(skeleton[28]))
                                     , (0, 153, 51), 5)

        # Draw a  line joining 9 to 10
        if int(skeleton[29]*100) != 0 and int(skeleton[32]*100) != 0:
            img = cv2.line(img, (int(skeleton[27]), int(skeleton[28])),
                                     (int(skeleton[30]), int(skeleton[31]))
                                     , (0, 153, 102), 5)

        # Draw a  line joining 1 to 11
        if int(skeleton[5] *100)!= 0 and int(skeleton[35]*100) != 0:
            img = cv2.line(img, (int(skeleton[3]), int(skeleton[4])),
                                     (int(skeleton[33]), int(skeleton[34]))
                                     , (0, 153, 153), 5)

        # Draw a  line joining 11 to 12
        if int(skeleton[35]*100) != 0 and int(skeleton[38]*100) != 0:
            img = cv2.line(img, (int(skeleton[33]), int(skeleton[34])), 
                                     (int(skeleton[36]), int(skeleton[37]))
                                     , (0, 102, 153), 5)

        # Draw a  line joining 12 to 13
        if int(skeleton[38]*100) != 0 and int(skeleton[41]*100) != 0:
            img = cv2.line(img, (int(skeleton[36]), int(skeleton[37])),
                                     (int(skeleton[39]), int(skeleton[40]))
                                     , (0, 51, 153), 5)

        # Draw a  line joining 0 to 14
        if int(skeleton[2] *100)!= 0 and int(skeleton[44]*100) != 0:
            img = cv2.line(img, (int(skeleton[0]), int(skeleton[1])),
                                     (int(skeleton[42]), int(skeleton[43]))
                                     , (51, 0, 153), 5)

        # Draw a  line joining 0 to 15
        if int(skeleton[2] *100)!= 0 and int(skeleton[47]*100) != 0:
            img = cv2.line(img, (int(skeleton[0]), int(skeleton[1])), 
                                     (int(skeleton[45]), int(skeleton[46]))
                                     , (153, 0, 153), 5)

        # Draw a  line joining 14 to 16
        if int(skeleton[44]*100) != 0 and int(skeleton[50]*100) != 0:
            img = cv2.line(img, (int(skeleton[42]), int(skeleton[43])),
                                     (int(skeleton[48]), int(skeleton[49]))
                                     , (102, 0, 153), 5)

        # Draw a  line joining 15 to 17
        if int(skeleton[47]*100) != 0 and int(skeleton[53]*100) != 0:
            img = cv2.line(img, (int(skeleton[45]), int(skeleton[46])), 
                                     (int(skeleton[51]), int(skeleton[52]))
                                     , (153, 0, 102), 5)
    return img    

if __name__ == '__main__':
    activity='eating'
    number=2
    number=str(number)
    video_root='../Raw video/'
    json_root='../Openpose output/'
    
    temp=frame=2
    frame=frame-2
    
    # Raw video file
    vidcap = cv2.VideoCapture(video_root+activity+'/'+number+'.mp4')    
    
    # Start video from frame number 'frame'
    vidcap.set(1,frame)
    success,img = vidcap.read()
    
    while success:

        padded_fr="%012d" %frame
        re=number+'_'+padded_fr+'_keypoints.json'
        path_to_json = json_root+activity+'/'+number+'_output'
        filename=path_to_json+'/'+re
        
        print ('Read a new frame: ', success,filename)

        with open(filename, 'r+') as f:
            json_data = json.load(f)
            json_persons = json_data["people"]
            
            img=draw_skeleton(img,json_persons)
            img=cv2.putText(img,str(frame),(100,100),
                              cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                    
#        x=45
#        y=466
#        img = cv2.circle(img,(x,y),5,(255,255,255),-1)
        
        img = cv2.resize(img, (0,0), fx = 0.75, fy = 0.75)
        cv2.imshow('frame',img)
        
#        plt.figure(figsize=(10,10))
#        plt.imshow(img, interpolation='none') # Plot the image, turn off interpolation
#        plt.show() # Show the image window
        
        frame += 1
        
        success,img = vidcap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#        if frame== temp+3 :
#            break
        
        
    vidcap.release()
    cv2.destroyAllWindows()
