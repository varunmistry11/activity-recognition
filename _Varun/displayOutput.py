import numpy as np
import cv2
import json
import os
import csv
import matplotlib.pyplot as plot

PATH = '/media/varun/Windows/Varun/VJTI/Btech/Project/BitBucket/ashar-varun/'
#pathToVideos = 'C:/Varun/VJTI/Btech/Project/PreProcessing/'
pathToVideos = '/media/varun/WIndows/Varun/VJTI/Btech/Project/PreProcessing/'
#pathToSimilarityFolder = 'C:/Varun/VJTI/Btech/Project/Elimination result-20180214T171202Z-001/Elimination result/'
pathToSimilarityFolder = ''
#actionLabels = ['eating', 'kicking', 'pushing', 'running', 'snatching', 'vaulting', 'walking']
actionLabels = ['eating', 'kicking', 'pushing', 'pushing2', 'running', 'snatching', 'snatching2', 'vaulting', 'walking']
'''
videosPerActionLabel = {'eating' : 5,
                        'kicking' : 6,
                        'pushing' : 8,
                        'running' : 8,
                        'snatching' : 8,
                        'vaulting' : 8,
                        'walking' : 16}
'''
videosPerActionLabel = {'eating' : 10,
                        'kicking' : 6,
                        'pushing' : 8,
                        'pushing2' : 8,
                        'running' : 8,
                        'snatching' : 8,
                        'snatching2' : 1,
                        'vaulting' : 8,
                        'walking' : 16}

colours = [(0, 0, 153), (153, 0, 0), (153, 102, 0), (153, 153, 0), (153, 51, 0),
            (102, 153, 0), (51, 153, 0), (0, 153, 0), (0, 153, 51), (0, 153, 102),
            (0, 153, 153), (0, 102, 153), (0, 51, 153), (51, 0, 153), (153, 0, 153),
            (102, 0, 153), (153, 0, 102)]

def getBoundsOfPerson(pose_keypoints):
    length = len(pose_keypoints)
    left = None
    right = None
    top = None
    bottom = None

    for i in range(length):
        if i % 3 == 0:
            if pose_keypoints[i] != 0 and (left == None or pose_keypoints[i] < left):
                left = pose_keypoints[i]
            if right == None or pose_keypoints[i] > right:
                right = pose_keypoints[i]
        elif i % 3 == 1:
            if pose_keypoints[i] != 0 and (top == None or pose_keypoints[i] < top):
                top = pose_keypoints[i]
            if bottom == None or pose_keypoints[i] > bottom:
                bottom = pose_keypoints[i]

    return left, right, top, bottom

def getFramesListFromCSV(actionLabel, videoNum):
    csvfile = open(PATH + 'Csv/' + str(actionLabel) + '/' + str(videoNum) + '.csv')
    reader = csv.reader(csvfile, delimiter=',')
    firstRow = True
    framesList = []
    for row in reader:
        if firstRow :
            firstRow = False
        else :
            framesList.append([int(row[0]), int(row[1])])
    return framesList

def getJSONData(actionLabel, videoNum, frameNumber):
    data = json.load(open(PATH + 'Json/' + str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_%012d_keypoints.json' % frameNumber))
    '''
    if frameNumber < 10:
        data = json.load(open(str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_00000000000' + str(frameNumber) + '_keypoints.json'))
    elif frameNumber < 100:
        data = json.load(open(str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_0000000000' + str(frameNumber) + '_keypoints.json'))
    else:
        data = json.load(open(str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_000000000' + str(frameNumber) + '_keypoints.json'))
    '''
    return data

def isFrameNumberinFramesList(framesList, frameNumber):
    for row in framesList:
        if frameNumber >= row[0] and frameNumber <= row[1]:
            return True
    return False

def plotBoundingBox(frame, left, right, top, bottom, colour_i):
    colour_i = colour_i % len(colours)
    # top line
    frame = cv2.line(frame, (left, top), (right, top), colours[colour_i], 3)
    # right line
    frame = cv2.line(frame, (right, top), (right, bottom), colours[colour_i], 3)
    # bottom line
    frame = cv2.line(frame, (right, bottom), (left, bottom), colours[colour_i], 3)
    # left line
    frame = cv2.line(frame, (left, bottom), (left, top), colours[colour_i], 3)
    return frame

def draw_skeleton(img, json_persons):
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

def showOutput(actionLabel, videoNum):
    cap = cv2.VideoCapture(PATH + 'Raw_video/' + actionLabel + '/' + str(videoNum) + '.mp4')
    success, frame = cap.read()
    frameNumber = 0
    framesList = getFramesListFromCSV(actionLabel, videoNum)
    similarityData = json.load(open('./similarity/' + actionLabel + '/' + str(videoNum) + '.json'))
    previousFrameHadMultiplePeople = False
    indexList = []
    while(success):
        if isFrameNumberinFramesList(framesList, frameNumber):
            font = cv2.FONT_HERSHEY_SIMPLEX
            #frame = cv2.putText(frame, 'Action', (10, 600), font, 3, (0, 0, 150), 4, cv2.LINE_AA)
            openPoseData = getJSONData(actionLabel, videoNum, frameNumber)
            numberOfPeople = len(openPoseData["people"])
            print('frame number ', frameNumber)
            frame = draw_skeleton(frame, openPoseData["people"])
            if numberOfPeople > 1:
                if not previousFrameHadMultiplePeople:
                    previousFrameHadMultiplePeople = True
                indexList = similarityData[str(frameNumber)]
                colour_i = 0
                #frame = cv2.putText(frame, str(frameNumber) + ' ' + str(indexList), (500, 600), font, 3, (0, 0, 150), 4, cv2.LINE_AA)
                for index in indexList:
                    if index != -1:
                        print(index)
                        person_keypoints = openPoseData["people"][index]["pose_keypoints"]
                        left, right, top, bottom = getBoundsOfPerson(person_keypoints)
                        frame = plotBoundingBox(frame, int(left), int(right), int(top), int(bottom), colour_i)
                        colour_i += 1
            else:
                previousFrameHadMultiplePeople = False
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            #frame = cv2.putText(frame, 'No action', (10, 600), font, 3, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        #plot.imshow(frame, interpolation = 'none')
        #plot.show()
        success, frame = cap.read()
        frameNumber += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    actionLabel = 'pushing2'
    videoNum = 1
    showOutput(actionLabel, videoNum)
    '''
    data = getJSONData(actionLabel, videoNum, 179)
    print('for person 0')
    person_keypoints = data["people"][0]["pose_keypoints"]
    left, right, top, bottom = getBoundsOfPerson(person_keypoints)
    print(left, right, top, bottom)
    print('for person 1')
    person_keypoints = data["people"][1]["pose_keypoints"]
    left, right, top, bottom = getBoundsOfPerson(person_keypoints)
    print(left, right, top, bottom)
    '''
    '''
    framesList = getFramesListFromCSV(actionLabel, videoNum)
    print(framesList)
    print(isFrameNumberinFramesList(framesList, 304))
    '''
