import json
import csv
import os
import math

pathToPreProcessingDirectory = 'C:/Varun/VJTI/Btech/Project/PreProcessing/'

#actionLabels = ['eating', 'kicking', 'pushing', 'running', 'snatching', 'vaulting', 'walking']
actionLabels = ['eating', 'kicking', 'pushing', 'pushing2', 'running', 'snatching', 'snatching2', 'vaulting', 'walking']
videosPerActionLabel = {'eating' : 10,
                        'kicking' : 6,
                        'pushing' : 8,
                        'pushing2' : 8,
                        'running' : 8,
                        'snatching' : 8,
                        'snatching2' : 1,
                        'vaulting' : 8,
                        'walking' : 16}

def getSlopeAnglesList(keyPoints):
    slopeAnglesList = []
    # Line from 0 to 1
    if (keyPoints[3] - keyPoints[0]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[4] - keyPoints[1]) / (keyPoints[3] - keyPoints[0])
        slopeAnglesList.append(math.atan(slope))
    # Line from 1 to 2
    if (keyPoints[6] - keyPoints[3]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[7] - keyPoints[4]) / (keyPoints[6] - keyPoints[3])
        slopeAnglesList.append(math.atan(slope))
    # Line from 2 to 3
    if (keyPoints[9] - keyPoints[6]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[10] - keyPoints[7]) / (keyPoints[9] - keyPoints[6])
        slopeAnglesList.append(math.atan(slope))
    # Line from 3 to 4
    if (keyPoints[12] - keyPoints[9]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[13] - keyPoints[10]) / (keyPoints[12] - keyPoints[9])
        slopeAnglesList.append(math.atan(slope))
    # Line from 1 to 5
    if (keyPoints[15] - keyPoints[3]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[16] - keyPoints[4]) / (keyPoints[15] - keyPoints[3])
        slopeAnglesList.append(math.atan(slope))
    # Line from 5 to 6
    if (keyPoints[18] - keyPoints[15]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[19] - keyPoints[16]) / (keyPoints[18] - keyPoints[15])
        slopeAnglesList.append(math.atan(slope))
    # Line from 6 to 7
    if (keyPoints[21] - keyPoints[18]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[22] - keyPoints[19]) / (keyPoints[21] - keyPoints[18])
        slopeAnglesList.append(math.atan(slope))
    # Line from 1 to 8
    if (keyPoints[24] - keyPoints[3]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[25] - keyPoints[4]) / (keyPoints[24] - keyPoints[3])
        slopeAnglesList.append(math.atan(slope))
    # Line from 8 to 9
    if (keyPoints[27] - keyPoints[24]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[28] - keyPoints[25]) / (keyPoints[27] - keyPoints[24])
        slopeAnglesList.append(math.atan(slope))
    # Line from 9 to 10
    if (keyPoints[30] - keyPoints[27]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[31] - keyPoints[28]) / (keyPoints[30] - keyPoints[27])
        slopeAnglesList.append(math.atan(slope))
    # Line from 1 to 11
    if (keyPoints[33] - keyPoints[3]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[34] - keyPoints[4]) / (keyPoints[33] - keyPoints[3])
        slopeAnglesList.append(math.atan(slope))
    # Line from 11 to 12
    if (keyPoints[36] - keyPoints[33]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[37] - keyPoints[34]) / (keyPoints[36] - keyPoints[33])
        slopeAnglesList.append(math.atan(slope))
    # Line from 12 to 13
    if (keyPoints[39] - keyPoints[36]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[40] - keyPoints[37]) / (keyPoints[39] - keyPoints[36])
        slopeAnglesList.append(math.atan(slope))
    # Line from 0 to 14
    if (keyPoints[42] - keyPoints[0]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[43] - keyPoints[1]) / (keyPoints[42] - keyPoints[0])
        slopeAnglesList.append(math.atan(slope))
    # Line from 14 to 16
    if (keyPoints[48] - keyPoints[42]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[49] - keyPoints[43]) / (keyPoints[48] - keyPoints[42])
        slopeAnglesList.append(math.atan(slope))
    # Line from 0 to 15
    if (keyPoints[45] - keyPoints[0]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[46] - keyPoints[1]) / (keyPoints[45] - keyPoints[0])
        slopeAnglesList.append(math.atan(slope))
    # Line from 15 to 17
    if (keyPoints[51] - keyPoints[45]) == 0:
        slopeAnglesList.append(math.pi / 2)
    else:
        slope = (keyPoints[52] - keyPoints[46]) / (keyPoints[51] - keyPoints[45])
        slopeAnglesList.append(math.atan(slope))

    return slopeAnglesList

def getSimilarity(keyPoints1, keyPoints2):
    similarity = float(0)

    for i in range(len(keyPoints1)):
        if i % 3 != 2 and (keyPoints1[i] != 0 and keyPoints2[i] != 0):
            similarity += (keyPoints1[i] - keyPoints2[i]) ** 2
    similarity = math.sqrt(similarity)
    '''
    slopeAnglesList1 = getSlopeAnglesList(keyPoints1)
    slopeAnglesList2 = getSlopeAnglesList(keyPoints2)
    similarityDueToSlopes = float(0)
    for i in range(len(slopeAnglesList1)):
        similarityDueToSlopes += (slopeAnglesList1[i] - slopeAnglesList2[i]) ** 2
    similarityDueToSlopes = math.sqrt(similarityDueToSlopes)

    similarity = similarityDueToSlopes / similarity
    '''
    return similarity

def getMinSimilarityIndex(previousData, previousPersonIndex, currentData, remainingPeopleSet):
    minSimilarity = float("inf")
    minSimilarityIndex = float("inf")
    for currentRemainingPersonIndex in remainingPeopleSet:
        if isPersonInThresholdRegion(previousData["people"][previousPersonIndex]["pose_keypoints"], currentData["people"][currentRemainingPersonIndex]["pose_keypoints"]):
            #print('Person is in threshold region')
            currentSimilarity = getSimilarity(previousData["people"][previousPersonIndex]["pose_keypoints"], currentData["people"][currentRemainingPersonIndex]["pose_keypoints"])
            if currentSimilarity < minSimilarity:
                minSimilarity = currentSimilarity
                minSimilarityIndex = currentRemainingPersonIndex
    return minSimilarityIndex, minSimilarity

def isPersonInThresholdRegion(person1_keypoints, person2_keypoints):
    ############
    ### Option 1
    ############
    '''
    Checks if person2 is within the threshold region of person1
    Threshold region of person1 is defined as
    |                     |
    |       height        |
    |                     |
    |width |person1| width|
    |                     |
    |       height        |
    |                     |
    where width = width of person1 and height = height of person1
    '''
    leftOfPerson1, rightOfPerson1, topOfPerson1, bottomOfPerson1 = getBoundsOfPerson(person1_keypoints)
    widthOfPerson1 = abs(leftOfPerson1 - rightOfPerson1)
    heightOfPerson1 = abs(topOfPerson1 - bottomOfPerson1)
    thresholdLeft = leftOfPerson1 - widthOfPerson1
    thresholdRight = rightOfPerson1 + widthOfPerson1
    thresholdTop = topOfPerson1 - heightOfPerson1
    thresholdBottom = bottomOfPerson1 + heightOfPerson1
    leftOfPerson2, rightOfPerson2, topOfPerson2, bottomOfPerson2 = getBoundsOfPerson(person2_keypoints)
    if leftOfPerson2 < thresholdRight or rightOfPerson2 > thresholdLeft or topOfPerson2 < thresholdBottom or bottomOfPerson2 > thresholdTop:
        return True
    return False
    #############
    #### Option 2
    #############
    '''
    Using percentage of intersecting area
    '''
    '''
    leftOfPerson1, rightOfPerson1, topOfPerson1, bottomOfPerson1 = getBoundsOfPerson(person1_keypoints)
    leftOfPerson2, rightOfPerson2, topOfPerson2, bottomOfPerson2 = getBoundsOfPerson(person2_keypoints)
    left = max(leftOfPerson1, leftOfPerson2)
    right = min(rightOfPerson1, rightOfPerson2)
    bottom = min(bottomOfPerson1, bottomOfPerson2)
    top = max(topOfPerson1, topOfPerson2)
    if left < right and top < bottom:
        intersectionArea = (right - left) * (bottom - top)
        person1Area = (rightOfPerson1 - leftOfPerson1) * (bottomOfPerson1 - topOfPerson1)
        if (intersectionArea / person1Area) >= 0.3:
            return True
    return False
    '''

def getJSONData(actionLabel, videoNum, frameNumber):
    data = json.load(open('../Json/' + str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_%012d_keypoints.json' % frameNumber))
    return data

def getRemainingPeopleSet(numberOfPeople):
    setList = []
    for i in range(numberOfPeople):
        setList.append(i)
    return set(setList)

def writeSimilarityToJSON(actionLabel, videoNum, similarityDictionary):
    with open('similarity/' + str(actionLabel) + '/' + str(videoNum) + '.json', 'w') as JSONfile:
        json.dump(similarityDictionary, JSONfile)

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

def orderMultiplePeopleInOutput():
    for actionLabel in actionLabels:
        numOfVideos = videosPerActionLabel[actionLabel]
        #print('No. of videos = ' + str(numOfVideos))
        for videoNum in range(1, numOfVideos + 1):
            print(str(actionLabel) + ' video ' + str(videoNum))
            csvfile = open('../Csv/' + str(actionLabel) + '/' + str(videoNum) + '.csv')
            reader = csv.reader(csvfile, delimiter=',')
            firstRow = True
            similarityDictionary = {}
            for row in reader:
                if firstRow :
                    firstRow = False
                else :
                    print('start = ' + row[0] + ' end = ' + row[1])
                    #similarityList = []
                    previousFrameHadMultiplePeople = False
                    previousData = {}
                    for frameNumber in range(int(row[0]), int(row[1]) + 1):
                        data = getJSONData(actionLabel, videoNum, frameNumber)
                        numberOfPeople = len(data["people"])
                        if numberOfPeople > 1:
                            if previousFrameHadMultiplePeople:
                                previousNumberOfPeople = len(previousData["people"])
                                if previousNumberOfPeople < numberOfPeople:
                                    remainingPeopleSet = getRemainingPeopleSet(numberOfPeople)
                                    currentSimilarityList = []
                                    for previousPersonIndex in similarityDictionary[frameNumber - 1]:
                                        minSimilarityIndex, minSimilarity = getMinSimilarityIndex(previousData, previousPersonIndex, data, remainingPeopleSet)
                                        if minSimilarityIndex == float("inf"):
                                            currentSimilarityList.append(-1)
                                        else:
                                            remainingPeopleSet.remove(int(minSimilarityIndex))
                                            currentSimilarityList.append(int(minSimilarityIndex))
                                    #similarityList.append(currentSimilarityList)
                                    for index in remainingPeopleSet:
                                        currentSimilarityList.append(index)
                                    similarityDictionary[frameNumber] = currentSimilarityList
                                elif previousNumberOfPeople > numberOfPeople:
                                    remainingPeopleSet = getRemainingPeopleSet(numberOfPeople)
                                    currentSimilarityList = []
                                    currentToPreviousDict = {}
                                    for previousPersonIndex in similarityDictionary[frameNumber - 1]:
                                        if previousPersonIndex != -1:
                                            minSimilarityIndex, minSimilarity = getMinSimilarityIndex(previousData, previousPersonIndex, data, remainingPeopleSet)
                                            if minSimilarityIndex != float("inf"):
                                                if minSimilarityIndex in currentToPreviousDict:
                                                    if minSimilarity < currentToPreviousDict[minSimilarityIndex][1]:
                                                        currentToPreviousDict[minSimilarityIndex] = [previousPersonIndex, minSimilarity]
                                                else:
                                                    currentToPreviousDict[minSimilarityIndex] = [previousPersonIndex, minSimilarity]
                                            '''
                                            if len(remainingPeopleSet) == 0:
                                                break
                                            minSimilarityIndex, minSimilarity = getMinSimilarityIndex(previousData, previousPersonIndex, data, remainingPeopleSet)
                                            remainingPeopleSet.remove(int(minSimilarityIndex))
                                            currentSimilarityList.append(int(minSimilarityIndex))
                                            '''
                                    previousToCurrentDict = {}
                                    for currentIndex in currentToPreviousDict:
                                        previousToCurrentDict[currentToPreviousDict[currentIndex][0]] = currentIndex
                                    for previousPersonIndex in similarityDictionary[frameNumber - 1]:
                                        if previousPersonIndex == -1:
                                            currentSimilarityList.append(-1)
                                        elif previousPersonIndex in previousToCurrentDict:
                                            currentSimilarityList.append(previousToCurrentDict[previousPersonIndex])
                                        else:
                                            currentSimilarityList.append(-1)
                                    #similarityList.append(currentSimilarityList)
                                    similarityDictionary[frameNumber] = currentSimilarityList
                                else:
                                    remainingPeopleSet = getRemainingPeopleSet(numberOfPeople)
                                    currentSimilarityList = []
                                    for previousPersonIndex in similarityDictionary[frameNumber - 1]:
                                        if previousPersonIndex == -1:
                                            currentSimilarityList.append(-1)
                                        else:
                                            minSimilarityIndex, minSimilarity = getMinSimilarityIndex(previousData, previousPersonIndex, data, remainingPeopleSet)
                                            if minSimilarityIndex == float("inf"):
                                                currentSimilarityList.append(-1)
                                            else:
                                                remainingPeopleSet.remove(int(minSimilarityIndex))
                                                currentSimilarityList.append(int(minSimilarityIndex))
                                    #similarityList.append(currentSimilarityList)
                                    for index in remainingPeopleSet:
                                        currentSimilarityList.append(index)
                                    similarityDictionary[frameNumber] = currentSimilarityList
                            else:
                                previousFrameHadMultiplePeople = True
                                listofPeople = []
                                for i in range(numberOfPeople):
                                    listofPeople.append(i)
                                #similarityList.append(listofPeople)
                                similarityDictionary[frameNumber] = listofPeople
                        else:
                            previousFrameHadMultiplePeople = False
                        previousData = data
                    #print(similarityList)
            print('Similarity Dictionary')
            for key, value in similarityDictionary.items():
                print('Frame ' + str(key) + ' : ' + str(value))
            writeSimilarityToJSON(actionLabel, videoNum, similarityDictionary)

if __name__ == "__main__":
    orderMultiplePeopleInOutput()
