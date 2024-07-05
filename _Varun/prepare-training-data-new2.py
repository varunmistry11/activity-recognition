import numpy as np
import json
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf

#actionLabels = ['pushing', 'snatching']
actionLabels = ['eating', 'kicking', 'pushing', 'pushing2', 'running', 'snatching', 'snatching2', 'vaulting', 'walking']
#actionLabels = ['eating', 'kicking', 'running', 'vaulting', 'walking']
videosPerActionLabel = {'eating' : 5,
                        'kicking' : 6,
                        'pushing' : 8,
                        'pushing2' : 4,
                        'running' : 8,
                        'snatching' : 8,
                        'snatching2' : 1,
                        'vaulting' : 8,
                        'walking' : 16}

labelToOneHot = {
    'eating' : [1, 0, 0, 0, 0, 0, 0],
    'kicking' : [0, 1, 0, 0, 0, 0, 0],
    'pushing' : [0, 0, 1, 0, 0, 0, 0],
    'running' : [0, 0, 0, 1, 0, 0, 0],
    'snatching' : [0, 0, 0, 0, 1, 0, 0],
    'vaulting' : [0, 0, 0, 0, 0, 1, 0],
    'walking' : [0, 0, 0, 0, 0, 0, 1]
}
'''
labelToOneHot = {
    'eating' : [1, 0, 0, 0, 0],
    'kicking' : [0, 1, 0, 0, 0],
    'running' : [0, 0, 1, 0, 0],
    'vaulting' : [0, 0, 0, 1, 0],
    'walking' : [0, 0, 0, 0, 1]
}
'''

def getJSONData(actionLabel, videoNum, frameNumber):
    data = json.load(open('../Json/' + str(actionLabel) + '/' + str(videoNum) + '_output/' + str(videoNum) + '_%012d_keypoints.json' % frameNumber))
    return data

def convertQueueToCoordinateDiffList(queue):
    coordinateDiffList = []
    for keypointsDiff in queue:
        for i in range(len(keypointsDiff)):
            coordinateDiffList.append(keypointsDiff[i])
    return coordinateDiffList

def getDiff(currentKeypoints, previousKeypoints):
    diff = []
    for i in range(3, 42):
        '''
        if i % 3 != 2:
            diff.append(currentKeypoints[i] - previousKeypoints[i])
        '''
        if i % 3 == 0:
            diff.append((currentKeypoints[i] - previousKeypoints[i]) / 1280)
        if i % 3 == 1:
            diff.append((currentKeypoints[i] - previousKeypoints[i]) / 720)
    return diff

def createInputNumpyArray(N):
    '''
    N : number of consecutive frames to consider
    '''
    inputList = []
    labelList = []
    previousSum = 0
    minClassTuples = 1000000000
    tuplesOfClass = {}
    print("For given N =", N)
    for actionLabel in actionLabels:
        previousSum = len(inputList)
        numOfVideos = videosPerActionLabel[actionLabel]
        for videoNum in range(1, numOfVideos + 1):
            csvfile = open('../Csv/' + str(actionLabel) + '/' + str(videoNum) + '.csv')
            reader = csv.reader(csvfile, delimiter=',')
            firstRow = True
            '''Ignore walking videos 2-5 and 9, 10 since they have multiple
            people interfering the video'''
            if actionLabel == 'walking' and ((videoNum >= 2 and videoNum <= 5) or videoNum == 9 or videoNum == 10):
                continue
            if actionLabel == 'pushing' or actionLabel == 'pushing2' or actionLabel == 'snatching' or actionLabel == 'snatching2':
                similarityData = json.load(open('similarity/' + str(actionLabel) + '/' + str(videoNum) + '.json'))
            for row in reader:
                if firstRow :
                    firstRow = False
                else :
                    startFrameNum = int(row[0])
                    endFrameNum = int(row[1])
                    queue = []
                    listOfQueues = []
                    previousData = {}
                    isFirstFrame = True
                    isFirstFrameForPerson = {}
                    for frameNumber in range(startFrameNum, endFrameNum + 1):
                        data = getJSONData(actionLabel, videoNum, frameNumber)
                        if actionLabel == 'pushing' or actionLabel == 'pushing2' or actionLabel == 'snatching' or actionLabel == 'snatching2':
                            #print(actionLabel, videoNum, frameNumber, len(data["people"]))
                            #print("number of queues", len(listOfQueues))
                            #Case: Number of people > 0
                            if len(data["people"]) > 0:
                                if isFirstFrame:
                                    isFirstFrame = False
                                    #print("First Frame in sequence")
                                elif len(previousData["people"]) > 0:

                                    if len(listOfQueues) == 0:
                                        for _ in range(len(data["people"])):
                                            listOfQueues.append([])
                                            isFirstFrameForPerson[_] = True
                                    #print("Before cases, listOfQueues", len(listOfQueues))

                                    '''Case 1: Same number of people in current frame
                                    as in previous frame'''
                                    if len(data["people"]) == len(previousData["people"]):
                                        #print("Case 1: current == previous")

                                        if len(data["people"]) == 1:
                                            '''Since similarity json files do not contain
                                            entries for 1 people hence this case
                                            is handled separately'''
                                            #print("current has only 1 person")
                                            listOfQueues[0].append(getDiff(data["people"][0]["pose_keypoints"], previousData["people"][0]["pose_keypoints"]))
                                        else:
                                            #print("current has multiple people")
                                            for personIndex in range(len(listOfQueues)):
                                                if isFirstFrameForPerson[personIndex]:
                                                    '''Case: When this person was added
                                                    in previous frame so his queue is empty currently.'''
                                                    isFirstFrameForPerson[personIndex] = False
                                                else:
                                                    '''Only consider those people who were already
                                                    there in previous frame for getDiff'''
                                                    listOfQueues[personIndex].append(getDiff(data["people"][similarityData[str(frameNumber)][personIndex]]["pose_keypoints"], previousData["people"][similarityData[str(frameNumber - 1)][personIndex]]["pose_keypoints"]))

                                        '''Case 2: Number of people increased in current frame
                                        compared to previous frame'''
                                    elif len(data["people"]) > len(previousData["people"]):
                                        #print("Case 2: current > previous")

                                        if len(previousData["people"]) == 1:
                                            '''Case: Previous frame has 1 person and
                                            current frame has more people. Hence no similarity
                                            data available. So restart the queue.'''
                                            #print("current has only 1 person")
                                            listOfQueues = []
                                            isFirstFrame = True
                                            isFirstFrameForPerson = {}
                                        else:
                                            #print("current has multiple people")
                                            currentFrameAlreadyPresentIndices = []
                                            for previousPersonIndex in range(len(similarityData[str(frameNumber - 1)])):
                                                currentFrameAlreadyPresentIndices.append(similarityData[str(frameNumber)][previousPersonIndex])
                                                listOfQueues[previousPersonIndex].append(getDiff(data["people"][similarityData[str(frameNumber)][previousPersonIndex]]["pose_keypoints"], previousData["people"][similarityData[str(frameNumber - 1)][previousPersonIndex]]["pose_keypoints"]))
                                            currentFrameNewIndices = []
                                            for index in range(len(data["people"])):
                                                if index not in currentFrameAlreadyPresentIndices:
                                                    currentFrameNewIndices.append(index)
                                            for index in currentFrameNewIndices:
                                                listOfQueues.append([])
                                                isFirstFrameForPerson[len(listOfQueues) - 1] = True

                                        '''Case 3: Number of people decreased in current frame
                                        compared to previous frame'''
                                    else:
                                        #print("Case 3: current < previous")

                                        if len(data["people"]) == 1:
                                            #print("current has only 1 person")
                                            listOfQueues = []
                                            isFirstFrame = True
                                            isFirstFrameForPerson = {}
                                        else:
                                            #print("current has multiple people")

                                            for personIndex in range(len(similarityData[str(frameNumber)])):
                                                if similarityData[str(frameNumber)][personIndex] != -1:
                                                    listOfQueues[personIndex].append(getDiff(data["people"][similarityData[str(frameNumber)][personIndex]]["pose_keypoints"], previousData["people"][similarityData[str(frameNumber - 1)][personIndex]]["pose_keypoints"]))
                                            listOfQueuesCopy = []
                                            for personIndex in range(len(similarityData[str(frameNumber)])):
                                                if similarityData[str(frameNumber)][personIndex] != -1:
                                                    listOfQueuesCopy.append(listOfQueues[personIndex])
                                            listOfQueues = listOfQueuesCopy

                                    if len(listOfQueues) > 0:
                                        for personQueue in listOfQueues:
                                            #print("personQueue", len(personQueue))
                                            if len(personQueue) == (N - 1):
                                                #print("Queue full, appending to list")
                                                inputList.append(convertQueueToCoordinateDiffList(personQueue))
                                                if actionLabel == 'pushing2':
                                                    labelList.append(labelToOneHot['pushing'])
                                                elif actionLabel == 'snatching2':
                                                    labelList.append(labelToOneHot['snatching'])
                                                else:
                                                    labelList.append(labelToOneHot[actionLabel])
                                                personQueue.pop(0)
                                        #print("number of queues", len(listOfQueues))
                                        #print("Length of each queue")
                                        #for q in listOfQueues:
                                        #    print(len(q))
                            else:
                                #print(actionLabel, videoNum, frameNumber, " has no people")
                                '''
                                Since current frame has no person in it due to elimination,
                                hence start looking for N consecutive frames again
                                '''
                                isFirstFrame = True
                                isFirstFrameForPerson = {}
                                #queue = []
                                listOfQueues = []
                                #print("No person in current frame")
                                print(listOfQueues)
                            previousData = data
                        else:
                            ''' For classes other than pushing and snatching '''
                            if len(queue) == (N - 1):
                                inputList.append(convertQueueToCoordinateDiffList(queue))
                                labelList.append(labelToOneHot[actionLabel])
                                queue.pop(0)
                            if len(data["people"]) > 0:
                                if isFirstFrame:
                                    isFirstFrame = False
                                elif len(previousData["people"]) > 0:
                                    '''Append difference of coordinates between
                                    consecutive frames to the queue'''
                                    queue.append(getDiff(data["people"][0]["pose_keypoints"], previousData["people"][0]["pose_keypoints"]))
                            else:
                                #print(actionLabel, videoNum, frameNumber, " has no people")
                                '''
                                Since current frame has no person in it due to elimination,
                                hence start looking for N consecutive frames again
                                '''
                                isFirstFrame = True
                                queue = []
                            previousData = data
                    if actionLabel == 'pushing' or actionLabel == 'snatching':
                        if len(listOfQueues) > 0 and len(listOfQueues[0]) < (N - 1):
                            #print(str(N) + ' consecutive useful frames not present in ' + str(actionLabel) + ' ' + str(videoNum) + ' frames ' + str(startFrameNum) + ' to ' + str(endFrameNum))
                            ''' N consecutive frames not found in current (startFrameNum, endFrameNum)
                            frame range'''
                    elif len(queue) < (N - 1):
                        #print(str(N) + ' consecutive useful frames not present in ' + str(actionLabel) + ' ' + str(videoNum) + ' frames ' + str(startFrameNum) + ' to ' + str(endFrameNum))
                        ''' N consecutive frames not found in current (startFrameNum, endFrameNum)
                        frame range'''
        currentClassTuples = len(inputList) - previousSum
        tuplesOfClass[actionLabel] = currentClassTuples
        print(actionLabel, 'has', (len(inputList) - previousSum), 'tuples')
        if currentClassTuples < minClassTuples:
            minClassTuples = currentClassTuples
    print("Min class tuples", minClassTuples)
    balancedInputList = []
    balancedLabelList = []
    offset = 0
    for actionLabel in actionLabels:
        if actionLabel == 'pushing2' or actionLabel == 'snatching2':
            '''Just ignore since they are considered in their parent actions'''
        else:
            if actionLabel == 'pushing':
                indices = np.random.choice(tuplesOfClass[actionLabel] + tuplesOfClass['pushing2'], minClassTuples, replace=False)
                for index in indices:
                    balancedInputList.append(inputList[index + offset])
                    balancedLabelList.append(labelList[index + offset])
                offset = offset + tuplesOfClass[actionLabel] + tuplesOfClass['pushing2']
            elif actionLabel == 'snatching':
                indices = np.random.choice(tuplesOfClass[actionLabel] + tuplesOfClass['snatching2'], minClassTuples, replace=False)
                for index in indices:
                    balancedInputList.append(inputList[index + offset])
                    balancedLabelList.append(labelList[index + offset])
                offset = offset + tuplesOfClass[actionLabel] + tuplesOfClass['snatching2']
            else:
                indices = np.random.choice(tuplesOfClass[actionLabel], minClassTuples, replace=False)
                for index in indices:
                    balancedInputList.append(inputList[index + offset])
                    balancedLabelList.append(labelList[index + offset])
                offset = offset + tuplesOfClass[actionLabel]
    #inputNumpyArray = np.array(inputList)
    #labelNumpyArray = np.array(labelList)
    inputNumpyArray = np.array(balancedInputList)
    labelNumpyArray = np.array(balancedLabelList)
    print(inputNumpyArray)
    print(labelNumpyArray)
    print(inputNumpyArray.shape)
    print(labelNumpyArray.shape)
    return inputNumpyArray, labelNumpyArray

def printTrainTestClasses(y_train, y_test):
    eating_train = 0
    kicking_train = 0
    pushing_train = 0
    running_train = 0
    snatching_train = 0
    vaulting_train = 0
    walking_train = 0
    eating_test = 0
    kicking_test = 0
    pushing_test = 0
    running_test = 0
    snatching_test = 0
    vaulting_test = 0
    walking_test = 0
    train_max = np.argmax(y_train, axis=1)
    for index in np.nditer(train_max):
        if index == 0:
            eating_train = eating_train + 1
        elif index == 1:
            kicking_train = kicking_train + 1
        elif index == 2:
            pushing_train = pushing_train + 1
        elif index == 3:
            running_train = running_train + 1
        elif index == 4:
            snatching_train = snatching_train + 1
        elif index == 5:
            vaulting_train = vaulting_train + 1
        else:
            walking_train = walking_train + 1
    print("--- Training Data ---")
    print("Eating class has", eating_train, "tuples")
    print("Kicking class has", kicking_train, "tuples")
    print("Pushing class has", pushing_train, "tuples")
    print("Running class has", running_train, "tuples")
    print("Snatching class has", snatching_train, "tuples")
    print("Vaulting class has", vaulting_train, "tuples")
    print("Walking class has", walking_train, "tuples")
    test_max = np.argmax(y_test, axis=1)
    for index in np.nditer(test_max):
        if index == 0:
            eating_test = eating_test + 1
        elif index == 1:
            kicking_test = kicking_test + 1
        elif index == 2:
            pushing_test = pushing_test + 1
        elif index == 3:
            running_test = running_test + 1
        elif index == 4:
            snatching_test = snatching_test + 1
        elif index == 5:
            vaulting_test = vaulting_test + 1
        else:
            walking_test = walking_test + 1
    print("--- Testing Data ---")
    print("Eating class has", eating_test, "tuples")
    print("Kicking class has", kicking_test, "tuples")
    print("Pushing class has", pushing_test, "tuples")
    print("Running class has", running_test, "tuples")
    print("Snatching class has", snatching_test, "tuples")
    print("Vaulting class has", vaulting_test, "tuples")
    print("Walking class has", walking_test, "tuples")
    eating = eating_train + eating_test
    kicking = kicking_train + kicking_test
    pushing = pushing_train + pushing_test
    running = running_train + running_test
    snatching = snatching_train + snatching_test
    vaulting = vaulting_train + vaulting_test
    walking = walking_train + walking_test
    total = eating + kicking + pushing + running + snatching + vaulting + walking
    print("\neating", eating)
    print("kicking", kicking)
    print("pushing", pushing)
    print("running", running)
    print("snatching", snatching)
    print("vaulting", vaulting)
    print("walking", walking)
    print("\nTotal", total)

if __name__ == '__main__':
    inputNumpyArray, labelNumpyArray = createInputNumpyArray(12)
    print(inputNumpyArray.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(inputNumpyArray, labelNumpyArray, test_size = 0.3, random_state = 42, shuffle=True)

    printTrainTestClasses(y_train, y_test)
