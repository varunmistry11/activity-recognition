import numpy as np
import json
import csv
from sklearn.model_selection import train_test_split

#actionLabels = ['eating', 'kicking', 'pushing', 'running', 'snatching', 'vaulting', 'walking']
actionLabels = ['eating', 'kicking', 'running', 'vaulting', 'walking']
videosPerActionLabel = {'eating' : 5,
                        'kicking' : 6,
                        'pushing' : 8,
                        'running' : 8,
                        'snatching' : 8,
                        'vaulting' : 8,
                        'walking' : 16}
'''
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
            for row in reader:
                if firstRow :
                    firstRow = False
                else :
                    startFrameNum = int(row[0])
                    endFrameNum = int(row[1])
                    queue = []
                    previousData = {}
                    isFirstFrame = True
                    for frameNumber in range(startFrameNum, endFrameNum + 1):
                        data = getJSONData(actionLabel, videoNum, frameNumber)
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
                    if len(queue) < (N - 1):
                        #print(str(N) + ' consecutive useful frames not present in ' + str(actionLabel) + ' ' + str(videoNum) + ' frames ' + str(startFrameNum) + ' to ' + str(endFrameNum))
                        ''' N consecutive frames not found in current (startFrameNum, endFrameNum)
                        frame range'''
                        #return
        currentClassTuples = len(inputList) - previousSum
        tuplesOfClass[actionLabel] = currentClassTuples
        print(actionLabel, 'has', (len(inputList) - previousSum), 'tuples')
        if currentClassTuples < minClassTuples:
            minClassTuples = currentClassTuples
    print("\n\nMinimum tuples in any class =", minClassTuples)
    print("Hence selecting only", minClassTuples, "tuples from each class randomly")
    balancedInputList = []
    balancedLabelList = []
    offset = 0
    for actionLabel in actionLabels:
        indices = np.random.choice(tuplesOfClass[actionLabel], minClassTuples, replace=False)
        for index in indices:
            balancedInputList.append(inputList[index + offset])
            balancedLabelList.append(labelList[index + offset])
        offset = offset + tuplesOfClass[actionLabel]
    #inputNumpyArray = np.array(inputList)
    #labelNumpyArray = np.array(labelList)
    inputNumpyArray = np.array(balancedInputList)
    labelNumpyArray = np.array(balancedLabelList)
    print('\n')
    print(inputNumpyArray)
    print(labelNumpyArray)
    print(inputNumpyArray.shape)
    print(labelNumpyArray.shape)
    return inputNumpyArray, labelNumpyArray

def printTrainTestClasses(y_train, y_test):
    eating_train = 0
    kicking_train = 0
    running_train = 0
    vaulting_train = 0
    walking_train = 0
    eating_test = 0
    kicking_test = 0
    running_test = 0
    vaulting_test = 0
    walking_test = 0
    train_max = np.argmax(y_train, axis=1)
    for index in np.nditer(train_max):
        if index == 0:
            eating_train = eating_train + 1
        elif index == 1:
            kicking_train = kicking_train + 1
        elif index == 2:
            running_train = running_train + 1
        elif index == 3:
            vaulting_train = vaulting_train + 1
        else:
            walking_train = walking_train + 1
    print("--- Training Data ---")
    print("Eating class has", eating_train, "tuples")
    print("Kicking class has", kicking_train, "tuples")
    print("Running class has", running_train, "tuples")
    print("Vaulting class has", vaulting_train, "tuples")
    print("Walking class has", walking_train, "tuples")
    test_max = np.argmax(y_test, axis=1)
    for index in np.nditer(test_max):
        if index == 0:
            eating_test = eating_test + 1
        elif index == 1:
            kicking_test = kicking_test + 1
        elif index == 2:
            running_test = running_test + 1
        elif index == 3:
            vaulting_test = vaulting_test + 1
        else:
            walking_test = walking_test + 1
    print("--- Testing Data ---")
    print("Eating class has", eating_test, "tuples")
    print("Kicking class has", kicking_test, "tuples")
    print("Running class has", running_test, "tuples")
    print("Vaulting class has", vaulting_test, "tuples")
    print("Walking class has", walking_test, "tuples")
    eating = eating_train + eating_test
    kicking = kicking_train + kicking_test
    running = running_train + running_test
    vaulting = vaulting_train + vaulting_test
    walking = walking_train + walking_test
    total = eating + kicking + running + vaulting + walking
    print("\nTotal eating =", eating)
    print("Total kicking =", kicking)
    print("Total running =", running)
    print("Total vaulting =", vaulting)
    print("Total walking =", walking)
    print("\nTotal =", total)

if __name__ == '__main__':
    inputNumpyArray, labelNumpyArray = createInputNumpyArray(10)
    print(inputNumpyArray.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(inputNumpyArray, labelNumpyArray, test_size = 0.3, random_state = 42, shuffle=True)

    printTrainTestClasses(y_train, y_test)
