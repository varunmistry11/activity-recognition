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

def NeuralNetworkModel(data, numOfNodes_input):
    hiddenLayer_1 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_input, numOfNodes_hl1])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl1]))
    }
    hiddenLayer_2 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl1, numOfNodes_hl2])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl2]))
    }
    hiddenLayer_3 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl2, numOfNodes_hl3])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl3]))
    }
    outputLayer = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl3, numOfClasses])),
        'biases': tf.Variable(tf.random_normal([numOfClasses]))
    }

    # (input * weights) + biases
    layer1 = tf.add(tf.matmul(data, hiddenLayer_1['weights']), hiddenLayer_1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3, outputLayer['weights']), outputLayer['biases'])

    return output

def trainNeuralNetwork(data):
    prediction = NeuralNetworkModel(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    numOfEpochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(numOfEpochs):
            epochLoss = 0
            #X, y =
            _, c = sess.run([optimizer, cost], feed_dict={x: X, y: y})
            epochLoss += c
            print('Epoch', epoch, 'completed out of', numOfEpochs, 'loss:', epochLoss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))

if __name__ == '__main__':
    inputNumpyArray, labelNumpyArray = createInputNumpyArray(10)
    print(inputNumpyArray.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(inputNumpyArray, labelNumpyArray, test_size = 0.3, random_state = 42, shuffle=True)

    printTrainTestClasses(y_train, y_test)


    numOfNodes_hl1 = 100
    numOfNodes_hl2 = 200
    numOfNodes_hl3 = 500
    numOfNodes_hl4 = 1000
    numOfNodes_hl5 = 1500
    numOfNodes_hl6 = 1500
    numOfNodes_hl7 = 2500
    numOfNodes_hl8 = 1500
    numOfNodes_hl9 = 1000
    numOfNodes_hl10 = 800
    numOfNodes_hl11 = 500
    numOfNodes_hl12 = 200
    numOfClasses = 7
    batchSize = 100

    # size is given as [height, width]
    X = tf.placeholder('float', [None, X_train.shape[1]])
    y = tf.placeholder('float', [None, numOfClasses])

    keepProbability = tf.placeholder(tf.float32)

    numOfNodes_input = inputNumpyArray.shape[1]

    hiddenLayer_1 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_input, numOfNodes_hl1])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl1]))
    }
    hiddenLayer_2 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl1, numOfNodes_hl2])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl2]))
    }
    hiddenLayer_3 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl2, numOfNodes_hl3])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl3]))
    }
    hiddenLayer_4 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl3, numOfNodes_hl4])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl4]))
    }
    hiddenLayer_5 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl4, numOfNodes_hl5])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl5]))
    }
    hiddenLayer_6 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl5, numOfNodes_hl6])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl6]))
    }
    hiddenLayer_7 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl6, numOfNodes_hl7])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl7]))
    }
    hiddenLayer_8 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl7, numOfNodes_hl8])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl8]))
    }
    hiddenLayer_9 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl8, numOfNodes_hl9])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl9]))
    }
    hiddenLayer_10 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl9, numOfNodes_hl10])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl10]))
    }
    hiddenLayer_11 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl10, numOfNodes_hl11])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl11]))
    }
    hiddenLayer_12 = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl11, numOfNodes_hl12])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_hl12]))
    }
    outputLayer = {
        'weights': tf.Variable(tf.random_normal([numOfNodes_hl12, numOfClasses])),
        'biases': tf.Variable(tf.random_normal([numOfClasses]))
    }

    # (input * weights) + biases
    layer1 = tf.add(tf.matmul(X, hiddenLayer_1['weights']), hiddenLayer_1['biases'])
    layer1 = tf.nn.relu(layer1)
    #layer1 = tf.sigmoid(layer1)
    #layer1 = tf.nn.dropout(layer1, 0.25)

    layer2 = tf.add(tf.matmul(layer1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
    layer2 = tf.nn.relu(layer2)
    #layer2 = tf.sigmoid(layer2)
    #layer2 = tf.nn.dropout(layer2, 0.25)

    layer3 = tf.add(tf.matmul(layer2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
    layer3 = tf.nn.relu(layer3)
    #layer3 = tf.sigmoid(layer3)
    #layer3 = tf.nn.dropout(layer3, 0.25)

    layer4 = tf.add(tf.matmul(layer3, hiddenLayer_4['weights']), hiddenLayer_4['biases'])
    layer4 = tf.nn.relu(layer4)
    #layer4 = tf.sigmoid(layer4)
    #layer4 = tf.nn.dropout(layer4, 0.25)

    layer5 = tf.add(tf.matmul(layer4, hiddenLayer_5['weights']), hiddenLayer_5['biases'])
    layer5 = tf.nn.relu(layer5)
    #layer5 = tf.sigmoid(layer5)
    #layer5 = tf.nn.dropout(layer5, 0.25)

    layer6 = tf.add(tf.matmul(layer5, hiddenLayer_6['weights']), hiddenLayer_6['biases'])
    layer6 = tf.nn.relu(layer6)
    #layer6 = tf.sigmoid(layer6)
    #layer6 = tf.nn.dropout(layer6, 0.25)

    layer7 = tf.add(tf.matmul(layer6, hiddenLayer_7['weights']), hiddenLayer_7['biases'])
    layer7 = tf.nn.relu(layer7)
    #layer7 = tf.sigmoid(layer7)
    #layer7 = tf.nn.dropout(layer7, 0.25)

    layer8 = tf.add(tf.matmul(layer7, hiddenLayer_8['weights']), hiddenLayer_8['biases'])
    layer8 = tf.nn.relu(layer8)
    #layer8 = tf.sigmoid(layer8)
    #layer8 = tf.nn.dropout(layer8, 0.25)

    layer9 = tf.add(tf.matmul(layer8, hiddenLayer_9['weights']), hiddenLayer_9['biases'])
    layer9 = tf.nn.relu(layer9)
    #layer9 = tf.sigmoid(layer9)
    #layer9 = tf.nn.dropout(layer9, 0.25)

    layer10 = tf.add(tf.matmul(layer9, hiddenLayer_10['weights']), hiddenLayer_10['biases'])
    layer10 = tf.nn.relu(layer10)
    #layer10 = tf.sigmoid(layer10)
    #layer10 = tf.nn.dropout(layer10, 0.25)

    layer11 = tf.add(tf.matmul(layer10, hiddenLayer_11['weights']), hiddenLayer_11['biases'])
    layer11 = tf.nn.relu(layer11)
    #layer11 = tf.sigmoid(layer11)
    #layer11 = tf.nn.dropout(layer11, 0.25)

    layer12 = tf.add(tf.matmul(layer11, hiddenLayer_12['weights']), hiddenLayer_12['biases'])
    layer12 = tf.nn.relu(layer12)
    #layer12 = tf.sigmoid(layer12)
    #layer12 = tf.nn.dropout(layer12, 0.25)

    prediction = tf.add(tf.matmul(layer12, outputLayer['weights']), outputLayer['biases'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    numOfEpochs = 5000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(numOfEpochs):
            _, epochLoss = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})
            if epoch % 10 == 0:
                print('Epoch', epoch, 'completed out of', numOfEpochs, 'loss:', epochLoss)
            if epoch % 100 == 0:
                print('Accuracy:', accuracy.eval({X: X_test, y: y_test}))

        print('Accuracy:', accuracy.eval({X: X_test, y: y_test}))

    '''
    sess = tf.Session()
    labels = tf.placeholder(tf.float32)
    predictions = tf.placeholder(tf.float32)
    acc, _ = tf.metrics.accuracy(labels, predictions)
    max_label = tf.argmax(labels, 1)
    max_prediction = tf.argmax(predictions, 1)
    eq = tf.equal(max_label, max_prediction)
    cast = tf.cast(eq, tf.float32)
    my_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))

    feed_dict = {
        labels: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        predictions: [[1, 2, 3, 4, 5.1], [2, 5, 4, 3, 1]]
    }
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print(sess.run(acc, feed_dict))  # 0.0
    print(sess.run(max_label, feed_dict))
    print(sess.run(max_prediction, feed_dict))
    print(sess.run(eq, feed_dict))
    print(sess.run(cast, feed_dict))
    print(sess.run(my_acc, feed_dict))  # 1.0
    '''
