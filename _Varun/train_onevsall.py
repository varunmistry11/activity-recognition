import numpy as np
import tensorflow as tf
import json
import os
import myConfigList
#import prepare_unbalanced_data_csv as unbalanced

classNumToLabel = {
    1: 'eating',
    2: 'kicking',
    3: 'pushing',
    4: 'running',
    5: 'snatching',
    6: 'vaulting',
    7: 'walking'
}

def getBalancedOneVSAllList(classNum, X, y):
    numTuplesPerClass = {}
    tuplesToConsider = float("inf")
    for a_class in classNumToLabel:
        numTuplesPerClass[a_class] = 0
    for y_val in y:
        numTuplesPerClass[y_val] = numTuplesPerClass[y_val] + 1
    for a_class in numTuplesPerClass:
        print(a_class, numTuplesPerClass[a_class])
    currentClassTuples = numTuplesPerClass[classNum]
    for anotherClass in classNumToLabel:
        if anotherClass != classNum:
            if numTuplesPerClass[anotherClass] > (currentClassTuples / (len(classNumToLabel) - 1)):
                if int(currentClassTuples / (len(classNumToLabel) - 1)) < tuplesToConsider:
                    tuplesToConsider = int(currentClassTuples / (len(classNumToLabel) - 1))
            else:
                if numTuplesPerClass[anotherClass] < tuplesToConsider:
                    tuplesToConsider = numTuplesPerClass[anotherClass]
    print('currentClassTuples', currentClassTuples)
    print('tuplesToConsider', tuplesToConsider)
    total = tuplesToConsider * (len(classNumToLabel) - 1)
    print('total', total)

    inputTuplesPerClass = {}
    for Class in classNumToLabel:
        inputTuplesPerClass[Class] = []
    for index in range(len(X)):
        inputTuplesPerClass[y[index]].append(X[index])

    balancedInputList = []
    balancedOutputList = []
    for Class in classNumToLabel:
        if Class == classNum:
            indices = np.random.choice(numTuplesPerClass[Class], total, replace=False)
            for index in indices:
                balancedInputList.append(inputTuplesPerClass[Class][index])
                balancedOutputList.append([1, 0])
        else:
            indices = np.random.choice(numTuplesPerClass[Class], tuplesToConsider, replace=False)
            for index in indices:
                balancedInputList.append(inputTuplesPerClass[Class][index])
                balancedOutputList.append([0, 1])

    return balancedInputList, balancedOutputList

def readData(N, classNum):
    train = np.genfromtxt('unbalanced_data/train_' + str(N) + '_tuples.csv', delimiter=',')
    test = np.genfromtxt('unbalanced_data/test_'+str(N)+'_tuples.csv', delimiter=',')
    temp_train_X = train[:,0:-1]
    temp_train_y = train[:,-1]
    temp_test_X = test[:,0:-1]
    temp_test_y = test[:,-1]

    train_y_list = temp_train_y.tolist()
    train_X_list = temp_train_X.tolist()
    train_X2, train_y2 = getBalancedOneVSAllList(classNum, train_X_list, train_y_list)
    '''
    count = 0
    for y in train_y2:
        if y[0] == 1:
            count = count + 1
    print('same class tuples', count)
    print('other class tuples', len(train_y2) - count)
    '''
    train_X = np.array(train_X2)
    train_y = np.array(train_y2)

    test_y_list = temp_test_y.tolist()
    test_X_list = temp_test_X.tolist()
    test_X2, test_y2 = getBalancedOneVSAllList(classNum, test_X_list, test_y_list)
    '''
    count = 0
    for y in test_y2:
        if y[0] == 1:
            count = count + 1
    print('same class tuples', count)
    print('other class tuples', len(test_y2) - count)
    '''
    test_X = np.array(test_X2)
    test_y = np.array(test_y2)

    return train_X, train_y, test_X, test_y

def createNNModel(numOfNodes_input, listOfHiddenNodes, numOfNodes_output):
    hiddenLayers = {}

    for layerNumber in range(1, len(listOfHiddenNodes) + 1):
        if layerNumber == 1:
            hiddenLayers[layerNumber] = {
                'weights': tf.Variable(tf.random_normal([numOfNodes_input, listOfHiddenNodes[layerNumber - 1]])),
                'biases': tf.Variable(tf.random_normal([listOfHiddenNodes[layerNumber - 1]]))
            }
        else:
            hiddenLayers[layerNumber] = {
                'weights': tf.Variable(tf.random_normal([listOfHiddenNodes[layerNumber - 2], listOfHiddenNodes[layerNumber - 1]])),
                'biases': tf.Variable(tf.random_normal([listOfHiddenNodes[layerNumber - 1]]))
            }

    outputLayer = {
        'weights': tf.Variable(tf.random_normal([listOfHiddenNodes[-1], numOfNodes_output])),
        'biases': tf.Variable(tf.random_normal([numOfNodes_output]))
    }

    for layerNumber in range(1, len(listOfHiddenNodes) + 1):
        if layerNumber == 1:
            # (input * weights) + biases
            hiddenLayer = tf.add(tf.matmul(X, hiddenLayers[layerNumber]['weights']), hiddenLayers[layerNumber]['biases'])
            hiddenLayer = tf.nn.relu(hiddenLayer)
            #hiddenLayer = tf.sigmoid(hiddenLayer)
            #hiddenLayer = tf.nn.dropout(hiddenLayer, 0.25)
        else:
            # (input * weights) + biases
            hiddenLayer = tf.add(tf.matmul(previousLayer, hiddenLayers[layerNumber]['weights']), hiddenLayers[layerNumber]['biases'])
            hiddenLayer = tf.nn.relu(hiddenLayer)
            #hiddenLayer = tf.sigmoid(hiddenLayer)
            #hiddenLayer = tf.nn.dropout(hiddenLayer, 0.25)
        previousLayer = hiddenLayer

    prediction = tf.add(tf.matmul(previousLayer, outputLayer['weights']), outputLayer['biases'])

    return prediction

def trainNNModel(sess, prediction, train_X, train_y, test_X, test_y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    numOfEpochs = 3500
    #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    maxTestAccuracy = 0
    sumAccuracy = 0
    numOfEpochsCompleted = 0
    lastTenErrors = []

    for epoch in range(numOfEpochs):
        _, epochLoss = sess.run([optimizer, cost], feed_dict={X: train_X, y: train_y})

        if len(lastTenErrors) == 10:
            lastTenErrors.pop(0)
        lastTenErrors.append(epochLoss)

        if len(lastTenErrors) == 10:
            error = 0
            for e in lastTenErrors:
                error = error + e
            averageError = error / 10
            #print('Average Error in last 10 epochs', averageError)
            if averageError < 100:
                break

        test_accuracy = accuracy.eval(session=sess, feed_dict={X: test_X, y: test_y})
        sumAccuracy = sumAccuracy + test_accuracy
        if test_accuracy > maxTestAccuracy:
            maxTestAccuracy = test_accuracy
        if epoch % 10 == 0:
            print('Epoch', epoch, 'completed out of', numOfEpochs, 'loss:', epochLoss)
        if epoch % 100 == 0:
            print('Training Accuracy:', accuracy.eval(session=sess, feed_dict={X: train_X, y: train_y}), 'Testing Accuracy:', test_accuracy)
        numOfEpochsCompleted = numOfEpochsCompleted + 1
    print('Testing Accuracy after completion of training:', accuracy.eval(session=sess, feed_dict={X: test_X, y: test_y}))
    print('Max testing accuracy:', maxTestAccuracy)
    averageTestAccuracy = sumAccuracy / numOfEpochsCompleted
    print('Average testing accuracy:', averageTestAccuracy)
    #saveResults(10, {'100-100-100-100-100': {'max_test_accuracy': maxTestAccuracy, 'average_test_accuracy' : averageTestAccuracy}})
    #saveResults(10, '100-100-100-100-100', maxTestAccuracy, averageTestAccuracy)
    return maxTestAccuracy, averageTestAccuracy

def getConfigString(config):
    configString = ''
    for index in range(len(config)):
        if index == 0:
            configString = configString + str(config[index])
        else:
            configString = configString + '-' + str(config[index])
    return configString

def saveResults(N, classNum, config, maxTestAccuracy, averageTestAccuracy):
    with open('unbalanced_data/onevsall_results/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + config + '.json', 'w') as jsonfile:
        new_data = dict()
        new_data[config] = {'max_test_accuracy': float(maxTestAccuracy), 'average_test_accuracy': float(averageTestAccuracy)}
        #json.dump(new_data, jsonfile)
        jsonfile.write(json.dumps(dict(new_data)))

def getAllPossibleConfigList(maxLayers, minNodesInLayer, maxNodesInLayer, nodeIncrement):
    '''
    allConfigsPossible = []
    configList = []
    for numOfLayers in range(1, maxLayers + 1):
        print('numOfLayers', numOfLayers)
        if numOfLayers == 1:
            for numOfNodes in range(minNodesInLayer, maxNodesInLayer + 1, nodeIncrement):
                tempList = []
                tempList.append(numOfNodes)
                configList.append(tempList)
                allConfigsPossible.append(tempList)
            print(len(configList))
        else:
            configList = []
            for numOfNodes in range(minNodesInLayer, maxNodesInLayer + 1, nodeIncrement):
                for previousConfig in previousConfigList:
                    tempList = []
                    tempList.append(numOfNodes)
                    for nodes in previousConfig:
                        tempList.append(nodes)
                    configList.append(tempList)
                    allConfigsPossible.append(tempList)
            print(len(configList))
        previousConfigList = configList
    print(len(allConfigsPossible))
    return allConfigsPossible
    '''
    return myConfigList.myConfigurations


if __name__ == '__main__':
    '''
    train_X, train_y, test_X, test_y = readData(15, 4)

    minLayers = 1
    maxLayers = 6
    #minNodesInLayer = 50
    minNodesInLayer = 100
    maxNodesInLayer = 500
    #nodeIncrement = 50
    nodeIncrement = 100
    allConfigsPossible = getAllPossibleConfigList(maxLayers, minNodesInLayer, maxNodesInLayer, nodeIncrement)
    '''
    '''
    for N in range(5, 16, 5):
        print('N = ', N)
        for classNum in classNumToLabel:
            print('---For class', classNumToLabel[classNum], '----')
            train_X, train_y, test_X, test_y = readData(N, classNum)
            print(train_X.shape)
            print(train_y.shape)
            print(test_X.shape)
            print(test_y.shape)

            numOfClasses = 2

            X = tf.placeholder('float', [None, train_X.shape[1]])
            y = tf.placeholder('float', [None, numOfClasses])
            sess = tf.Session()

            for config in allConfigsPossible:
                print('\n\nN = ', N, 'class = ', classNumToLabel[classNum], 'config', config, '\n\n')
                if not os.path.isfile('./unbalanced_data/onevsall_results/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'):
                    prediction = createNNModel(train_X.shape[1], config, numOfClasses)
                    maxTestAccuracy, averageTestAccuracy = trainNNModel(sess, prediction, train_X, train_y, test_X, test_y)
                    saveResults(N, classNum, getConfigString(config), maxTestAccuracy, averageTestAccuracy)
    '''

    N = 5
    #for classNum in classNumToLabel:
    classNum = 3
    print('---For class', classNumToLabel[classNum], '----')
    train_X, train_y, test_X, test_y = readData(N, classNum)
    numOfClasses = 2

    X = tf.placeholder('float', [None, train_X.shape[1]])
    y = tf.placeholder('float', [None, numOfClasses])
    sess = tf.Session()

    numOfNodes = 100
    for numOfLayers in range(1, 16):
        print('\n\nN =', N, 'numOfLayers =', numOfLayers)
        config = []
        for _ in range(1, numOfLayers + 1):
            config.append(numOfNodes)

        if not os.path.isfile('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'):
            prediction = createNNModel(train_X.shape[1], config, numOfClasses)
            maxTestAccuracy, averageTestAccuracy = trainNNModel(sess, prediction, train_X, train_y, test_X, test_y)
            #saveResults(N, classNum, getConfigString(config), maxTestAccuracy, averageTestAccuracy)
            with open('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json', 'w') as jsonfile:
                new_data = dict()
                new_data[getConfigString(config)] = {'max_test_accuracy': float(maxTestAccuracy), 'average_test_accuracy': float(averageTestAccuracy)}
                #json.dump(new_data, jsonfile)
                jsonfile.write(json.dumps(dict(new_data)))
    '''
    N = 5
    classNum = 3
    print('---For class', classNumToLabel[classNum], '----')
    train_X, train_y, test_X, test_y = readData(N, classNum)
    numOfClasses = 2

    X = tf.placeholder('float', [None, train_X.shape[1]])
    y = tf.placeholder('float', [None, numOfClasses])
    sess = tf.Session()

    for numOfNodes in [10, 50, 100, 200, 300, 400, 500, 1000]:
        #numOfNodes = 100
        #for numOfLayers in range(1, 16):
        numOfLayers = 7
        print('\n\nN =', N, 'numOfLayers =', numOfLayers)
        config = []
        for _ in range(1, numOfLayers + 1):
            config.append(numOfNodes)

        if not os.path.isfile('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'):
            prediction = createNNModel(train_X.shape[1], config, numOfClasses)
            maxTestAccuracy, averageTestAccuracy = trainNNModel(sess, prediction, train_X, train_y, test_X, test_y)
            #saveResults(N, classNum, getConfigString(config), maxTestAccuracy, averageTestAccuracy)
            with open('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json', 'w') as jsonfile:
                new_data = dict()
                new_data[getConfigString(config)] = {'max_test_accuracy': float(maxTestAccuracy), 'average_test_accuracy': float(averageTestAccuracy)}
                #json.dump(new_data, jsonfile)
                jsonfile.write(json.dumps(dict(new_data)))
    '''
    '''
    for N in range(15, 4, -1):
        print('N = ', N)
        for classNum in classNumToLabel:
            print('---For class', classNumToLabel[classNum], '----')
            train_X, train_y, test_X, test_y = readData(N, classNum)
            print(train_X.shape)
            print(train_y.shape)
            print(test_X.shape)
            print(test_y.shape)

            numOfClasses = 2

            X = tf.placeholder('float', [None, train_X.shape[1]])
            y = tf.placeholder('float', [None, numOfClasses])
            sess = tf.Session()

            for config in allConfigsPossible:
                print('\n\nN = ', N, 'class = ', classNumToLabel[classNum], 'config', config, '\n\n')
                if not os.path.isfile('./unbalanced_data/onevsall_results/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'):
                    prediction = createNNModel(train_X.shape[1], config, numOfClasses)
                    maxTestAccuracy, averageTestAccuracy = trainNNModel(sess, prediction, train_X, train_y, test_X, test_y)
                    saveResults(N, classNum, getConfigString(config), maxTestAccuracy, averageTestAccuracy)
    #trainAllNNModels()
    '''
    '''
    with open('balanced_training_result.json') as jsonfile:
        old_data = json.load(jsonfile)
    old_data.update({'100-100-100': {'max_test_accuracy': 0.1, 'average_test_accuracy': 0.1}})
    '''
    '''
    for N in range(3, 16):
        with open('balanced_training_result_N_' + str(N) + '.json', 'w') as jsonfile:
            old_data = dict()
            old_data.update({'100-100-100': {'max_test_accuracy': 0.1, 'average_test_accuracy': 0.1}})
            jsonfile.write(json.dumps(old_data))
    '''
    '''
    train_X, train_y, test_X, test_y = readData(10)
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)
    numOfClasses = 7
    #Written outside since they have to be accessed from botn createNNModel() and trainNNModel()
    X = tf.placeholder('float', [None, train_X.shape[1]])
    y = tf.placeholder('float', [None, numOfClasses])
    prediction = createNNModel(train_X.shape[1], [100, 500, 500, 500, 100], numOfClasses)
    trainNNModel(prediction, train_X, train_y, test_X, test_y)
    '''
