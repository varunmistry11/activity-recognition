import numpy as np
import tensorflow as tf
import json
import os
import myConfigList

def readData(N):
    train = np.genfromtxt('balanced_data/train_' + str(N) + '_tuples.csv', delimiter=',')
    test = np.genfromtxt('balanced_data/test_'+str(N)+'_tuples.csv', delimiter=',')
    train_X = train[:,0:-1]
    temp_train_y = train[:,-1]
    test_X = test[:,0:-1]
    temp_test_y = test[:,-1]

    classNumToOneHot = {
        1: [1, 0, 0, 0, 0, 0, 0],
        2: [0, 1, 0, 0, 0, 0, 0],
        3: [0, 0, 1, 0, 0, 0, 0],
        4: [0, 0, 0, 1, 0, 0, 0],
        5: [0, 0, 0, 0, 1, 0, 0],
        6: [0, 0, 0, 0, 0, 1, 0],
        7: [0, 0, 0, 0, 0, 0, 1]
    }

    train_y_list = temp_train_y.tolist()
    temp_list = []
    for y in train_y_list:
        temp_list.append(classNumToOneHot[y])
    train_y = np.array(temp_list)

    test_y_list = temp_test_y.tolist()
    temp_list = []
    for y in test_y_list:
        temp_list.append(classNumToOneHot[y])
    test_y = np.array(temp_list)

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

    numOfEpochs = 5000
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

def saveResults(N, config, maxTestAccuracy, averageTestAccuracy):
    '''
    with open('balanced_training_result_N_' + str(N) + '.json') as jsonfile:
        old_data = json.loads(jsonfile.read())
    with open('balanced_training_result_N_' + str(N) + '.json', 'w') as jsonfile:
        old_data[config] = {'max_test_accuracy': float(maxTestAccuracy), 'average_test_accuracy': float(averageTestAccuracy)}
        #old_data[config] = [maxTestAccuracy, averageTestAccuracy]
        #json.dump(old_data, jsonfile)
        jsonfile.write(json.dumps(dict(old_data)))
    '''
    with open('balanced_data/results/b_N_' + str(N) + '_NN_' + config + '.json', 'w') as jsonfile:
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

def trainAllNNModels():
    minLayers = 1
    maxLayers = 3
    minNodesInLayer = 50
    maxNodesInLayer = 200
    nodeIncrement = 50
    allConfigsPossible = getAllPossibleConfigList(maxLayers, minNodesInLayer, maxNodesInLayer, nodeIncrement)
    for N in range(3, 4):
        print('N = ', N)
        train_X, train_y, test_X, test_y = readData(N)
        print(train_X.shape)
        print(train_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        numOfClasses = 7

        for config in allConfigsPossible:
            if not os.path.isfile('./balanced_data/results/b_N_' + str(N) + '_NN_' + getConfigString(config) + '.json'):
                prediction = createNNModel(train_X.shape[1], config, numOfClasses)
                maxTestAccuracy, averageTestAccuracy = trainNNModel(prediction, train_X, train_y, test_X, test_y)
                saveResults(N, getConfigString(config), maxTestAccuracy, averageTestAccuracy)

if __name__ == '__main__':

    minLayers = 1
    maxLayers = 6
    #minNodesInLayer = 50
    minNodesInLayer = 100
    maxNodesInLayer = 500
    #nodeIncrement = 50
    nodeIncrement = 100
    allConfigsPossible = getAllPossibleConfigList(maxLayers, minNodesInLayer, maxNodesInLayer, nodeIncrement)
    for N in range(3, 16):
        print('N = ', N)
        train_X, train_y, test_X, test_y = readData(N)
        print(train_X.shape)
        print(train_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        numOfClasses = 7

        X = tf.placeholder('float', [None, train_X.shape[1]])
        y = tf.placeholder('float', [None, numOfClasses])
        sess = tf.Session()

        for config in allConfigsPossible:
            print('\n\nN = ', N, 'config', config, '\n\n')
            if not os.path.isfile('./balanced_data/results/b_N_' + str(N) + '_NN_' + getConfigString(config) + '.json'):
                prediction = createNNModel(train_X.shape[1], config, numOfClasses)
                maxTestAccuracy, averageTestAccuracy = trainNNModel(sess, prediction, train_X, train_y, test_X, test_y)
                saveResults(N, getConfigString(config), maxTestAccuracy, averageTestAccuracy)
    #trainAllNNModels()

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
    '''
    with open('test.json') as jsonfile:
        old_data = json.load(jsonfile)
    data = {
        100: {
            50: 0.4
        },
        200: {
            10: 0.9
        }
    }
    old_data.update(data)
    with open('test.json', 'w') as jsonfile:
        json.dump(old_data, jsonfile)
    '''
    '''
    data = {
        '10': {
            '100': {
                'accuracy' : 15,
                '50': {'accuracy' : 20}
            },
            '200': {'accuracy' : 200}
        },
        '12': {
            '100': {
                'accuracy' : 15,
                '50': {'accuracy' : 20}
            },
            '200': {'accuracy' : 200}
        }
    }
    temp = {
    '100-500-100': 0.6,
    '100-500-200': 0.7,
    '100-100-500-100': 0.9
    }
    print(temp)
    for key in temp:
        print(key)
    temp['100-500-100'] = 2018
    for key in temp:
        print(key)
    '''
