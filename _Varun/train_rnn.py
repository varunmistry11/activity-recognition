import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
import os
import myConfigList
import random
import matplotlib.pyplot as plt
from sklearn import metrics

def readData(N):
    train = np.genfromtxt('balanced_raw_data/train_' + str(N) + '_tuples.csv', delimiter=',')
    test = np.genfromtxt('balanced_raw_data/test_'+str(N)+'_tuples.csv', delimiter=',')
    print('train shape', train.shape)
    print('test shape', test.shape)
    X = []
    y = []
    X1 = train[:,0:-1]
    print('X1 shape', X1.shape)
    y1 = train[:,-1]
    print('y1 shape', y1.shape)
    X2 = test[:,0:-1]
    print('X2 shape', X2.shape)
    y2 = test[:,-1]
    print('y2 shape', y2.shape)

    for _X in X1.tolist():
        X.append(_X)
    for _X in X2.tolist():
        X.append(_X)
    print('len(X)', len(X))
    print('len(X[0])', len(X[0]))

    classNumToOneHot = {
        1: [1, 0, 0, 0, 0, 0, 0],
        2: [0, 1, 0, 0, 0, 0, 0],
        3: [0, 0, 1, 0, 0, 0, 0],
        4: [0, 0, 0, 1, 0, 0, 0],
        5: [0, 0, 0, 0, 1, 0, 0],
        6: [0, 0, 0, 0, 0, 1, 0],
        7: [0, 0, 0, 0, 0, 0, 1]
    }

    for _y in y1.tolist():
        y.append(_y)
    for _y in y2.tolist():
        y.append(_y)
    print('len(y)', len(y))

    dataPerClass = {}
    for classNum in classNumToOneHot:
        dataPerClass[classNum] = []
    for index in range(len(X)):
        dataPerClass[y[index]].append(X[index])
    for classNum in dataPerClass:
        print(classNum, ':', len(dataPerClass[classNum]))

    totalClassSize = len(dataPerClass[1])
    validationSize = int(0.2 * totalClassSize)
    testingSize = int(0.2 * totalClassSize)
    trainingSize = totalClassSize - validationSize - testingSize
    print('total', totalClassSize)
    print('training', trainingSize)
    print('validation', validationSize)
    print('testing', testingSize)

    for classNum in dataPerClass:
        random.shuffle(dataPerClass[classNum])

    train_list = []
    validation_list = []
    testing_list = []

    for classNum in dataPerClass:
        for index in range(trainingSize):
            temp_list = []
            for val in dataPerClass[classNum][index]:
                temp_list.append(val)
            temp_list.append(classNumToOneHot[classNum])
            train_list.append(temp_list)

        for index in range(trainingSize, trainingSize + validationSize):
            temp_list = []
            for val in dataPerClass[classNum][index]:
                temp_list.append(val)
            temp_list.append(classNumToOneHot[classNum])
            validation_list.append(temp_list)

        for index in range(trainingSize + validationSize, len(dataPerClass[classNum])):
            temp_list = []
            for val in dataPerClass[classNum][index]:
                temp_list.append(val)
            temp_list.append(classNumToOneHot[classNum])
            testing_list.append(temp_list)

    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(testing_list)

    X_train_list = []
    X_validation_list = []
    X_test_list = []
    y_train_list = []
    y_validation_list = []
    y_test_list = []

    for l in train_list:
        X_train_list.append(l[0:-1])
        y_train_list.append(l[-1])
    for l in validation_list:
        X_validation_list.append(l[0:-1])
        y_validation_list.append(l[-1])
    for l in testing_list:
        X_test_list.append(l[0:-1])
        y_test_list.append(l[-1])

    X_train = np.array(X_train_list)
    X_validation = np.array(X_validation_list)
    X_test = np.array(X_test_list)
    y_train = np.array(y_train_list)
    y_validation = np.array(y_validation_list)
    y_test = np.array(y_test_list)
    '''
    train_X = train[:,0:-1]
    temp_train_y = train[:,-1]
    test_X = test[:,0:-1]
    temp_test_y = test[:,-1]


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
    '''
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def recurrentNeuralNetwork(rnn_size, n_classes, n_chunks, chunk_size, X):
    layer = {
        'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }
    print(X.shape)
    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, chunk_size])
    X = tf.split(0, n_chunks, X)

    #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    #outputs, states = rnn.rnn(lstm_cell, X, dtype=tf.float32)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

if __name__ == '__main__':
    N = 12
    n_epochs = 250
    n_classes = 7
    #batch_size = 128

    # Fill n_chunks of input to the lstm in sequence where each chunk has
    # size of chunk_size
    chunk_size = 13 * 2
    n_chunks = N

    rnn_size = 512

    X = tf.placeholder('float', [None, n_chunks, chunk_size], name="X")
    y = tf.placeholder('float', name="y")

    #train_X, train_y, test_X, test_y = readData(N)
    X_train, X_validation, X_test, y_train, y_validation, y_test = readData(N)
    print(X_train.shape)
    print(X_validation.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_validation.shape)
    print(y_test.shape)

    layer = {
        'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    X2 = tf.transpose(X, [1, 0, 2])
    X3 = tf.reshape(X2, [-1, chunk_size])
    #X = tf.split(0, n_chunks, X)
    X4 = tf.split(X3, n_chunks, 0)

    #########
    # Version 1
    #########
    '''
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, X4, dtype=tf.float32)
    '''

    #########
    # Version 2
    #########
    lstm_cell_1 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cells, X4, dtype=tf.float32)

    #########
    # Version 3
    #########
    '''
    lstm_cell_1 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_3 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3], state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cells, X4, dtype=tf.float32)
    '''

    #########
    # Version 4
    #########
    '''
    lstm_cell_1 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_3 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_4 = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3, lstm_cell_4], state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cells, X4, dtype=tf.float32)
    '''

    prediction = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name="prediction")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    lossList = []
    trainingAccuracyList = []
    validationAccuracyList = []

    #saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        X_train = X_train.reshape((X_train.shape[0], n_chunks, chunk_size))

        for epoch in range(n_epochs):
            epochLoss = 0
            _, loss = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})
            epochLoss += loss
            lossList.append(epochLoss)

            training_accuracy = accuracy.eval(session=sess, feed_dict={X: X_train, y: y_train})
            trainingAccuracyList.append(training_accuracy)
            validation_accuracy = accuracy.eval(session=sess, feed_dict={X: X_validation.reshape((X_validation.shape[0], n_chunks, chunk_size)), y: y_validation})
            validationAccuracyList.append(validation_accuracy)

            if epoch % 10 == 0:
                print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epochLoss)
                print('Training Accuracy:', training_accuracy, 'Validation Accuracy:', validation_accuracy)

        #save_path = saver.save(sess, "/home/varun/models/rnn_model_N_11")
        #print("Model saved in path: %s" % save_path)

        #testing_accuracy = accuracy.eval(session=sess, feed_dict={X: X_test.reshape((X_test.shape[0], n_chunks, chunk_size)), y: y_test})
        testing_one_hot_prediction, testing_loss, testing_accuracy = sess.run([prediction, cost, accuracy], feed_dict={X: X_test.reshape((X_test.shape[0], n_chunks, chunk_size)), y: y_test})
        print('\nTesting Accuracy after completion of training:', testing_accuracy)


    #plt.plot(lossList, label="epochloss")
    plt.plot(trainingAccuracyList, label="Training Accuracy")
    plt.plot(validationAccuracyList, label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc='lower right', shadow=True)
    plt.show()

    testing_prediction = testing_one_hot_prediction.argmax(1)
    y_test = y_test.argmax(1)
    precision = 100 * metrics.precision_score(y_test, testing_prediction, average="weighted")
    recall = 100 * metrics.recall_score(y_test, testing_prediction, average="weighted")
    f1_score = 100 * metrics.f1_score(y_test, testing_prediction, average="weighted")
    print('\nPrecision', precision)
    print('Recall', recall)
    print('f1_score', f1_score)

    print('\nConfusion matrix')
    confusion_matrix = metrics.confusion_matrix(y_test, testing_prediction)
    print(confusion_matrix)
    print(confusion_matrix.tolist())

    #print('\nNormalized confusion matrix')
    #normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    #print(normalised_confusion_matrix)


    with open('report/rnn/rnn_result_v2_N_' + str(N) + '_rnn_size_' + str(rnn_size) + '.json', 'w') as jsonfile:
        #new_data = dict()
        #new_data[config] = {'max_test_accuracy': float(maxTestAccuracy), 'average_test_accuracy': float(averageTestAccuracy)}
        #json.dump(new_data, jsonfile)
        data = dict()
        data[N] = {
            'training_accuracy': float(trainingAccuracyList[-1]),
            'validation_accuracy': float(validationAccuracyList[-1]),
            'testing_accuracy': float(testing_accuracy),
            'n_epochs': int(n_epochs),
            'rnn_size': int(rnn_size),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'confusion_matrix': confusion_matrix.tolist()
        }
        jsonfile.write(json.dumps(dict(data)))
