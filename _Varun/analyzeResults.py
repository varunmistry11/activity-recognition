import json
import myConfigList
import csv
import matplotlib.pyplot as plt

classNumToLabel = {
    1 : 'eating',
    2 : 'kicking',
    3 : 'pushing',
    4 : 'running',
    5 : 'snatching',
    6 : 'vaulting',
    7 : 'walking'
}

def getConfigString(config):
    configString = ''
    for index in range(len(config)):
        if index == 0:
            configString = configString + str(config[index])
        else:
            configString = configString + '-' + str(config[index])
    return configString

def getAllBalancedNNResults():
    results = {}
    for N in range(5, 16):
        for config in myConfigList.myConfigurations:
            data = json.load(open('./balanced_data/results/b_N_' + str(N) + '_NN_' + getConfigString(config) + '.json'))
            if str(N) not in results:
                results[str(N)] = {}
            results[str(N)][getConfigString(config)] = data[getConfigString(config)]
    return results

def printAllBalancedNNResults():
    results = getAllBalancedNNResults()
    maxTestAccuracy = 0
    for N in range(5, 16):
        for config in results[str(N)]:
            #print(config, results[str(N)][config]['max_test_accuracy'])
            if results[str(N)][config]['max_test_accuracy'] > maxTestAccuracy:
                maxTestAccuracy = results[str(N)][config]['max_test_accuracy']
                maxConfig = config
                maxN = N
    print('---- Balanced NN Result highlight ----')
    print('Max accuracy network')
    print('N =', maxN, 'config =', maxConfig, 'accuracy =', maxTestAccuracy)

def getAllOneVSAllNNResults():
    results = {}
    for N in range(5, 16):
        for classNum in range(1, 8):
            for config in myConfigList.myConfigurations:
                data = json.load(open('./unbalanced_data/onevsall_results/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'))
                if str(N) not in results:
                    results[str(N)] = {}
                if str(classNum) not in results[str(N)]:
                    results[str(N)][str(classNum)] = {}
                results[str(N)][str(classNum)][getConfigString(config)] = data[getConfigString(config)]
    return results

def printAllOneVSAllNNResults():
    results = getAllOneVSAllNNResults()
    maxTestAccuracy = {}
    for classNum in range(1, 8):
        maxTestAccuracy[str(classNum)] = {
            'maxTestAccuracy': 0,
            'config': '',
            'N': 0,
        }
    for N in range(5, 16):
        for classNum in range(1, 8):
            for config in results[str(N)][str(classNum)]:
                #print(config, results[str(N)][config]['max_test_accuracy'])
                if results[str(N)][str(classNum)][config]['max_test_accuracy'] > maxTestAccuracy[str(classNum)]['maxTestAccuracy']:
                    maxTestAccuracy[str(classNum)]['maxTestAccuracy'] = results[str(N)][str(classNum)][config]['max_test_accuracy']
                    maxTestAccuracy[str(classNum)]['config'] = config
                    maxTestAccuracy[str(classNum)]['N'] = N
    print('---- One VS All NN Result highlight ----')
    print('Max accuracy networks')
    for classNum in range(1, 8):
        print('class =', classNum)
        print('N =', maxTestAccuracy[str(classNum)]['N'], 'config =', maxTestAccuracy[str(classNum)]['config'], 'accuracy =', maxTestAccuracy[str(classNum)]['maxTestAccuracy'])

def getAllRNNResults():
    results = {}
    # Version 1
    temp = {}
    for N in range(5, 16):
        data = json.load(open('./balanced_raw_data/balanced_rnn_results/rnn_result_v1_N_' + str(N) + '.json'))
        temp[N] = data
    results[1] = temp
    # Version 2
    temp = {}
    for N in range(8, 15):
        data = json.load(open('./balanced_raw_data/balanced_rnn_results/rnn_result_v2_N_' + str(N) + '.json'))
        temp[N] = data
    results[2] = temp
    # Version 3
    temp = {}
    for N in [8, 10, 12, 14]:
        data = json.load(open('./balanced_raw_data/balanced_rnn_results/rnn_result_v3_N_' + str(N) + '.json'))
        temp[N] = data
    results[3] = temp
    # Version 2
    temp = {}
    for N in [8, 10, 12, 14]:
        data = json.load(open('./balanced_raw_data/balanced_rnn_results/rnn_result_v4_N_' + str(N) + '.json'))
        temp[N] = data
    results[4] = temp

    return results

def printAllRNNResults():
    rnnResults = getAllRNNResults()
    maxTestAccuracy = 0
    print('\n---- Version 1 results ----')
    for N in range(5, 16):
        if rnnResults[1][N][str(N)]['testing_accuracy'] > maxTestAccuracy:
            maxTestAccuracy = rnnResults[1][N][str(N)]['testing_accuracy']
            maxN = N
            maxVersion = 1
        print('N =', N, 'testing_accuracy =', rnnResults[1][N][str(N)]['testing_accuracy'], 'f1_score =', rnnResults[1][N][str(N)]['f1_score'])
    print('\n---- Version 2 results ----')
    for N in range(8, 15):
        if rnnResults[2][N][str(N)]['testing_accuracy'] > maxTestAccuracy:
            maxTestAccuracy = rnnResults[2][N][str(N)]['testing_accuracy']
            maxN = N
            maxVersion = 2
        print('N =', N, 'testing_accuracy =', rnnResults[2][N][str(N)]['testing_accuracy'], 'f1_score =', rnnResults[2][N][str(N)]['f1_score'])
    print('\n---- Version 3 results ----')
    for N in [8, 10, 12, 14]:
        if rnnResults[3][N][str(N)]['testing_accuracy'] > maxTestAccuracy:
            maxTestAccuracy = rnnResults[3][N][str(N)]['testing_accuracy']
            maxN = N
            maxVersion = 3
        print('N =', N, 'testing_accuracy =', rnnResults[3][N][str(N)]['testing_accuracy'], 'f1_score =', rnnResults[3][N][str(N)]['f1_score'])
    print('\n---- Version 4 results ----')
    for N in [8, 10, 12, 14]:
        if rnnResults[4][N][str(N)]['testing_accuracy'] > maxTestAccuracy:
            maxTestAccuracy = rnnResults[4][N][str(N)]['testing_accuracy']
            maxN = N
            maxVersion = 4
        print('N =', N, 'testing_accuracy =', rnnResults[4][N][str(N)]['testing_accuracy'], 'f1_score =', rnnResults[4][N][str(N)]['f1_score'])

    print('\nMaximum test accuracy =', maxTestAccuracy, 'obtained for RNN version', maxVersion, 'N =', maxN)

def printRNNResult(version, N):
    data = json.load(open('./balanced_raw_data/balanced_rnn_results/rnn_result_v' + str(version) + '_N_' + str(N) + '.json'))
    print('Result of RNN version', version, 'N =', N)
    print('testing_accuracy =', data[str(N)]['testing_accuracy'])
    print('training_accuracy =', data[str(N)]['training_accuracy'])
    print('validation_accuracy =', data[str(N)]['validation_accuracy'])
    print('n_epochs =', data[str(N)]['n_epochs'])
    print('rnn_size =', data[str(N)]['rnn_size'])
    print('precision =', data[str(N)]['precision'])
    print('recall =', data[str(N)]['recall'])
    print('f1_score =', data[str(N)]['f1_score'])
    print('confusion_matrix')
    for row in data[str(N)]['confusion_matrix']:
        print(row)

def writeResultsToCSV():
    '''
    # Write balanced NN results
    results = getAllBalancedNNResults()
    with open('balanced_nn_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Configuration', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        for config in myConfigList.myConfigurations:
            configResults = []
            configResults.append(getConfigString(config))
            for N in range(5, 16):
                configResults.append(results[str(N)][getConfigString(config)]['max_test_accuracy'])
            writer.writerow(configResults)

    # Write onevsall NN results
    results = getAllOneVSAllNNResults()
    with open('onevsall_nn_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for classNum in range(1, 8):
            writer.writerow(['Class ' + str(classNum)])
            writer.writerow(['Configuration', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            for config in myConfigList.myConfigurations:
                configResults = []
                configResults.append(getConfigString(config))
                for N in range(5, 16):
                    configResults.append(results[str(N)][str(classNum)][getConfigString(config)]['max_test_accuracy'])
                writer.writerow(configResults)
    '''
    # Write RNN results
    rnnResults = getAllRNNResults()
    with open('rnn_results_new.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Version 1 : single LSTM cell'])
        writer.writerow([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        accuracyList = []
        for N in range(5, 16):
            accuracyList.append(rnnResults[1][N][str(N)]['testing_accuracy'])
        writer.writerow(accuracyList)

        writer.writerow(['Version 2 : stack of 2 LSTM cells'])
        writer.writerow([8, 9, 10, 11, 12, 13, 14])
        accuracyList = []
        for N in range(8, 15):
            accuracyList.append(rnnResults[2][N][str(N)]['testing_accuracy'])
        writer.writerow(accuracyList)

        writer.writerow(['Version 3 : stack of 3 LSTM cells'])
        writer.writerow([8, 10, 12, 14])
        accuracyList = []
        for N in [8, 10, 12, 14]:
            accuracyList.append(rnnResults[3][N][str(N)]['testing_accuracy'])
        writer.writerow(accuracyList)

        writer.writerow(['Version 4 : stack of 4 LSTM cells'])
        writer.writerow([8, 10, 12, 14])
        accuracyList = []
        for N in [8, 10, 12, 14]:
            accuracyList.append(rnnResults[4][N][str(N)]['testing_accuracy'])
        writer.writerow(accuracyList)

        writer.writerow(['Version 1 : single LSTM cell'])
        writer.writerow(['N', 'training_accuracy', 'validation_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score'])
        for N in range(5, 16):
            accuracyList = []
            accuracyList.append(N)
            accuracyList.append(rnnResults[1][N][str(N)]['training_accuracy'])
            accuracyList.append(rnnResults[1][N][str(N)]['validation_accuracy'])
            accuracyList.append(rnnResults[1][N][str(N)]['testing_accuracy'])
            accuracyList.append(rnnResults[1][N][str(N)]['precision'])
            accuracyList.append(rnnResults[1][N][str(N)]['recall'])
            accuracyList.append(rnnResults[1][N][str(N)]['f1_score'])
            writer.writerow(accuracyList)

        writer.writerow(['Version 2 : stack of 2 LSTM cells'])
        writer.writerow(['N', 'training_accuracy', 'validation_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score'])
        for N in range(8, 15):
            accuracyList = []
            accuracyList.append(N)
            accuracyList.append(rnnResults[2][N][str(N)]['training_accuracy'])
            accuracyList.append(rnnResults[2][N][str(N)]['validation_accuracy'])
            accuracyList.append(rnnResults[2][N][str(N)]['testing_accuracy'])
            accuracyList.append(rnnResults[2][N][str(N)]['precision'])
            accuracyList.append(rnnResults[2][N][str(N)]['recall'])
            accuracyList.append(rnnResults[2][N][str(N)]['f1_score'])
            writer.writerow(accuracyList)

        writer.writerow(['Version 3 : stack of 3 LSTM cells'])
        writer.writerow(['N', 'training_accuracy', 'validation_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score'])
        for N in [8, 10, 12, 14]:
            accuracyList = []
            accuracyList.append(N)
            accuracyList.append(rnnResults[3][N][str(N)]['training_accuracy'])
            accuracyList.append(rnnResults[3][N][str(N)]['validation_accuracy'])
            accuracyList.append(rnnResults[3][N][str(N)]['testing_accuracy'])
            accuracyList.append(rnnResults[3][N][str(N)]['precision'])
            accuracyList.append(rnnResults[3][N][str(N)]['recall'])
            accuracyList.append(rnnResults[3][N][str(N)]['f1_score'])
            writer.writerow(accuracyList)

        writer.writerow(['Version 4 : stack of 4 LSTM cells'])
        writer.writerow(['N', 'training_accuracy', 'validation_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score'])
        for N in [8, 10, 12, 14]:
            accuracyList = []
            accuracyList.append(N)
            accuracyList.append(rnnResults[4][N][str(N)]['training_accuracy'])
            accuracyList.append(rnnResults[4][N][str(N)]['validation_accuracy'])
            accuracyList.append(rnnResults[4][N][str(N)]['testing_accuracy'])
            accuracyList.append(rnnResults[4][N][str(N)]['precision'])
            accuracyList.append(rnnResults[4][N][str(N)]['recall'])
            accuracyList.append(rnnResults[4][N][str(N)]['f1_score'])
            writer.writerow(accuracyList)

def plotResults():
    # Plots for balanced NN

    results = getAllBalancedNNResults()

    '''
    # Plot max accuracy for each N
    accuracyList = []
    for N in range(5, 16):
        max = 0
        for config in results[str(N)]:
            if results[str(N)][config]['max_test_accuracy'] > max:
                max = results[str(N)][config]['max_test_accuracy']
        accuracyList.append(max)
    listN = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fig, ax = plt.subplots()
    rects = plt.bar(listN, accuracyList)
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), '%.4f' % rect.get_height(), ha='center', va='bottom')
    plt.xticks(listN)
    plt.xlabel('N')
    plt.ylabel('Max. accuracy')
    plt.show()
    '''
    '''
    # Plot max. accuracy for different hidden layers for a N
    accuracyList = []
    N = 10
    numOfHiddenLayers = [3, 4, 5, 6, 7]
    currLayers = 3
    max = 0
    for config in myConfigList.myConfigurations:
        if len(config) == currLayers:
            if results[str(N)][getConfigString(config)]['max_test_accuracy'] > max:
                max = results[str(N)][getConfigString(config)]['max_test_accuracy']
        else:
            accuracyList.append(max)
            currLayers += 1
            max = results[str(N)][getConfigString(config)]['max_test_accuracy']
    accuracyList.append(max)
    fig, ax = plt.subplots()
    rects = plt.bar(numOfHiddenLayers, accuracyList)
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), '%.4f' % rect.get_height(), ha='center', va='bottom')
    plt.xticks(numOfHiddenLayers)
    plt.xlabel('Number of hidden layers')
    plt.ylabel('Max. accuracy for N = ' + str(N))
    plt.show()
    '''

    # Plots for onevsall NN

    results = getAllOneVSAllNNResults()

    '''
    # Plot max. accuracy for each class
    classes = []
    accuracyList = []
    for classNum in range(1, 8):
        classes.append(classNumToLabel[classNum])
        max = 0
        for N in range(5, 16):
            for config in results[str(N)][str(classNum)]:
                if results[str(N)][str(classNum)][config]['max_test_accuracy'] > max:
                    max = results[str(N)][str(classNum)][config]['max_test_accuracy']
        accuracyList.append(max)
    fig, ax = plt.subplots()
    rects = plt.bar(classes, accuracyList)
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), '%.4f' % rect.get_height(), ha='center', va='bottom')
    plt.xticks(classes)
    plt.xlabel('Classes')
    plt.ylabel('Max. accuracy')
    plt.show()
    '''

    # Plot max. accuracy for each N for a class
    classNum = 3
    listN = []
    accuracyList = []
    for N in range(5, 16):
        max = 0
        listN.append(N)
        for config in results[str(N)][str(classNum)]:
            if results[str(N)][str(classNum)][config]['max_test_accuracy'] > max:
                max = results[str(N)][str(classNum)][config]['max_test_accuracy']
        accuracyList.append(max)
    fig, ax = plt.subplots()
    rects = plt.bar(listN, accuracyList)
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), '%.4f' % rect.get_height(), ha='center', va='bottom')
    plt.xticks(listN)
    plt.xlabel('N')
    plt.ylabel('Max. accuracy for class = ' + classNumToLabel[classNum])
    plt.show()

def analyzeReport():

    N = 5
    classNum = 3
    numOfNodes = 100
    numOfLayers = 15
    configList = []
    for layers in range(1, numOfLayers + 1):
        temp = []
        for _ in range(1, layers + 1):
            temp.append(numOfNodes)
        configList.append(temp)
    layersList = []
    accuracyList = []
    with open('report/onevsall/onevsall_accvslayers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N = 5, class = pushing, number of nodes per layer = 100'])
        writer.writerow(['No. of layers', 'Accuracy'])
        for config in configList:
            data = json.load(open('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'))
            writer.writerow([len(config), data[getConfigString(config)]['max_test_accuracy']])

    '''
    N = 12
    with open('report/rnn/rnn_acc_vs_rnnsize.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N = 12, stack of 2 LSTM cells'])
        writer.writerow(['rnn_size', 'accuracy'])
        for rnn_size in [32, 64, 128, 256, 512]:
            data = json.load(open('report/rnn/rnn_result_v2_N_' + str(N) + '_rnn_size_' + str(rnn_size) + '.json'))
            writer.writerow([rnn_size, data[str(N)]['testing_accuracy']])
    '''
    '''
    N = 5
    classNum = 3
    with open('report/onevsall/accvsnumofnodes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N = 5, class = pushing, numOfLayers = 7'])
        writer.writerow(['numOfNodes per layer', 'accuracy'])
        for numOfNodes in [10, 50, 100, 200, 300, 400, 500, 1000]:
            numOfLayers = 7
            #print('\n\nN =', N, 'numOfLayers =', numOfLayers)
            config = []
            for _ in range(1, numOfLayers + 1):
                config.append(numOfNodes)
            data = json.load(open('report/onevsall/onevsall_N_' + str(N) + '_class_' + str(classNum) + '_NN_' + getConfigString(config) + '.json'))
            writer.writerow([numOfNodes, data[getConfigString(config)]['max_test_accuracy']])
    '''


if __name__ == '__main__':
    #printAllRNNResults()
    #printRNNResult(2, 12)
    #printAllBalancedNNResults()
    #printAllOneVSAllNNResults()
    #writeResultsToCSV()
    #plotResults()
    #results = getAllOneVSAllNNResults()
    analyzeReport()
    '''
    with open('onevsall_nn_results_new.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Configuration', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        for config in myConfigList.myConfigurations:
            accuracyList = []
            average = 0
            accuracyList.append(getConfigString(config))
            for N in range(5, 16):
                sum = 0
                for classNum in range(1, 8):
                    sum += results[str(N)][str(classNum)][getConfigString(config)]['max_test_accuracy']
                average = sum / 7
                accuracyList.append(average)
            writer.writerow(accuracyList)
    '''
