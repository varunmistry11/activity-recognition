import numpy as np
import csv
import json
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# training a linear SVM classifier
from sklearn.svm import SVC
import random

numToLabel= {
     1:'eating',
     2:'kicking',
     3:'pushing2',
     4:'running',
     5:'snatching2',
     6:'vaulting',
     7:'walking',
     8:'pushing',
     9:'snatching'
}
for N in range(5,16):
    print(N)
    train_loaded = np.genfromtxt('Data/train_'+ str(N) + '_tuples.csv', delimiter=',')

    test = np.genfromtxt('Data/test_'+ str(N) + '_tuples.csv', delimiter=',')

    #splitting training data
    training_data = [0]
    #testing_data = [0]

    for Class in numToLabel:

        training_data.insert(Class, [x for x in train_loaded if x[len(x)-1] == Class])
        #testing_data.insert(Class, [x for x in test_loaded if x[len(x) - 1] == Class])


    #    train=pd.read_csv('../_Varun/train_'+str(N)+'_tuples.csv')
    #    test=pd.read_csv('../_Varun/test_'+str(N)+'_tuples.csv')
    with open('Results/oneVSall_unbalancedTestSet_N='+str(N)+'.csv', 'w') as csvfile:

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['activity', 'accuracy', 'positive data', 'negative data', 'Test positive data','Test negative data','kernel', 'C'])

        for Class in numToLabel:
            positiveClass = training_data[Class]
            negativeClass = []

            print(numToLabel[Class])
            for ClassIterator in numToLabel:
                if ClassIterator != Class:
                    if len(positiveClass)/8 > len(training_data[ClassIterator]):
                        no_of_elements = len(training_data[ClassIterator])
                    else :
                        no_of_elements = int(len(positiveClass)/8)
                    negativeClass.extend(random.sample(training_data[ClassIterator], no_of_elements))

            posDataLen =  len(positiveClass)
            negDataLen =  len(negativeClass)

            train = positiveClass
            train.extend(negativeClass)

            train = np.array(train)

            '''
            #splitting testing data
            positiveClass = testing_data[Class]
            negativeClass = []

            for ClassIterator in numToLabel:
                if ClassIterator != Class:
                    if len(positiveClass)/8 > len(testing_data[ClassIterator]):
                        no_of_elements = len(testing_data[ClassIterator])
                    else :
                        no_of_elements = int(len(positiveClass)/8)
                    negativeClass.extend(random.sample(testing_data[ClassIterator], no_of_elements))

            print("Testing")
            print(len(positiveClass))
            print(len(negativeClass))

            posTestDataLen = len(positiveClass)
            negTestDataLen = len(negativeClass)

            test = positiveClass
            test.extend(negativeClass)

            test = np.array(test)
            '''
            X_train = train[:, 0:-1]

            y_train = train[:, -1]
            #print("y_train_lenght : " + str(len(y_train[0])))
            X_test = test[:, 0:-1]
            #print("x_test_lenght : " + str(len(X_test[0])))
            y_test = test[:, -1]

            #print(np.unique(y_test,return_counts=True))

            #for Class in numToLabel:

            temp_train =  (y_train==Class).astype(int)
            temp_test =  (y_test==Class).astype(int)

            #print(np.unique(y_test,return_counts=True))

            print(np.unique(y_train,return_counts=True))
            print(np.unique(y_test,return_counts=True))
            values,counts = np.unique(y_test, return_counts=True)
            posTestDataLen = 0
            negTestDataLen = 0

            for i in range(0,len(values)):

                if values[i] == Class:

                    posTestDataLen = counts[i]
                else:
                    negTestDataLen = negTestDataLen + counts[i]

            kernels= ['linear','rbf','sigmoid','poly']
            opt_kernel = ''
            opt_C = 0
            opt_accuracy = 0
            opt_cm = [[]]
            for k in kernels:
                for c in range(1,1002,50):
                    svm_model_linear = SVC(kernel=k, C = c).fit(X_train, temp_train)
                    svm_predictions = svm_model_linear.predict(X_test)
                    accuracy = svm_model_linear.score(X_test, temp_test)



                    if accuracy > opt_accuracy:
                        cm = confusion_matrix(temp_test, svm_predictions)
                        opt_cm = cm
                        opt_C = c
                        opt_kernel = k
                        opt_accuracy = accuracy

            csvwriter.writerow([numToLabel[Class], opt_accuracy, posDataLen, negDataLen,posTestDataLen,negTestDataLen, opt_kernel, opt_C])

            print("\n\n")
            print(numToLabel[Class])
            print("Accuracy: " + str(opt_accuracy))
            print("Kernel: " + str(opt_kernel))
            print("C: " + str(opt_C))
            print(opt_cm)
            print("\n\n")
    # model accuracy for X_test
    #accuracy = svm_model_linear.score(X_test, temp_test)

    # creating a confusion matrix
    #cm = confusion_matrix(temp_test, svm_predictions)
    #csvwriter.writerow([Class, accuracy])

