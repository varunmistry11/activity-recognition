# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:02:38 2018

@author: Dishang
"""

#import pandas as pd
from metrics import scores
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
numToLabel= {
     1:'eating',
     2:'kicking',
#     3:'pushing',
     4:'pushing2',
     5:'running',
#     6:'snatching',
     7:'snatching2',
     8:'vaulting',
     9:'walking' 
}

def get_model(input_dim,output_dim=1):
# =============================================================================
#     Define your model here
# =============================================================================
    
    #model=OneVsRestClassifier(LinearSVC(random_state=0),n_jobs=-1)
    
    #model=OneVsRestClassifier(LogisticRegression())
    
    model = Sequential()
    
    model.add(Dense(234, input_dim=input_dim, activation='relu'))
    model.add(Dense(180, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    
    return model
    

if __name__ == '__main__':
    N=10
    threshold=0.75
    train=np.genfromtxt('../_Varun/train_'+str(N)+'_tuples.csv', delimiter=',')
    test =np.genfromtxt('../_Varun/test_'+str(N)+'_tuples.csv', delimiter=',')

#    train=pd.read_csv('../_Varun/train_'+str(N)+'_tuples.csv')
#    test=pd.read_csv('../_Varun/test_'+str(N)+'_tuples.csv')
    
    train_X=train[:,0:-1]
    train_y=train[:,-1]
    
    test_X=test[:,0:-1]
    test_y=test[:,-1]
    
    for Class in numToLabel:
        print('----','Class',Class,numToLabel[Class],'----')
        model=get_model(train_X.shape[1])
        model.fit(train_X, (train_y==Class).astype(int),epochs=10,verbose=0)
        predict=model.predict(test_X)
        predict=predict.flatten()
        scores(predict,(test_y==Class).astype(int),threshold)
#        plt.hist(predict,bins=20)
#        plt.show()
        
        