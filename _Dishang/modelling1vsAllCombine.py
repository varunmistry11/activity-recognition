# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:35:12 2018

@author: Dishang
"""

#import pandas as pd
from metrics import scores,total_accuracy
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,Dropout

import numpy as np
import matplotlib.pyplot as plt
numToLabel= {
     1:'eating',
     2:'kicking',
     3:'pushing2',
     4:'running',
     5:'snatching2',
     6:'vaulting',
     7:'walking' ,
#     8:'pushing',
#     9:'snatching'
}

def get_model(in_,output_dim=1):
# =============================================================================
#     Define your model here
# =============================================================================
    
    #model=OneVsRestClassifier(LinearSVC(random_state=0),n_jobs=-1)
    
    #model=OneVsRestClassifier(LogisticRegression())
    
    model = Sequential()
    
    model.add(Dense(100, input_dim=in_, activation='sigmoid'))
#    model.add(Dropout(0.2))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
#    model.add(Dense(500, activation='sigmoid'))
#    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(7, activation='sigmoid'))
    model.add(Dense(output_dim, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    
    return model
    

if __name__ == '__main__':
    N=12
    threshold=0.75
    train=np.genfromtxt('../_Varun/train_'+str(N)+'_tuples.csv', delimiter=',')
    test =np.genfromtxt('../_Varun/test_'+str(N)+'_tuples.csv', delimiter=',')

#    train=pd.read_csv('../_Varun/train_'+str(N)+'_tuples.csv')
#    test=pd.read_csv('../_Varun/test_'+str(N)+'_tuples.csv')
    
    train_X=train[:,0:-1]
    train_y=train[:,-1]
    
    test_X=test[:,0:-1]
    test_y=test[:,-1]
    
    models=[]
        
    for Class in numToLabel:
#        print('----','Class',Class,numToLabel[Class],'----')
        model=get_model(train_X.shape[1])
        model.fit(train_X, (train_y==Class).astype(int),epochs=100,verbose=0)
        models.append(model)
        print(Class)
#        predict=model.predict(test_X)
#        predict=predict.flatten()
#        scores(predict,(test_y==Class).astype(int),threshold)
#        plt.hist(predict,bins=20)
#        plt.show()
    output=[]
    for m in models:
        output.append(m.predict(test_X))
    output=np.array(output)
    output=output.reshape(output.shape[0:2])
    output=output.transpose()
    max_output_index=output.argmax(axis=1)
    predict=max_output_index+1
    
#    predict=[]
#    for i in range(0,len(test_X)):
#        output=[]
#        for m in models:
#            output.append(m.predict(test_X[i:i+1]))
#        predict.append(output.index(max(output)))
#    #        print(1)
        
    total_accuracy(predict,test_y)