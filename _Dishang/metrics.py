# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:01:28 2018

@author: Dishang
"""
import numpy as np
import pandas as pd
import sklearn.metrics as sm
#import pprint
#pp = pprint.PrettyPrinter(indent=1)

def accuracy(predict,y):
# =============================================================================
#     tp/tp+fn+fp
# =============================================================================
    num=np.logical_and(predict,y).sum()
    denom=np.logical_or(predict,y).sum()
    print('Accuracy:\t',str(num)+'/'+str(denom)+'\t',num/denom,
          'FP+FN:',denom-num)


def precision(predict,y):
# =============================================================================
#     tp/tp+fp
# =============================================================================
    num=np.logical_and(predict,y).sum()
    denom=(predict==1).sum()
    print('Precision:\t',str(num)+'/'+str(denom)+'\t',num/denom,
          'FP:',denom-num)
    
def recall(predict,y):
# =============================================================================
#     tp/tp+fn
# =============================================================================
    num=np.logical_and(predict,y).sum()
    denom=(y==1).sum()
    print('Recall:\t\t',str(num)+'/'+str(denom)+'\t',num/denom,
          'FN:',denom-num)
    
def per_class_accuracy(predict,y):
    Class,count=np.unique(y,return_counts=True)
    y_count=dict(zip(Class,count))
    
    predict_count=[]
    for c in Class:
        predict_count.append(np.logical_and((y==c),(predict==c)).sum())    
    predict_count=dict(zip(Class,predict_count))
    
    predicted_Class,predicted_count=np.unique(predict,return_counts=True)
    predicted_count=dict(zip(predicted_Class,predicted_count))
    
    print('----','Per class Accuracy:','----')
    print('Class','\t','Predicted','\t','Accuracy','\t')
    for c in Class:
        try:
            print(c,':','\t',predicted_count[c],'\t\t',
                  str(predict_count[c])+'/'+str(y_count[c]),'\t',
                  str((predict_count[c]/y_count[c])*100)+'%')
        except KeyError:
            print(c,':','\t',0,'\t\t',
                  str(predict_count[c])+'/'+str(y_count[c]),'\t',
                  str((predict_count[c]/y_count[c])*100)+'%')
    

def total_accuracy(predict,y):
    
    num=(predict==y).sum()
    denom=(y<7).sum()
    
    print('Total accuracy:',str(num)+'/'+str(denom),'\t',num/denom,'%')
    
    
    
#def accuracy(predict,y):
#    total_accuracy(predict,y)
#    per_class_accuracy(predict,y)
    
def scores(predict,y,threshold=0.99):
    predict=(predict>=threshold).astype(int)
    accuracy(predict,y)
    precision(predict,y)
    recall(predict,y)
    
if __name__ == '__main__':
    scores(predict,(test_y==Class).astype(int),threshold)
    pass