# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 07:37:58 2020

@author: Rajesh Thennan

Parallel_Dataprep_2.py should have completed successfully
Run this and proceed to modelBuild.py

"""


import pandas as pd # pip install pandas
from os import listdir
import numpy as np 
from math import ceil
from sklearn.preprocessing import StandardScaler # pip install -U scikit-learn
from pickle import dump

sourcePicksPath = 'splitPicks2\\'
sourcePicks = listdir(sourcePicksPath)
refTbRows = []
totalRows = len(sourcePicks)
for x in range(totalRows):
    refTbRows.append(x)

inData = pd.read_pickle(sourcePicksPath+'\\'+sourcePicks[0])   

for x in range(1,len(refTbRows)):
    #print(x)
    inData = inData.append(pd.read_pickle(sourcePicksPath+'\\'+sourcePicks[x]))
    
    
'''5. Shuffle'''

inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)
inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)
inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)


'''7. Balance'''
maxCount = inData.Label.value_counts().max()

negative = inData[inData['Label']==0]
positive = inData[inData['Label']==1]

while len(negative) < maxCount:
    diff = maxCount - len(negative)
    rowsToCopy = min(diff,len(negative))
    negative = pd.concat([negative,negative[:rowsToCopy]], ignore_index=True)
    
    negative = negative.reindex(np.random.permutation(negative.index))
    negative.reset_index(inplace=True,drop=True)   

'''8. Shuffle again'''
positive = positive.reindex(np.random.permutation(positive.index))
positive.reset_index(inplace=True,drop=True)    

'''9. Sequential combine =***OPTIONAL***'''
inData = pd.DataFrame()

for x in range(maxCount):
    inData = inData.append(positive.iloc[x])
    inData = inData.append(negative.iloc[x])
    
    
'''10. Split Train and test'''
splitSize = ceil(len(inData)*0.8)
train = inData[:splitSize]
test = inData[splitSize+1:]


'''11. Split X and Y'''
X_train_tab=train[['Age', 'Sex']].values
X_train_img=train['imgData'].values
Y_train=train['Label'].values

X_test_tab=test[['Age', 'Sex']].values
X_test_img=test['imgData'].values
Y_test=test['Label'].values


'''12. Scale Tabular Data
Creating a scalar object that can be saved and reused at inference'''
scScaler = StandardScaler()
scScaler = scScaler.fit(X_train_tab)
X_train_tab = scScaler.transform(X_train_tab)
X_test_tab = scScaler.transform(X_test_tab)


'''Saving the Scalar object'''
dump(scScaler, open('scScaler.pkl', 'wb'))

'''Reshaping the Image array so TensorFlow likes it'''

imgSize = inData['imgData'].iloc[0].shape[0]
X_train_img = np.stack(X_train_img)
X_train_img = np.array(X_train_img).reshape(-1, imgSize, imgSize, 1)

X_test_img = np.stack(X_test_img)
X_test_img = np.array(X_test_img).reshape(-1, imgSize, imgSize, 1)

np.save('X_train_tab.npy', X_train_tab)
np.save('X_train_img.npy', X_train_img)
np.save('Y_train.npy', Y_train)

np.save('X_test_tab.npy', X_test_tab)
np.save('X_test_img.npy', X_test_img)
np.save('Y_test.npy', Y_test)    