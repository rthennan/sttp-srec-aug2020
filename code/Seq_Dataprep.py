# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 03:38:13 2020

@author: Rajesh Thennan

Sequential Processing

***The images folder and chestXrayCoronaMetadata.csv should be accssible in teh current working directory***

1. Read All images and find smallest dimension
2.Tabular Data Read -  Tabular Data with Image Path and class
3. Image data read and preprocessing
    3.1 Reading corresponding images and converting to grayscale
    3.2 resize image to smallest dimension
    3.3 Scale down further if required (not doing here)
    3.4 0-255 sclaing - data scaling

4. Cleanup - remove missing rows
5. Shuffle
6. Change Labels to integers
7. Balance
8. Shuffle again
9. Sequential combine -***OPTIONAL***
10. Split Train and test
11. Split X and Y
12. Scale Tabular Data
13. Export as Numpy Array, ready for model

"""

import pandas as pd # pip install pandas
import os
import numpy as np 
import cv2 # pip install opencv-python
from PIL import Image
from math import ceil
from sklearn.preprocessing import StandardScaler # pip install -U scikit-learn
from pickle import dump
from datetime import datetime as dt

msg = 'Started Preprocessing '
print('\n')
print(str(dt.now())[:19],msg)

imgPath = 'images'

'''1. Read All images and find smallest dimension. If the smallest dimension is too small, consider cleaning up such small images and recomputing'''
imageFiles = os.listdir(imgPath)
imgSize = 100000
for img in imageFiles:
    tempImg = Image.open(os.path.join(imgPath,img))
    imgSize =  min(min(tempImg.size),imgSize)

'''Show Temp Images'''
# =============================================================================
# tempImg = cv2.imread(os.path.join(imgPath,img) ,cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# =============================================================================

'''2.Tabular Data Read - Tabular Data with Image Path and class'''
inData = pd.read_csv('chestXrayCoronaMetadata.csv')



'''3. Image data read and preprocessing'''
def imgRead(inPath):
    '''Try catch block to avoid code failure due to other unforeseen issues'''
    try:
        '''3.1 Reading corresponding images and converting to grayscale'''
        tempImageArray = cv2.imread(os.path.join(imgPath,inPath) ,cv2.IMREAD_GRAYSCALE)
        tempImageArray  = cv2.resize(tempImageArray, (imgSize, imgSize)) 
        #3.4 0-255 sclaing - data scaling
        tempImageArray = tempImageArray/255 
        tempImageArray = np.array(tempImageArray).reshape(imgSize, imgSize)
        return tempImageArray
        
        ''' Add other preprocessing like Denoising, Image Enhacement, ROI, denoising, etc. here'''
    except:
        return np.nan    

inData['imgData']  = inData['filePath'].apply(imgRead)    


# =============================================================================
# cv2.imshow('image',inData['imgData'].iloc[100])
# cv2.waitKey(0)
# =============================================================================


'''4. Cleanup - remove missing rows'''
print('Nan Count ->\n',inData.isna().sum().max(),' Nan % ->',round(inData.isna().sum().max()*100/len(inData),2))

inData.dropna(inplace=True)
print('Nan Count -> \n',inData.isna().sum())

#Dropping unnescary column - filepath
inData.drop(['filePath'], axis = 1, inplace = True) 

'''5. Shuffle'''

inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)
inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)
inData = inData.reindex(np.random.permutation(inData.index))
inData.reset_index(inplace=True,drop=True)



'''6. Change Labels to integers'''
labelLookup={'Normal':0,'Pnemonia':1}
inData['Label'] = inData['Label'].apply(lambda x:labelLookup[x])

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

msg = 'Finished Preprocessing '
print('\n')
print(str(dt.now())[:19],msg) 





















