# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 07:37:58 2020

@author: Rajesh Thennan

Parallel_Dataprep_1.py should have completed successfully
Run this and proceed to Parallel_Dataprep_3.py
"""


import pandas as pd # pip install pandas
from os import makedirs, path, listdir
import numpy as np 
import cv2 # pip install opencv-python
from multiprocessing import Pool, cpu_count
from datetime import datetime as dt


imgPath = 'images'

sourcePicksPath = 'splitPicks1\\'
sourcePicks = listdir(sourcePicksPath)
refTbRows = []
totalRows = len(sourcePicks)
for x in range(totalRows):
    refTbRows.append(x)

destPicksPath ='splitPicks2\\'
if not path.exists(destPicksPath):
    makedirs(destPicksPath)

def imgRead(dfIn):
    inPath = dfIn['filePath']
    imgSize = dfIn['imgSize']
    '''Try catch block to avoid code failure due to other unforeseen issues'''
    try:
        '''3.1 Reading corresponding images and converting to grayscale'''
        tempImageArray = cv2.imread(path.join(imgPath,inPath) ,cv2.IMREAD_GRAYSCALE)
        tempImageArray  = cv2.resize(tempImageArray, (imgSize, imgSize)) 
        #3.4 0-255 sclaing - data scaling
        tempImageArray = tempImageArray/255 
        tempImageArray = np.array(tempImageArray).reshape(imgSize, imgSize)
        return tempImageArray
        
        ''' Add other preprocessing like Denoising, Image Enhacement, ROI, denoising, etc. here'''
    except:
        return np.nan  

    
def imgPreProcess1(rowNum):
    
    msg = str(dt.now())+ '   Began Processing Row  =>  '+str(rowNum) + '/'+ str(totalRows)
    print(msg)
    try:
        inData = pd.read_pickle(sourcePicksPath+'\\'+sourcePicks[rowNum])       
        inData['imgData']  = inData.apply(imgRead, axis = 1) 
        inData.dropna(inplace=True)
        inData.drop(['filePath'], axis = 1, inplace = True) 
        
        '''6. Change Labels to integers'''
        labelLookup={'Normal':0,'Pnemonia':1}
        if len(inData) > 0 :
            inData['Label'] = inData['Label'].apply(lambda x:labelLookup[x])    
            inData.to_pickle(destPicksPath+'\\'+str(rowNum)+'.csv')
    except:
        pass
    msg = str(dt.now())+ '   Done Processing Row  =>  '+str(rowNum) + '/'+ str(totalRows)
    print(msg) 
    
    
    
if __name__ == '__main__':
    msg = 'Started Preprocessing '
    print('\n')
    print(msg)  
    p = Pool(max(cpu_count()-2,1)) #Specify number of threads
    p.map(imgPreProcess1, refTbRows)
    print('Preprocessing Completed')
    print('\n')      