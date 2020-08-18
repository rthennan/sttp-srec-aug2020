# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 07:37:58 2020

@author: Rajesh Thennan
Splitting the data into chunks for paralle processing
***The images folder and chestXrayCoronaMetadata.csv should be accssible in teh current working directory***

Run this and proceed to Parallel_Dataprep_2.py

"""


import pandas as pd # pip install pandas
from PIL import Image
from os import makedirs, path, listdir
from math import ceil

imgPath = 'images'


'''1. Read All images and find smallest dimension. If the smallest dimension is too small, consider cleaning up such small images and recomputing'''
imageFiles = listdir(imgPath)
imgSize = 100000
for img in imageFiles:
    tempImg = Image.open(path.join(imgPath,img))
    imgSize =  min(min(tempImg.size),imgSize)
    
    
'''2.Tabular Data Read - Tabular Data with Image Path and class'''
inData = pd.read_csv('chestXrayCoronaMetadata.csv')
inData['imgSize'] = imgSize


splitCount = 100


combinedPickDir = 'splitPicks1'
if not path.exists(combinedPickDir):
    makedirs(combinedPickDir)    


batchCount = ceil(len(inData)/splitCount)
    
for y in range(splitCount):
    inData[(y*batchCount):((y+1)*batchCount)].to_pickle(combinedPickDir+'\\'+str(y)+'.csv')        