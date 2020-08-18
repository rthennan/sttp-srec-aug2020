# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16  06:03:19 2020

@author: Rajesh Thennan

Seq_Dataprep.py or Parallel_Dataprep_3.py or Parallel_Dataprep_V2.py should have been compleetd successfully
"""
import tensorflow as tf # pip install tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import  Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from datetime import datetime as dt
from os import path, makedirs
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


batch_size = 500
num_classes = 2
epochs = 20
dpOut = 0.2


ver='V1'



X_train_tab = np.load('X_train_tab.npy',allow_pickle=True)
X_train_img = np.load('X_train_img.npy',allow_pickle=True)
Y_train = np.load('Y_train.npy',allow_pickle=True)


X_test_tab = np.load('X_test_tab.npy',allow_pickle=True)
X_test_img = np.load('X_test_img.npy',allow_pickle=True)
Y_test = np.load('Y_test.npy',allow_pickle=True)


tensLogDir = 'CNNlogs_binary'
NAME = ver+'_'+str(dt.now().strftime("%Y_%m_%d_%H_%M_%S"))
logDirs = tensLogDir+'\\'+NAME
if not path.exists(logDirs):
    makedirs(logDirs)
tensorboard = TensorBoard(log_dir=logDirs)

def modelSize(model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

#Building a CNN for processing Images
cnnModelIn = Input(shape=(X_train_img[0].shape))
cnnModel = Conv2D(32, (3, 3), activation='relu', padding='same')(cnnModelIn)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D((2, 2))(cnnModel)
cnnModel = Dropout(dpOut)(cnnModel)

cnnModel = Conv2D(64, (3, 3), activation='relu', padding='same')(cnnModelIn)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = MaxPooling2D((2, 2))(cnnModel)
cnnModel = Dropout(dpOut)(cnnModel)

cnnModel= Flatten()(cnnModel)

#Building an ANN for processing Tabular data - Exogenous / demographic information
exoModIn = Input(shape=X_train_tab.shape[1])
exoMod = Dense(10, activation='relu')(exoModIn)
exoMod = Dropout(dpOut)(exoMod)

combinedMod = concatenate([exoMod,cnnModel])

combinedMod = Dense(10, activation='relu')(combinedMod)
combinedMod = Dropout(dpOut)(combinedMod)  
combinedMod = BatchNormalization()(combinedMod)    
   
combinedMod = Dense(num_classes, activation='softmax')(combinedMod)


mergedModel = Model(inputs=[exoModIn,cnnModelIn], outputs=[combinedMod])

opt = Adam(learning_rate =0.01, decay=1e-6)

mergedModel.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

trainableParams = modelSize(mergedModel)
print(dt.now(),' =====> Count of Trainable Params:',trainableParams)
print("/n***/n***/n***")
print("*** Version: ***", ver)
print(dt.now(),' =====> Count of Trainable Params:',trainableParams)
print("/n***/n***/n***")
print("*** Version: ***", ver)    
print(dt.now(),' =====> Count of Trainable Params:',trainableParams)
print("/n***/n***/n***")    
print("*** Version: ***", ver)    
print("/n***/n***/n***") 


filepath = ver+"-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch

checkpoint = ModelCheckpoint(r"models\{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones


history1 = mergedModel.fit([X_train_tab,X_train_img], y=Y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=2,
                     callbacks=[tensorboard,checkpoint],
                     validation_data=([X_test_tab,X_test_img], Y_test))
                     #) 
                     
print("/n***/n***/n***")
mergedModel.save(NAME+'.h5')
print("*** Version: ***", ver)
print("/n***/n***/n***")
msg = 'Finished Training For '+ver
print(dt.now(),' =====> Count of Trainable Params:',trainableParams)

# Plot confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=NAME,
                          cmap=plt.cm.Blues):

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(NAME+'_confusionMatrix.png', bbox_inches='tight')
  #plt.show()

p_test = mergedModel.predict([X_test_tab, X_test_img]).argmax(axis=1)
#rounded_labels=np.argmax(yTest, axis=1)

cm = confusion_matrix(Y_test, p_test)
plot_confusion_matrix(cm, [1,0])
cmName = ver+'confusMatrix.npy'
np.save(cmName, cm)                         









