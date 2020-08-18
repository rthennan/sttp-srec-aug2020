# Simple Image Classifaction model using CNN
<br><br>
**Seq_Dataprep.py**
<br>
**The images folder and chestXrayCoronaMetadata.csv should be accssible in the current working directory**
<br>
1. Read All images and find smallest dimension<br>
2.Tabular Data Read -  Tabular Data with Image Path and class<br>
3. Image data read and preprocessing<br>
    3.1 Reading corresponding images and converting to grayscale<br>
    3.2 resize image to smallest dimension<br>
    3.3 Scale down further if required (not doing here)<br>
    3.4 0-255 sclaing - data scaling<br>

4. Cleanup - remove missing rows<br>
5. Shuffle<br>
6. Change Labels to integers<br>
7. Balance<br>
8. Shuffle again<br>
9. Sequential combine -***OPTIONAL***<br>
10. Split Train and test<br>
11. Split X and Y<br>
12. Scale Tabular Data<br>
13. Export as Numpy Array, ready for model<br>
<br>
**modelBuild.py**
<br>
1. Concatenated model, combining an ANN from tabular data and CNN a from image data
2. Used Tensorflow callbacks TensorBoard and checkpoint(ModelCheckpoint)
<br>
<br>
**Parallel_Dataprep_1.py , Parallel_Dataprep_2.py and Parallel_Dataprep_3.py**
<br>
-Simplified parallel processing equivalent of Seq_Dataprep.py
