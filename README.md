# STTP - SREC - Rajesh Thennan
This repo is to host the corresponding slide deck and code from my presentation in the below conference:
<br><br>
One Week ONLINE Short Term Training Programme (STTP) on “Research Issues and Challenges in Deep learning-based Medical Image analysis and Medical Diagnosis" organaised by the  Department of Electronics and Communication Engineering of Sri Ramakrishna Engineering College, Coimbatore,
<br>
Tamilnadu
<br>
India
<br>
<br>
contact: rajesh.thennan@gmail.com
<br>

Image data obtained from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset and simplified

# Concatenated Image Classifaction model using CNN (for Image) and ANN (for Tabular Data)
<br><br>
**The images folder and chestXrayCoronaMetadata.csv should be accssible in the current working directory**
<br><br>
**Use Parallel processing if you have more than 4 CPU threads**
<br>
**Seq_Dataprep.py** 
<br>
<br>
1.Read All images and find smallest dimension<br>
2.Tabular Data Read -  Tabular Data with Image Path and class<br>
3.Image data read and preprocessing<br>
    3.1 Reading corresponding images and converting to grayscale<br>
    3.2 resize image to smallest dimension<br>
    3.3 Scale down further if required (not doing here)<br>
    3.4 0-255 sclaing - data scaling<br>

4.Cleanup - remove missing rows<br>
5.Shuffle<br>
6.Change Labels to integers<br>
7.Balance<br>
8.Shuffle again<br>
9.Sequential combine -***OPTIONAL***<br>
10.Split Train and test<br>
11.Split X and Y<br>
12.Scale Tabular Data<br>
13.Export as Numpy Array, ready for model<br>
<br>
<br>
**Skip the Parallel Processing files and jump to modelBuild if you choose to use Sequential Processing**
<br>
<br>
**Parallel_Dataprep_1.py , Parallel_Dataprep_2.py and Parallel_Dataprep_3.py**
<br>
-Simplified parallel processing equivalent of Seq_Dataprep.py
<br>
<br>
**Parallel_Dataprep_V2.py**
<br>
-Slightly more complex process.
<br>
-Tedious to troubleshoot.
<br>
-But gets the job done in one file.


**modelBuild.py**
<br>
1.Concatenated model, combining an ANN from tabular data and CNN a from image data<br>
2.Used Tensorflow callbacks TensorBoard and checkpoint(ModelCheckpoint)
<br>
<br>
