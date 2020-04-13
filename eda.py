

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import seaborn as sns
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Mount drive 
#from google.colab import drive
#drive.mount('/content/drive')
#!ls

#Get all file names from the dataset folder
cwd = os.getcwd()
mylist= os.listdir('/all_actors')
print(len(mylist))

#Get labels from filename
feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[:1]=='a':
        feeling_list.append('male_angry')
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
    elif item[:1]=='h':
        feeling_list.append('male_happy')
    #elif item[:1]=='n':
       # feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')

#initialize number of coefficients in mfcc
n_mfcc=13

#Get MFCC feature for every sample
mfcclist=[]
bookmark=0
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
        X, sample_rate = librosa.load('all_actors/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
        mfcclist.append(mfccs)

#create dataframe assign labels
df=pd.DataFrame(mfcclist)
df['Labels']=feeling_list

#Group samples based on number of sentiments
grouped = df.groupby('Labels')

#Find unique labels from feeling_list to visualize clusters
unique=set(feeling_list)

#Take mean of samples across classes. one row should represent one class 
df_mean=pd.DataFrame()

i=0
for label in unique:
  x=grouped.get_group(label)
 
  df_mean[i]=x.mean()
  
  i=i+1
df_mean.dropna(inplace=True)
#print(df_mean)
df_mean=df_mean.T
df_mean['Labels']=unique
print(df_mean)

#plot cluster for every sentiment and centroid
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('MFCC 0',fontsize=20)
plt.ylabel('MFCC 1',fontsize=20)
plt.title("Categorical analysis",fontsize=20)

targets=unique
number=len(targets)
cmap = plt.get_cmap('gnuplot')

colors = [cmap(i) for i in np.linspace(0, 1, number)]
for target, color in zip(targets,colors):
   
    indicesToKeep = df['Labels'] == target
    
    plt.scatter(df.loc[indicesToKeep, 0]
               , df.loc[indicesToKeep,1], c = color, s = 50)
    
for target, color in zip(targets,colors):
   
    indicesToKeep = df_mean['Labels'] == target
    
    plt.scatter(df_mean.loc[indicesToKeep, 0]
             , df_mean.loc[indicesToKeep,1], c = color, s = 650,marker='h')
#plt.scatter(df_mean.loc[:, 0]
 #          , df_mean.loc[:,1], c = 'cyan', s = 150, marker='X')

plt.legend(targets,prop={'size': 15})

plt.show()

#find intercluster distance across centroids
from_label=[]
to_label=[]
intercluster_distance=[]
for i in range(df_mean.shape[0]):
  for j in range(df_mean.shape[0]):
    from_label.append(df_mean.loc[i,'Labels'])
    to_label.append(df_mean.loc[j,'Labels'])
    intercluster_distance.append(distance.euclidean(df_mean.iloc[i,0:n_mfcc], df_mean.iloc[j,0:n_mfcc]))

intercluster_distance_arr=np.array(intercluster_distance).reshape(10,10)
intercluster_distance_df=pd.DataFrame(intercluster_distance_arr)
intercluster_distance_df.columns=unique
intercluster_distance_df.index=unique
intercluster_distance_df.style.background_gradient(cmap='Blues')

#find intracluster distance of every sample to its centroid
from_label=[]
to_label=[]
intracluster_distance=[]
for i in range(df.shape[0]):
  for j in range(df_mean.shape[0]):
    if(df.loc[i,'Labels']==df_mean.loc[j,'Labels']):
        intracluster_distance.append(distance.euclidean(df.iloc[i,0:n_mfcc], df_mean.iloc[j,0:n_mfcc]))

df['Intracluster distance']=intracluster_distance
df


