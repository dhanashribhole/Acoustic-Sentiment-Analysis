
'''LibROSA is a python package for music and audio analysis. 
It provides the building blocks necessary to create music information retrieval systems.'''

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
from scipy.spatial import distance
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import json
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from prettytable import PrettyTable

#Confusion matrix plotting
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

#Segmenting all 10 sentiments, confusion matrix
def all_sentiments(feeling_list,mfcc_array):
  labels=feeling_list
  opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
  
  df=pd.DataFrame(mfcc_array)
  x_train,y_train,x_test,y_test,lb=train_test_split(df,labels)
  
  unique=set(labels)
  op_len=len(unique)
  ip_len=len(x_train[0])
  #CNN
  
  modelcnn=model_CNN(ip_len,op_len)
  print(modelcnn.summary())
  modelcnn.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
  cnnhistory=modelcnn.fit(x_train, y_train, batch_size=16, epochs=700, validation_data=(x_test, y_test))
  
  plot_graph(cnnhistory)
  save_model(modelcnn)
  loaded_model=load_evaluate(modelcnn,x_train, y_train,x_test,y_test)
 
  #Predict emotions on test data
  preds = loaded_model.predict(x_test, 
                           batch_size=32, 
                           verbose=1)
  preds1=preds.argmax(axis=1)
  abc = preds1.astype(int).flatten()

  predictions = (lb.inverse_transform((abc)))

  preddf = pd.DataFrame({'predictedvalues': predictions})
  actual=y_test.argmax(axis=1)
  abc123 = actual.astype(int).flatten()
  actualvalues = (lb.inverse_transform((abc123)))

  actualdf = pd.DataFrame({'actualvalues': actualvalues})
  finaldf = actualdf.join(preddf)
  uniq=set(finaldf.iloc[:,0])
  
  #Save Actual Vs Predicted labels in predictions.csv
  
  finaldf.to_csv('Predictions.csv', index=False)
  actual = actualdf
  predicted = preddf
  
  #Plot confusion matrix
  
  results = confusion_matrix(actual, predicted) 
  
  plot_confusion_matrix(cm = results,
                      normalize    = False,
                      target_names = set(feeling_list),
                      title        = "Confusion Matrix")
  
  confusion_df=pd.DataFrame(results)
  confusion_df.columns=uniq
  confusion_df.index=uniq
  
  confusion_df_new=confusion_df.T
  
  #Check from observations ,if samples are getting misclassified based on genders or based on sentiments
  
  female_calm_total= confusion_df_new.female_calm.female_calm+confusion_df_new.female_calm.female_happy+confusion_df_new.female_calm.male_fearful+confusion_df_new.female_calm.female_angry+confusion_df_new.female_calm.male_calm+confusion_df_new.female_calm.male_sad+confusion_df_new.female_calm.female_sad+confusion_df_new.female_calm.male_happy+confusion_df_new.female_calm.female_fearful+confusion_df_new.female_calm.male_angry
  female_happy_total= confusion_df_new.female_happy.female_calm+confusion_df_new.female_happy.female_happy+confusion_df_new.female_happy.male_fearful+confusion_df_new.female_happy.female_angry+confusion_df_new.female_happy.male_calm+confusion_df_new.female_happy.male_sad+confusion_df_new.female_happy.female_sad+confusion_df_new.female_happy.male_happy+confusion_df_new.female_happy.female_fearful+confusion_df_new.female_happy.male_angry

  male_fearful_total= confusion_df_new.male_fearful.female_calm+confusion_df_new.male_fearful.female_happy+confusion_df_new.male_fearful.male_fearful+confusion_df_new.male_fearful.female_angry+confusion_df_new.male_fearful.male_calm+confusion_df_new.male_fearful.male_sad+confusion_df_new.male_fearful.female_sad+confusion_df_new.male_fearful.male_happy+confusion_df_new.male_fearful.female_fearful+confusion_df_new.male_fearful.male_angry
  female_angry_total= confusion_df_new.female_angry.female_calm+confusion_df_new.female_angry.female_happy+confusion_df_new.female_angry.male_fearful+confusion_df_new.female_angry.female_angry+confusion_df_new.female_angry.male_calm+confusion_df_new.female_angry.male_sad+confusion_df_new.female_angry.female_sad+confusion_df_new.female_angry.male_happy+confusion_df_new.female_angry.female_fearful+confusion_df_new.female_angry.male_angry

  male_calm_total= confusion_df_new.male_calm.female_calm+confusion_df_new.male_calm.female_happy+confusion_df_new.male_calm.male_fearful+confusion_df_new.male_calm.female_angry+confusion_df_new.male_calm.male_calm+confusion_df_new.male_calm.male_sad+confusion_df_new.male_calm.female_sad+confusion_df_new.male_calm.male_happy+confusion_df_new.male_calm.female_fearful+confusion_df_new.male_calm.male_angry
  male_sad_total= confusion_df_new.male_sad.female_calm+confusion_df_new.male_sad.female_happy+confusion_df_new.male_sad.male_fearful+confusion_df_new.male_sad.female_angry+confusion_df_new.male_sad.male_calm+confusion_df_new.male_sad.male_sad+confusion_df_new.male_sad.female_sad+confusion_df_new.male_sad.male_happy+confusion_df_new.male_sad.female_fearful+confusion_df_new.male_sad.male_angry

  female_sad_total= confusion_df_new.female_sad.female_calm+confusion_df_new.female_sad.female_happy+confusion_df_new.female_sad.male_fearful+confusion_df_new.female_sad.female_angry+confusion_df_new.female_sad.male_calm+confusion_df_new.female_sad.male_sad+confusion_df_new.female_sad.female_sad+confusion_df_new.female_sad.male_happy+confusion_df_new.female_sad.female_fearful+confusion_df_new.female_sad.male_angry
  male_happy_total= confusion_df_new.male_happy.female_calm+confusion_df_new.male_happy.female_happy+confusion_df_new.male_happy.male_fearful+confusion_df_new.male_happy.female_angry+confusion_df_new.male_happy.male_calm+confusion_df_new.male_happy.male_sad+confusion_df_new.male_happy.female_sad+confusion_df_new.male_happy.male_happy+confusion_df_new.male_happy.female_fearful+confusion_df_new.male_happy.male_angry

  female_fearful_total= confusion_df_new.female_fearful.female_calm+confusion_df_new.female_fearful.female_happy+confusion_df_new.female_fearful.male_fearful+confusion_df_new.female_fearful.female_angry+confusion_df_new.female_fearful.male_calm+confusion_df_new.female_fearful.male_sad+confusion_df_new.female_fearful.female_sad+confusion_df_new.female_fearful.male_happy+confusion_df_new.female_fearful.female_fearful+confusion_df_new.female_fearful.male_angry
  male_angry_total= confusion_df_new.male_angry.female_calm+confusion_df_new.male_angry.female_happy+confusion_df_new.male_angry.male_fearful+confusion_df_new.male_angry.female_angry+confusion_df_new.male_angry.male_calm+confusion_df_new.male_angry.male_sad+confusion_df_new.male_angry.female_sad+confusion_df_new.male_angry.male_happy+confusion_df_new.male_angry.female_fearful+confusion_df_new.male_angry.male_angry
  
  #Actual Vs predicted for female sad and male sad
  
  x = PrettyTable()

  x.field_names = ["Type","Female sad", "Male sad", "Others"]

  x.add_row(["Female sad", confusion_df_new.female_sad.female_sad, confusion_df_new.female_sad.male_sad, female_sad_total-(confusion_df_new.female_sad.female_sad+confusion_df_new.female_sad.male_sad)])
  x.add_row(["Male sad", confusion_df_new.male_sad.female_sad, confusion_df_new.male_sad.male_sad, male_sad_total-(confusion_df_new.male_sad.female_sad+confusion_df_new.male_sad.male_sad)])

  print("Actual Vs Predicted")
  print(x)  
  
  #Actual Vs predicted for Female happy and Male happy
  
  x = PrettyTable()

  x.field_names = ["Type","Female happy", "Male happy", "Others"]

  x.add_row(["Female happy", confusion_df_new.female_happy.female_happy, confusion_df_new.female_happy.male_happy, female_happy_total-(confusion_df_new.female_happy.female_happy+confusion_df_new.female_happy.male_happy)])
  x.add_row(["Male happy", confusion_df_new.male_happy.female_happy, confusion_df_new.male_happy.male_happy, male_happy_total-(confusion_df_new.male_happy.male_happy+confusion_df_new.male_happy.female_happy)])

  print("Actual Vs Predicted")
  print(x)
  
  #Actual Vs predicted for Female angry and Male angry
  
  x = PrettyTable()

  x.field_names = ["Type","Female angry", "Male angry", "Others"]

  x.add_row(["Female angry", confusion_df_new.female_angry.female_angry, confusion_df_new.female_angry.male_angry, female_angry_total-(confusion_df_new.female_angry.female_angry+confusion_df_new.female_angry.male_angry)])
  x.add_row(["Male angry", confusion_df_new.male_angry.female_angry, confusion_df_new.male_angry.male_angry, male_angry_total-(confusion_df_new.male_angry.male_angry+confusion_df_new.male_angry.female_angry)])

  print("Actual Vs Predicted")
  print(x)
  
  #Actual Vs predicted for Female calm and Male calm
     
  x = PrettyTable()

  x.field_names = ["Type","Female calm", "Male calm", "Others"]

  x.add_row(["Female calm", confusion_df_new.female_calm.female_calm, confusion_df_new.female_calm.male_calm, female_calm_total-(confusion_df_new.female_calm.female_calm+confusion_df_new.female_calm.male_calm)])
  x.add_row(["Male calm", confusion_df_new.male_calm.female_calm, confusion_df_new.male_calm.male_calm, male_calm_total-(confusion_df_new.male_calm.male_calm+confusion_df_new.male_calm.female_calm)])

  
  print("Actual Vs Predicted")
  print(x)
  
  #Actual Vs predicted for Female fearful and Male fearful
     
  x = PrettyTable()

  x.field_names = ["Type","Female fearful", "Male fearful", "Others"]

  x.add_row(["Female fearful", confusion_df_new.female_fearful.female_fearful, confusion_df_new.female_fearful.male_fearful, female_fearful_total-(confusion_df_new.female_fearful.female_fearful+confusion_df_new.female_fearful.male_fearful)])
  x.add_row(["Male fearful", confusion_df_new.male_fearful.female_fearful, confusion_df_new.male_fearful.male_fearful, male_fearful_total-(confusion_df_new.male_fearful.male_fearful+confusion_df_new.male_fearful.female_fearful)])

  
  print("Actual Vs Predicted")
  print(x)



#Splits the samples in the dataset into train set and test set
def train_test_split(df,feeling_list):
  labels = pd.DataFrame(feeling_list)
  newdf = pd.concat([df,labels], axis=1)
  
  training_samples=round(df.shape[0]*0.8)
  rnewdf=newdf
  rnewdf = shuffle(rnewdf)
  rnewdf.shape
  rnewdf=rnewdf.fillna(0)
  #newdf1 = np.random.rand(len(rnewdf)) < 0.7
  train = rnewdf[0:training_samples]
  test = rnewdf[training_samples:]
  
  trainfeatures = train.iloc[:, :-1]
  trainlabel = train.iloc[:, -1:]
  testfeatures = test.iloc[:, :-1]
  testlabel = test.iloc[:, -1:]
  X_train = np.array(trainfeatures)
  y_train = np.array(trainlabel)
  X_test = np.array(testfeatures)
  y_test = np.array(testlabel)

  lb = LabelEncoder()
  
  #Perform One hot encoding 
  
  y_train = np_utils.to_categorical(lb.fit_transform(y_train))
  y_test = np_utils.to_categorical(lb.fit_transform(y_test))
  x_traincnn =np.expand_dims(X_train, axis=2)
  x_testcnn= np.expand_dims(X_test, axis=2)
  
  return x_traincnn,y_train,x_testcnn,y_test,lb

#Sentiment analysis : positive and negative emotions
#Positive emotion bucket : calm,happy
#Negative emotion bucket : sad,fearful,angry



#CNN model initializing
def model_CNN(inputlen,outputlen):
  
  model = Sequential()

  model.add(Conv1D(256, 5,padding='same',
                   input_shape=(inputlen,1)))
  model.add(Activation('relu'))
  model.add(Conv1D(128, 5,padding='same'))
  model.add(Activation('relu'))
  model.add(Dropout(0.1))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(128, 5,padding='same',))
  model.add(Activation('relu'))
 
  model.add(Conv1D(128, 5,padding='same',))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(outputlen))
  model.add(Activation('softmax'))
 
  
  return model

#Save the model in the directory

def save_model(model):
  model_name = 'Emotion_Voice_Detection_Model.h5'
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  
  # Save model and weights
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

def plot_graph(cnnhistory):
  plt.plot(cnnhistory.history['loss'])
  plt.plot(cnnhistory.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

#Evaluate the model 
def load_evaluate(model,x_traincnn, y_train,x_testcnn,y_test):
  opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  
  # load weights into new model
  loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
  print("Loaded model from disk")

  # evaluate loaded model on test data and train data
  loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  print("Train Accuracy :")
  score = loaded_model.evaluate(x_traincnn,y_train, verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
  print("Test Accuracy :")
  score = loaded_model.evaluate(x_testcnn,y_test, verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
  return loaded_model

from google.colab import drive
drive.mount('/content/drive')
!ls

#Load all samples from folder in mylist
cwd = os.getcwd()
mylist= os.listdir('drive/My Drive/all_actors')

# Labelling ten classes
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
    elif item[:2]=='sa':
        feeling_list.append('male_sad')

#setting number of mfcc coefficient per frame to 13.
n_mfcc=13

#Extracting the MFCC features for all ten classes
'''MFCC as feature : For audio analysis, the shape of the vocal tract manifests itself in the envelope of the short time power spectrum, 
and the feature which accurately represent this envelope is MFCC.
Mel Frequency Cepstral Coefficents (MFCCs) is the feature widely used in automatic speech and speaker recognition.
sampling rate=44.1KHz is and duration of 2.5 sec of audio has been considered while loading package through librosa'''

#In our analysis 13 MFCC across every frame of sample is used 
mfcclist=[]
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
        X, sample_rate = librosa.load('drive/My Drive/all_actors/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc)
        mfcclist.append(mfccs)

#Convert 2D array of mfccs to 1D array
df_reshape=[]
for data in mfcclist:
 
  d = np.reshape(data,-1).tolist()
  df_reshape.append(d)

#Exclude MFCC having less dimension it supposed to be
list_mfcc=[]
for data, i in zip(df_reshape,range(len(df_reshape))):
  if len(data) == len(df_reshape[0]):
    
    list_mfcc.append(data)
  else:
    index_exclude=i
len(list_mfcc[0])
mfcc_array = np.array(list_mfcc)
mfcc_array.shape

#Exclude label corresponding to excluded MFCC index
feeling_list.pop(index_exclude)



#Segment 5 sentiments across 2 genders (10 classes)
all_sentiments(feeling_list,mfcc_array)



