import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import threading
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten,MaxPool2D,Conv2D,Dropout,MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras import models, layers, regularizers
from keras import regularizers
from keras.preprocessing import image
from keras.utils import img_to_array

#Redefineed model architecture

input_shape = (72,72,1)
num_classes = 7
model = models.Sequential()

# First
model.add(layers.Conv2D(32,kernel_size=(3,3),activation = 'relu', padding='same', input_shape = input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Second
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Third
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Fully connected
model.add(layers.Flatten())

# Input layer includes 1024 nodes
model.add(layers.Dense(512, activation='relu'))

# Hidden layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

new_model = keras.models.load_model('E:\EmojifyProject\FER_model.h5')

cv2.ocl.setUseOpenCL(False)

# Define emotion list
emotion_list = {0: "Angry",1:"Disgust",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}

# Define current file's path
cur_path = os.path.dirname(os.path.abspath(__file__))

# Define emotion images directory
emoji_dist = {0: cur_path + "/emotion/angry.png",1: cur_path + "/emotion/disgust.png",2: cur_path+ "/emotion/fear.png",
            3: cur_path +  "/emotion/happy.png",4: cur_path +  "/emotion/neutral.png", 
            5: cur_path+ "/emotion/sad.png",6: cur_path+  "/emotion/surprise.png" }


face_classifier = cv2.CascadeClassifier(r"C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
classifier =keras.models.load_model('E:\EmojifyProject\FER_model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



while True:
    #Capture video
    _, frame = cap.read()
    labels = []
    #Convert to 1 channel image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detect face
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        #Create rectangle bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),3)
        #Crop input image
        roi_gray = gray[y:y+h,x:x+w]
        #Resize to CNN input shape
        roi_gray = cv2.resize(roi_gray,(72,72),interpolation=cv2.INTER_AREA)
        # Successfully detect image
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        # No detect
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imshow('Emotion Regconition Tool',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
