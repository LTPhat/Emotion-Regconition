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


emotion_list = {0: "Angry",1:"Disgust",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}

# Define current file's path
cur_path = os.path.dirname(os.path.abspath(__file__))

# Define emotion images directory
emoji_dist = {0: cur_path + "/emotion/angry.png",1: cur_path + "/emotion/disgust.png",2: cur_path+ "/emotion/fear.png",
            3: cur_path +  "/emotion/happy.png",4: cur_path +  "/emotion/neutral.png", 
            5: cur_path+ "/emotion/sad.png",6: cur_path+  "/emotion/surprise.png" }
global last_frame1
last_frame1 = np.zeros((480,640,3) , dtype = np.uint8)
global cap1
show_text = [0]
global frame_number

# Function to capture real time image and predict
def show_subject():
    cap1 = cv2.VideoCapture(0)
    if not cap1:
        print("Can't open the webcam")
    global frame_number
    # Count frame in realtime video
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number >= length:
        exit()
    cap1.set(1,frame_number)
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize((600,500))
    bounding_box = cv2.CascadeClassifier(r"C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    #Convert to gray frame
    gray_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scale_factor = 1.3, minNeighbors = 5)
    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame1,(x,y-50),(x+w,y+h+10),(255,0,0),2)
        roi_gray_frame = gray_frame[y:y+h,x:x+w]
        # Crop input image to correct shape of input_shape in model
        croppe_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)
        #predict image
        prediction = model.predict(croppe_img)
        # index of class
        max_index = int(np.argmax(prediction))
        cv2.putText(frame1,emotion_list[max_index],(x+20,y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2, cv2.LINE_AA)
        show_text[0]=max_index
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


# Function to show avatar emotion
def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text = emoji_dist[show_text[0]],font=("arial",45,"bold"))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10,show_avatar)

if __name__ == "__main__":
    # Create GUI 
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3 = tk.Label(master=root,bd=10,fg = "#CDCDCD",bg = "black")
    lmain.pack(side = LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side = RIGHT)
    lmain2.place(x=900,y=350)

    root.title('Emotion Recognition')
    root.geometry("1400x900+100+10")
    root['bg']="black"
    exitButton = Button(root,text="Quit",fg = "red",command= root.destroy,font= ("Arial",25,"bold")).pack(side = BOTTOM)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()
cv2.destroyAllWindows()