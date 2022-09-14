from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten,MaxPool2D,Conv2D,Dropout,MaxPooling2D,BatchNormalization
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras import models, layers, regularizers
from keras import regularizers
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
#--------------------------------------------
#PREPROCESSING
# Define train, test path
train_path = "E:\Emojify Project\train"
test_path = "E:\Emojify Project\test"

# create objects for Data Generation
# Some kinds of data augmentation
# zoom_range: thực hiện zoom ngẫu nhiên trong một phạm vi nào đó
# width_shift_range: Dịch theo chiều ngang ngẫu nhiên trong một phạm vi nào đó
# height_shift_range: Dịch ảnh theo chiều dọc trong một phạm vi nào đó
# brightness_range: Tăng cường độ sáng của ảnh trong một phạm vi nào đó.
# vertical_flip: Lật ảnh ngẫu nhiên theo chiều dọc
# rotation_range: Xoay ảnh góc tối đa là 45 độ
# shear_range: Làm méo ảnh
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1/255)

# Augmenting train and test
# directory: đặt đường dẫn có các classes của folder.
# target_size: là size của các ảnh input đầu vào, mỗi ảnh sẽ được resized theo kích thước này.
# color_mode: Nếu hình ảnh là màu đen và màu trắng hoặci là grayscale thì set "grayscale" hoặc nếu nó gồm 3 channels thì set "rgb"
# batch_size : Số lượng ảnh được yielded từ generator cho mỗi lô batch.
# class_mode : set "binary" nếu bạn có 2 classes để dự đoán, nếu không thì bạn set "categorical". trong trường hợp nếu bạn đang lập trình một hệ thống tự động Autoencoder, thì cả input và output đều là ảnh, trong trường hợp này thì bạn set là input
# shuffle: set True nếu bạn muốn đổi thứ tự hình ảnh, ngược lại set False.
# seed : Random seed để áp dụng tăng hình ảnh ngẫu nhiên và xáo trộn thứ tự của hình ảnh
train_set = train_datagen.flow_from_directory(train_path,target_size=(72,72),
                                              batch_size=64, color_mode="grayscale",
                                              class_mode = "categorical")
test_set=test_datagen.flow_from_directory(test_path,
                                             target_size=(72,72),
                                             batch_size=64,
                                             color_mode='grayscale',
                                             class_mode='categorical')

# ------------------------------------
#BUILD MODEL
# Define input_shape to CNN model
# img_rows, img_colums, color_channels
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VAL   = test_set.n//test_set.batch_size

result = model.fit(train_set, steps_per_epoch=STEP_SIZE_TRAIN, epochs=80, verbose=1, validation_data=test_set, validation_steps=STEP_SIZE_VAL)

model.save('E:\Emojify Project\FER_model.h5')
model.save_weights('E:\Emojify Project\FER_model_weight.h5')