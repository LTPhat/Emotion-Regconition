# Emotion-Regconition

A project applying Keras to build CNN model and OpenCV library to detect emotion through webcam.
## About the dataset
- The data consists of 48x48 pixel grayscale images of faces. 

- The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

- The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise).

- The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
- Link to dataset: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
## Notebook
- Install Kaggle.
```sh
!pip install -q kaggle

```
- Upload API dataset in json file from local.
```sh
Saving kaggle.json to kaggle.json
{'kaggle.json': b'{"username":"ltp0203","key":"fffdf39b9c00572aa906e7402280150a"}'}

```
- Create kaggle folder
```sh
! mkdir ~/.kaggle
```
- Copy the kaggle.json to folder created.
```sh
! cp kaggle.json ~/.kaggle/
```
- Permission for the json to act
```sh
! chmod 600 ~/.kaggle/kaggle.json
```
- Download dataset from kaggle FER2013
```sh
! kaggle datasets download -d msambare/fer2013
```
- All steps of building model are included in **Emojify.ipynb**.
## Training 
Training from local is saved in **train.py**.
## Model's weights
Saved in **FER_model_weight.h5**
## Main 
Saved in **emojify.py**
## Result 
Saved in **capture** folder.
