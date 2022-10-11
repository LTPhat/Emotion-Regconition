# Emotion-Regconition

A project applying Keras to build CNN model and OpenCV library to detect emotion through camera and local web.

**Finished day: 15/09/2022.**
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
- **Samples from dataset**:
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/sample_training1.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/sample_training2.png)
- All steps of building model are included in **Emojify.ipynb**.
## Training 
Training from local is saved in **train.py**.
## Model's weights
Saved in **FER_model_weight.h5**
## Main 
Saved in **emojify.py**
## Test model
- **Prediction on some images in test set:**
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/predict1.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/predict2.png)

- Results on OpenCV webcam are saved in **capture** folder.
## Deploy model on Web using Flask
- Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.
- Jinja: By Ronacher, is a template engine for the Python programming language. Similar to the Django web framework, it handles templates in a sandbox.
- Install and update Flask
```sh
$ pip install -U Flask
```
- Define Flask app:
```sh
app = Flask(__name__)
```
- Run app:
```sh
if __name__ == '__main__':
    app.run(debug=True)
```
- Web style sheet: **static/style.css**.
- Default main site: **html_templates/index.html**.
- Prediction site: **html_templates/prediction.html**.
- Flask app: **app.py**.
- **Web Surface:**

![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/web_surface.png)
- **Some prediction of images uploaded from users:**

![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/happy_result2.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/happy_result.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/disgust_result.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/neutral_result.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/surprise_result.png)
![alt text](https://github.com/LTPhat/Emotion-Regconition/blob/main/image_web/sad_result.png)
