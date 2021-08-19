# project_3_facemasks

----
## Introduction
##### In this repository, we set out to train a model to classify images into three categories wearing a with_mask, without_mask, mask_weared_incorrect.

----
## Dataset
##### The dataset is from Kaggle (https://www.kaggle.com/vijaykumar1799/face-mask-detection). It contains a total of 8982 128 x 128 images of individuals evenly split into three directories mask_weared_incorrect, with_mask, and without_mask.

----
## model.ipynb
##### This is a jupyter notebook used to create, train, and save the model. It uses the libraries matplotlib, NumPy, os, PIL, TensorFlow, and pandas. The dataset is split into two parts, 80% training, and 20% validation. Following the split of the dataset, a model is created for image classification. The model is then fitted with 100 epochs and a callback function to stop early when validation lost is at its' minimum with a patience of 10. After the model is fitted the accuracy and loss data is put into a pandas dataframe and plotted. the model is then saved under the name my_model.h5.

----
## face_detection.py & prediction.ipynb
##### face_detection.py file utilizes dlib functions to detect faces within an image. The “shape_predictor_68_face_landmarks.dat” model file has been trained to recognize facial point annotations and we used this model file to assist dlib’s “get_frontal_face_detector” function.
##### By grabbing the upper and lower rectangle points among each identifiable face in the image, we can then individually crop the image by each occurrence and save to a folder which will later be used with our TensorFlow mask identifying model.
##### The user has the option of providing a web-based image from a URL or they can upload a local image file.
##### In predection.ipynb, we load the saved model, import face_detection.py, then use tensorflow's model.predict in a for loop to which iterates through the saved images to predict their classes and confidence percentage.  

----
## YouGov data and analysis
##### Tableau was used to do some analysis over data pulled from YouGov. The data was covering the percentage of people that wore mask out in public from different countries around the world. The first graph is just covering the percentage of people in USA that wore mask in public. The other three graphs cover different regions of the world and how they differ from each other. The analysis can be seen at https://rbutler2.github.io/project_3_facemasks/YouGov.html.

----
## Collaborators
##### Richard Butler, Jared Sanderson, Braxton Van Cleave, and Sean Waters.

----
## Github pages
##### https://rbutler2.github.io/project_3_facemasks/index.html
