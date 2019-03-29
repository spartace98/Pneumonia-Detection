# Pneumonia-Detection
To determine the presence of Pneumonia using Deep Neural Network models


This script aims to adopt state of the art deep learning models, specific image classification using convolution neural network (CNN) to predict the presence of pneumonia from X-ray images of patients lungs. 

I will be tuning the various hyperparameters of the model to dertermine the optimum layers and filters for the highest accuracy, while balancing the runtime required. 

The purpose of the research is to equip medical professionals with more accurate and reliable tool for pneumonia diagnosis, favouring the use of machine learning techniques over possible less reliable manual diagnosis by sight of doctors. 

Background
-
A research conducted by the United States National Institute of Health (NIH) revealed that among the 800 patients who were admitted to the Emergency Department (ED) for Community Acquired Pneumonia (CAP), 27.3% of them eventually had a non-pneumonia diagnosis upon discharge. While medical professionals have traditionally played a vital role in diagnosising and treating patients, it is dangerous to assume that that their opinions are infallible. Hence, experts today suggest that patients can look for another doctor to get a second opinion. 

My machine learning model aims to be that second opinion. My model was trained on 5216 patients X-Ray images, and validated on 624 patients. The model was trained at 50 epochs to ensure that the model is neither overfitting nor underfitting, and the model that leads to the highest validation accuracy was saved. It is pivotal that my model has a higher prediction accuracy that conventional doctors, otherwise the machine learning would be pointless. 

Results
-
My model was able to reach a peak of 93.1% prediction accuracy, which is significanly higher than the diagnosis accuracy of doctors (72.7%). 
Attached in this repository are the graphs of the training and validation accuracy and loss of my model. My model was ran on 50 epochs. 

I have also included my saved model that was able to attain the highest accuracy. It can be loaded using the code below:

model = load_model('my_model.h5')
