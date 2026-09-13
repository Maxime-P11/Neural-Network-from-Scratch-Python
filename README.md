This is a neural network from scratch
the project uses :
the class:
numpy
pickle

the mnist training:
numpy
pickle
os

Number draw:
numpy
os
pygame
cv2
math




NN_Numpy_Class.py is a neural network class it can be used on other project it is not linked to the mnist dataset, the main functions to use it are:
train_batch, to train it input a x by n numpy array, x being the batch size and n the inputs and the same for the targets array
test_batch to get the loss of a batch on a test dataset
forward : to use the trained model, you must feed it a 1 by n numpy array and it return a 1 by n output array

model.pkl is the pretrained model on the mnist dataset

Number_draw.py is is to test the model :
with the mouse you can draw a number and it will show the prediction of the model
press any key on the keyboard to clear the screen
it use some preprossecing to mimic mnist prepossecing, and to be resitent to translation and scaling

NN_Numpy_train.py train the model it uses:
mnist_test.csv, to load it it's take a while, so it will create a pkl file of the inputs automaticly
IMPORTANT:
it also uses mnist_train.csv, but it was to big to fit here, but you can find it on kaggle : https://www.kaggle.com/datasets/oddrationale/mnist-in-csv, or by other mean
like mnist_test.csv it also create a pkl file for faster loading
while training you can quit it at any time, but prefferably after it prints : 'model saved' to avoid corruption of the model
when you relauch the file, it will resume wwhere it left off

if you want to use it for an other dataset you will need to do :
1) delete model.pkl or rename it
2) create training loop from scratch or modify : readcsv and getlabelandinput in NN_Numpy_train.py
