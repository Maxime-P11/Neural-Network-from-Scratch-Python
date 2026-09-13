This is a neural network from scratch
the project uses :__
the class:__
numpy__
pickle__

the mnist training:__
numpy__
pickle__
os__

Number draw:__
numpy__
os__
pygame__
cv2__
math__




NN_Numpy_Class.py is a neural network class it can be used on other project it is not linked to the mnist dataset, the main functions to use it are:__
train_batch, to train it input a x by n numpy array, x being the batch size and n the inputs and the same for the targets array__
test_batch to get the loss of a batch on a test dataset__
forward : to use the trained model, you must feed it a 1 by n numpy array and it return a 1 by n output array__

model.pkl is the pretrained model on the mnist dataset__

Number_draw.py is is to test the model :__
with the mouse you can draw a number and it will show the prediction of the model__
press any key on the keyboard to clear the screen__
it use some preprossecing to mimic mnist prepossecing, and to be resitent to translation and scaling__

NN_Numpy_train.py train the model it uses:__
mnist_test.csv, to load it it's take a while, so it will create a pkl file of the inputs automaticly__
IMPORTANT:__
it also uses mnist_train.csv, but it was to big to fit here, but you can find it on kaggle : https://www.kaggle.com/datasets/oddrationale/mnist-in-csv, or by other mean__
like mnist_test.csv it also create a pkl file for faster loading__
while training you can quit it at any time, but prefferably after it prints : 'model saved' to avoid corruption of the model__
when you relauch the file, it will resume wwhere it left off__

if you want to use it for an other dataset you will need to do :__
1) delete model.pkl or rename it__
2) create training loop from scratch or modify : readcsv and getlabelandinput in NN_Numpy_train.py__
