import numpy as np
from os import path
import pickle
from NN_Numpy_class import NNFromScratch


    
def readcsv(output_size, directory_dataset):
    with open(directory_dataset,"r") as file:
        file = file.readlines()

    inputs = []
    labels = []
    eye_array = np.eye(output_size, dtype=np.float32)
    for line in file:
        #remove new line and turn string into array
        values = str.split(line.replace('\n', ''),',')
        values = np.array(values, dtype=np.float32)
        #convert from 0-255 to 0-1
        inputs.append(values[1:]/255)

        labels.append(eye_array[int(values[0])])
    return inputs, labels

def getlabelandinput(directory, dir_dataset='mnist_train.csv'):
    if path.exists(directory):
        print('loading inputs and labels from premade pkl file')
        with open(directory,'rb') as f:
            inputs_all, labels_all = pickle.load(f)

    else:
        print('loading inputs and labels from csv file please wait')
        inputs_all, labels_all = readcsv(output_size = 10, directory_dataset = path.join(CURRENTDIR,dir_dataset))
        inputs_all, labels_all = np.array(inputs_all), np.array(labels_all)

        with open(directory,'wb') as f:
            pickle.dump((inputs_all, labels_all), f)

    return inputs_all, labels_all


if __name__ == '__main__':

    CURRENTDIR = path.dirname(path.realpath(__file__))
    
    inputs_labels_dir = path.join(CURRENTDIR,'input_labels_premade.pkl')
    inputs, labels = getlabelandinput(inputs_labels_dir)

    inputs_labels_dir_test = path.join(CURRENTDIR,'input_labels_premade_test.pkl')
    inputs_test, labels_test = getlabelandinput(inputs_labels_dir_test, 'mnist_test.csv')

    

    
    print('ready for training')

    FCNN = NNFromScratch(0.03, [784, 64, 32, 10])

    MODELSAVEPATH = path.join(CURRENTDIR,'model.pkl')
    if path.exists(MODELSAVEPATH):
        best_test_loss = FCNN.load_model(MODELSAVEPATH)
    else:
        best_test_loss = 1e15

    epoch = 0
    BATCH_SIZE = 99
    
    while True:

        current_batch_idx = 0
        loss = 0
        while current_batch_idx<len(inputs):

            loss += FCNN.train_batch(inputs = inputs[current_batch_idx:min(BATCH_SIZE+current_batch_idx, len(inputs))],
                                     labels = labels[current_batch_idx:min(BATCH_SIZE+current_batch_idx, len(inputs))])
            current_batch_idx += BATCH_SIZE



        epoch += 1
        print(f"epoch : {epoch}, loss : {loss}")
        if epoch%1==0:
            pass
            test_loss = FCNN.test_batch(inputs = inputs_test,
                                        labels = labels_test)
            print(f"epoch : {epoch}, test loss : {test_loss}")

            if test_loss<best_test_loss:
                FCNN.save_model(MODELSAVEPATH, best_test_loss)
                print('model saved')