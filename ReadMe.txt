
------------------------------------------------------------------------------------------------------------------------
***Implementation Details***:

split_data.py: This script uses sklearn train_test_split function to split the data set into training, validation, and testing data sets.
The data set is split into 60% training data, 20% validation data, and 20% testing data.
It also uses pandas.get_dummies() to convert the data set to one hot binary file and save the data sets in the following files:
training.txt
validation.txt
testing.txt

config.py: This script contains the following variables: NUM_TRAINING_ITERATIONS, LEARNING_RATE, CONVERGENCE_THRESHOLD.

models.py: This script contains the following functions: evaluation and backpropagation. Also contains the following classes: Layer and Cfile.
The evaluation function takes the input data and the weights and biases of the network and returns the output of the network.
The backpropagation function takes the input data, the weights and biases of the network, and the learning rate and returns the updated weights and biases of the network.
The Cfile class takes the file name as input and has the following functions:
write_to_file: This function takes the data as input and writes the data to the file.

formulas.py: This script contains the following functions: sigmoid, sigmoid_derivative, error, and error_derivative.
The sigmoid function takes the input data and returns the sigmoid of the input data.
The sigmoid_derivative function takes the input data and returns the derivative of the sigmoid of the input data.
The error function takes the output of the network and the target data and returns the error of the network.
The error_derivative function takes the output of the network and the target data and returns the derivative of the error of the network.

proj_test.py: This script contains the following functions: train, validation, test, and main.
The train function takes the training data, the validation data, the testing data, the number of epochs(NUM_TRAINING_ITERATIONS) , the learning rate, and the number of hidden layers and returns the weights and biases of the network.
The validation function takes the validation data, the weights and biases of the network, and the number of hidden layers and returns the accuracy of the network.
The test function takes the testing data, the weights and biases of the network, and the number of hidden layers and returns the accuracy of the network.
The main function takes the number of hidden layers as input and calls the train and test functions and prints the accuracy of the network.

Steps to run the code:

make sure the following files are present in the same directory as the proj_test.py file:
1) agaricus-lepiota.data
2) split_data.py
3) requirements.txt
4) config.py
5) models.py
6) formulas.py
7) proj_test.py


1)Install all the required packages using the following command:
pip install -r requirements.txt

2) To split the data set into training, validation, and testing data sets, run the following command:
python split_data.py

3) To run the code, run the following command:
python proj_test.py


------------------------------------------------------------------------------------------------------------------------
Make sure the following files are present in the same directory as the MLP_classifier_model.py file:
1) train(visual).csv

To run the MLP_classifier_model, run the following command:
python MLP_classifier_model.py






