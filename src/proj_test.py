import random

from config import LEARNING_RATE, NUM_TRAINING_ITERATIONS, CONVERGENCE_THRESHOLD
from src.formulas import err
from src.models import Layer, cfile

f = None
curr_point = 0
target = []
attrs = []
total_runs = 0
data = None
num_incorrect = 0
prev_sample_err = 0
curr_sample_err = 0

def parse_data(fname):
    # reset all the data
    global curr_point
    global total_runs
    global target
    global attrs
    global num_incorrect
    global prev_sample_err
    global curr_sample_err
    global data
    global f

    curr_point = 0
    total_runs = 0
    target = []
    attrs = []
    num_incorrect = 0
    prev_sample_err = 0
    curr_sample_err = 0

    # set the proper data file
    data_file = 'err.txt'
    if fname == 'training.txt':
        data_file = 'training_err.txt'
    elif fname == 'validation.txt':
        data_file = 'val_err.txt'
    elif fname == 'testing.txt':
        data_file = 'testing_err.txt'

    # clear the file
    open(data_file, 'w+').close()

    # open the data file for logging
    data = cfile(data_file, 'w')

    f = open(fname, 'r').readlines()

    for row in f:
        row = [x.strip() for x in row.split(',')]
        row = [int(num) for num in row]
        target.append(int(row[0]))  # row[0] represents the target class
        attrs.append(row[1:])  # row[1:] represents the attributes

if __name__ == '__main__':
    print("Parsing the training dataset...")
    # parse the training dataset and store its information into globals
    parse_data('training.txt')

    # set up the layers to be used
    x = Layer(6, attrs[curr_point], 1)
    y = Layer(3, x.layer_out, 2)

    print("Begin training the neural network:")
    # iterate through to train the neural network
    while total_runs < NUM_TRAINING_ITERATIONS:

        # set up the first layer and evaluate it
        x.input_vals = attrs[curr_point]
        x.eval()

        # set up the second layer and evaluate it
        y.input_vals = x.layer_out
        y.eval()

        # backpropagate
        y.backprop(target[curr_point])
        x.backprop(y)

        # get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down to check err
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0


        # increment the number incorrect if it's wrong
        if temp != target[curr_point]:
            num_incorrect += 1

        # check to see if we have converged
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print("Data has converged at the " + str(total_runs) + "th run.")
                break

        # print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # iterate
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # close the file
    data.close()

    print("Neural network is done training! Hit enter to validation processing.")
    accuracy = 1 - (float(num_incorrect) / NUM_TRAINING_ITERATIONS)
    print("Accuracy on the training set: " + str(accuracy))
    print("Error percentage on the training set: " + str(float(num_incorrect) / NUM_TRAINING_ITERATIONS))
    input()

    print("Parsing the validation dataset...")
    # parse the validation dataset and store its information into globals
    parse_data('validation.txt')

    print("Begin validating the neural network:")
    # iterate through to validate the neural network
    while total_runs < len(f):

        # set up the first layer and evaluate it
        x.input_vals = attrs[curr_point]
        x.eval()

        # set up the second layer and evaluate it
        y.input_vals = x.layer_out
        y.eval()

        # get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down to check err
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # increment the number incorrect if it's wrong
        if temp != target[curr_point]:
            num_incorrect += 1

        # check to see if we have converged
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print("Data has converged at the " + str(total_runs) + "th run.")
                break

        # print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # iterate
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # close the file
    data.close()

    print("Neural network is done validating! Hit enter to test it.")
    accuracy = 1 - (float(num_incorrect) / len(f))
    print("Accuracy on the validation set: " + str(accuracy))
    print("Error percentage on the validation set: " + str(float(num_incorrect) / len(f)))
    input()

    print("Begin testing the neural network:")
    # parse the testing data and store its information into globals
    parse_data('testing.txt')

    # iterate through to test the neural network
    while curr_point < len(f):

        # set up the first layer and evaluate it
        x.input_vals = attrs[curr_point]
        x.eval()

        # set up the second layer and evaluate it
        y.input_vals = x.layer_out
        y.eval()

        # get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down to check err
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # increment the number incorrect if it's wrong
        if temp != target[curr_point]:
            num_incorrect += 1

        # print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current Error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # iterate
        total_runs += 1
        curr_point += 1

    data.close()
    print("Testing done! Check out the generated output files ('testing_err.txt' and 'training_err.txt')")
    accuracy = 1 - (float(num_incorrect) / len(f))
    print("Accuracy on the testing set: " + str(accuracy))
    print("Error percentage on the testing set: " + str(float(num_incorrect) / len(f)))
