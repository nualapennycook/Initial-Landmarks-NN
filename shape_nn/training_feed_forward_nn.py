from shape_nn.feed_forward_nn import Feedforward
from typing import List
import torch

def train_network(x_data = List, y_data = List, epoch = int, hidden_size=300, learning_rate=0.01) -> List:
    '''
    Function to train a neural network on a data set. 
    Takes data as input, and trains and tests on the set.
    Calls invertible_nn to implement the invertible neural network for the training.
    :param x_data: List, The independent variables, i.e. the data to input to the network.
    :param y_data: List, The dependent variables, i.e. the data to compare the network output to.
    :param epoch: int, The number of training epochs (loops).
    :param hidden_size: int, The number of neurons in the hidden layers of the network.
    :param learning_rate: float, learning rate of the algorithm.
    :out: Training landmarks, list of the training results from each training epoch, test landmarks, and the training loss and test loss.
    '''

    # Splitting the data into test and train sets
    # ~90% data for training and ~10% for testing
    # Choosing test points evenly through the data points
    num_of_test = int(len(x_data)/10)
    num_of_train = len(x_data) - num_of_test
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(num_of_test):
        x_test += [x_data[i*10]]
        x_train += x_data[i*10+1:(i+1)*10]
        y_test += [y_data[i*10]]
        y_train += y_data[i*10+1:(i+1)*10]

    training_landmarks = y_train

    # Now convert x and y to pytorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    # Initialising the FeedFoward neural network from the class definition
    shape_model = Feedforward(2, hidden_size=hidden_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(shape_model.parameters(), lr=learning_rate)

    # Testing the model performance before training
    shape_model.eval()
    y_pred = shape_model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item()/(len(x_test)))

    collect_train_pred = []

    # Training the model
    shape_model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_train_pred = shape_model(x_train)
        collect_train_pred.append(y_train_pred)
        # Compute Loss
        loss = criterion(y_train_pred.squeeze(), y_train)

        train_loss = loss.item()/num_of_train
        print('Epoch {}: train loss: {}'.format(epoch, train_loss))
        # Backward pass
        loss.backward()
        optimizer.step()

    # Performance after training
    shape_model.eval()
    y_test_pred = shape_model(x_test)
    after_train = criterion(y_test_pred.squeeze(), y_test) 
    test_loss = after_train.item()/(len(x_test))
    print('Test loss after Training' , test_loss)

    # Converting the translated training points from a tensor to a list to output
    for i in range(len(collect_train_pred)):
        data_epoch = collect_train_pred[i].tolist()
        collect_train_pred[i] = [[data_epoch[i][0] for i in range(num_of_train)], [data_epoch[i][1] for i in range(num_of_train)]]

    # Translated test points
    y_test_pred = y_test_pred.tolist()
    y_test_pred = [[y_test_pred[i][0] for i in range(num_of_test)], [y_test_pred[i][1] for i in range(num_of_test)]]

    return training_landmarks, collect_train_pred, y_test_pred, [train_loss, test_loss]

