from shape_nn.invertible_nn import Invertible
from typing import List
import torch

def train_network(x_data = List, y_data = List, epoch = int, hidden_size=300, number_of_blocks = 2, learning_rate=0.01) -> List:
    '''
    Function to train an invertible neural network on a data set. 
    Takes data as input, and trains and tests on the set.
    Calls invertible_nn to implement the invertible neural network for the training.
    :param x_data: List, The independent variables, i.e. the data to input to the network.
    :param y_data: List, The dependent variables, i.e. the data to compare the network output to.
    :param epoch: int, The number of training epochs (loops).
    :param hidden_size: int, The number of neurons in the hidden layers of the network.
    :param number_of_blocks: int, The number of subnetwork blocks of the invertible neural network.
    :param learning_rate: float, learning rate of the algorithm.
    :out: List, output data points, prediction values from the neural network for y values.
    '''

    # Splitting the data into test and train sets
    # ~80% data for training and ~20% for testing
    num_of_train = int(0.8*len(x_data))
    x_train = x_data[:num_of_train]
    x_test = x_data[num_of_train:]
    y_train = y_data[:num_of_train]
    y_test = y_data[num_of_train:]

    # Now convert x and y to pytorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    # Initialising the Invertible neural network from the class definition
    shape_model = Invertible(2, hidden_size=hidden_size, number_of_blocks=number_of_blocks)
    shape_model.initialise_inn()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shape_model.inn.parameters(), lr=learning_rate)

    # Testing the model performance before training
    # Output of the inn is a tuple of tensors, the latter being the log of the jacobian determinant
    # We don't have a use for it, so it is dumped
    y_pred, dump = shape_model.inn(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item()/(len(x_test)))

    collect_train_pred = []

    # Training the model
    shape_model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_train_pred, dump = shape_model.inn(x_train)
        collect_train_pred.append(y_train_pred)
        # Compute Loss
        loss = criterion(y_train_pred.squeeze(), y_train)
    
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()/num_of_train))
        # Backward pass
        loss.backward()
        optimizer.step()

    # Performance after training
    shape_model.eval()
    # Dump unused output
    y_test_pred, dump = shape_model.inn(x_test)
    after_train = criterion(y_test_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item()/(len(x_test)))

    for i in range(len(collect_train_pred)):
        data_epoch = collect_train_pred[i].tolist()
        collect_train_pred[i] = [[data_epoch[i][0] for i in range(num_of_train)], [data_epoch[i][1] for i in range(num_of_train)]]

    return collect_train_pred

