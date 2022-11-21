from shape_nn.feed_forward_nn import Feedforward
from typing import List
import torch

def train_network(x_data = List, y_data = List, epoch = int, hidden_size=300) -> List:

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

    # Initialising the FeedFoward neural network from the class definition
    shape_model = Feedforward(2, hidden_size=hidden_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(shape_model.parameters(), lr=0.01)

    # Testing the model performance before training
    shape_model.eval()
    y_pred = shape_model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())

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
    
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    # Performance after training
    shape_model.eval()
    y_test_pred = shape_model(x_test)
    after_train = criterion(y_test_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())

    for i in range(len(collect_train_pred)):
        data_epoch = collect_train_pred[i].tolist()
        collect_train_pred[i] = [[data_epoch[i][0] for i in range(num_of_train)], [data_epoch[i][1] for i in range(num_of_train)]]

    return collect_train_pred

