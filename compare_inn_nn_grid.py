from shape_nn.feed_forward_nn import Feedforward
import matplotlib.pyplot as plt
import torch
from shape_nn.invertible_nn import Invertible

'''
Script to compare the result of image registration using a feedforward and an invertible neural network.
Uses the warp grid example.
Produces a plot comparing the training losses for feedforward and invertible neural networks.
'''

# initialising the grid resolution for the problem
grid_resolution = 20


def compute_loss(epoch):
    '''
    Computing the training loss of the warp grid problem using a feedforward neural network.
    :param epoch: int, the number of epochs of the neural network.
    '''
    # Assigning the landmark data points
    x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

    # Construcing the grid of test points
    grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
    grid_points_y = grid_points_x.copy()
    
    plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
    plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

    # Reshaping the grid points into coordinate pairs for input to the neural network
    grid_points = [[plotting_grid_points_x[i], plotting_grid_points_y[i]] for i in range(len(plotting_grid_points_x))]

    # Initialising the FeedFoward neural network from the class definition
    shape_model = Feedforward(2, hidden_size=300)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shape_model.parameters(), lr=0.01)

    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)
    grid_tensor = torch.FloatTensor(grid_points)

    collect_train_pred = []
    collect_warped_grid = []

    # Training the model
    shape_model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_train_pred = shape_model(x_train)
        warped_grid_step = shape_model(grid_tensor)
        collect_warped_grid.append(warped_grid_step)
        collect_train_pred.append(y_train_pred)
        # Compute Loss
        loss = criterion(y_train_pred.squeeze(), y_train)

        train_loss = loss.item()
        print('Epoch {}: train loss: {}'.format(epoch, train_loss))
        # Backward pass
        loss.backward()
        optimizer.step()

    return train_loss

def compute_inn_loss(epoch):
    '''
    Computing the training loss of the warp grid problem using an invertible neural network.
    :param epoch: int, the number of epochs of the neural network.
    '''
    # Assigning the landmark data points
    x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

    # Construcing the grid of test points
    grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
    grid_points_y = grid_points_x.copy()

    plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
    plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

    grid_points = [[plotting_grid_points_x[i], plotting_grid_points_y[i]] for i in range(len(plotting_grid_points_x))]

    # Initialising the invertible neural network from the class definition
    shape_model = Invertible(2, hidden_size=50, number_of_blocks=2)
    shape_model.initialise_inn()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shape_model.parameters(), lr=0.05)

    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)
    grid_tensor = torch.FloatTensor(grid_points)

    collect_train_pred = []
    collect_warped_grid = []

    # Training the model
    shape_model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_train_pred, dump = shape_model.inn(x_train)
        warped_grid_step, dump = shape_model.inn(grid_tensor)
        collect_warped_grid.append(warped_grid_step)
        collect_train_pred.append(y_train_pred)
        # Compute Loss
        loss = criterion(y_train_pred.squeeze(), y_train)
    
        train_loss = loss.item()
        print('Epoch {}: train loss: {}'.format(epoch, train_loss))
        # Backward pass
        loss.backward()
        optimizer.step()

    return train_loss

def main():
    '''
    Plots the training error for both feedforward and invertible neural networks.
    '''
    # Generating the loss/error for a range of epochs
    number_of_epochs = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    train_loss = [0 for epoch in number_of_epochs]
    train_inn_loss = [0 for epoch in number_of_epochs]
    for i in range(len(number_of_epochs)):
        train_loss[i] = compute_loss(number_of_epochs[i])
        train_inn_loss[i] = compute_inn_loss(number_of_epochs[i])

    # Plotting the loss
    fig, ax = plt.subplots()
    plt.loglog(number_of_epochs, train_loss)
    plt.loglog(number_of_epochs, train_inn_loss)
    plt.legend(['Feed forward loss', 'Invertible loss'])
    plt.title('Comparison of Convergence of Feed Forward and Invertible Neural Networks')
    plt.savefig('plots/compare_warp_grid_loss.png')
    plt.show()


if __name__ == '__main__':
    main()

