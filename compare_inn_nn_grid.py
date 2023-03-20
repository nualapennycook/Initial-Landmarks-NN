from shape_nn.feed_forward_nn import Feedforward
import matplotlib.pyplot as plt
import torch
from shape_nn.invertible_nn import Invertible

grid_resolution = 20


def compute_loss(epoch):
    x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

    grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
    grid_points_y = grid_points_x.copy()

    plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
    plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

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
    x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

    grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
    grid_points_y = grid_points_x.copy()

    plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
    plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

    grid_points = [[plotting_grid_points_x[i], plotting_grid_points_y[i]] for i in range(len(plotting_grid_points_x))]

    # Initialising the FeedFoward neural network from the class definition
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
    number_of_epochs = [100, 500, 1000, 1500, 2000]
    train_loss = [0 for epoch in number_of_epochs]
    train_inn_loss = [0 for epoch in number_of_epochs]
    for i in range(len(number_of_epochs)):
        train_loss[i] = compute_loss(number_of_epochs[i])
        train_inn_loss[i] = compute_inn_loss(number_of_epochs[i])
    fig, ax = plt.subplots()
    plt.plot(number_of_epochs, train_loss)
    plt.plot(number_of_epochs, train_inn_loss)
    plt.legend(['Feed forward loss', 'Invertible loss'])
    plt.show()




if __name__ == '__main__':
    main()

