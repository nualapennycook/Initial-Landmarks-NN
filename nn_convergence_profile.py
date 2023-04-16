from shape_nn.feed_forward_nn import Feedforward
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

'''
Compares convergence of the warp grid example using a feedforward neural network using different gradient descent methods.
Plots the results of each registration for easy comparison.
'''

# Initialise the dimensions of the grid
grid_resolution = 20

# Initialise the format for the plots
fig, axs = plt.subplots(2, 2)

plot_spaces = [[0, 0], [0, 1], [1, 0], [1, 1]]

# The gradient descent methods used
plot_titles = ['Resilient Backpropagation', 'RMSprop', 'Stochastic Gradient Descent', 'Adam']

def main():
    '''
    Computes and plots the warp grid registrations for each gradient descent method
    '''
    for grid_square in range(4):

        # Initialising the number of epochs for the NN
        epoch = 100

        # Initialising the landmark points for the registration problem
        x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
        y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

        # Initialising the grid
        grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
        grid_points_y = grid_points_x.copy()

        plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
        plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

        grid_points = [[plotting_grid_points_x[i], plotting_grid_points_y[i]] for i in range(len(plotting_grid_points_x))]

        plotting_x_data = [[x_data[i][0] for i in range(len(x_data))], [x_data[i][1] for i in range(len(x_data))]]
        plotting_y_data = [[y_data[i][0] for i in range(len(y_data))], [y_data[i][1] for i in range(len(y_data))]]

        # Initialising the FeedFoward neural network from the class definition
        shape_model = Feedforward(2, hidden_size=300)
        criterion = torch.nn.MSELoss()

        # Setting the gradient descent method for each iteration.
        if grid_square == 0:
            optimizer = torch.optim.Rprop(shape_model.parameters(), lr=0.01)
        elif grid_square == 1:
            optimizer = torch.optim.RMSprop(shape_model.parameters(), lr=0.01)
        elif grid_square == 2:
            optimizer = torch.optim.SGD(shape_model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adam(shape_model.parameters(), lr=0.01)

        # Start of the training 
        x_train = torch.FloatTensor(x_data)
        y_train = torch.FloatTensor(y_data)
        grid_tensor = torch.FloatTensor(grid_points)

        num_of_train = len(x_train)
        num_of_grid = len(grid_tensor)

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
        
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()

        # Now plot the results
        for i in range(len(collect_train_pred)):
            data_epoch = collect_train_pred[i].tolist()
            grid_epoch = collect_warped_grid[i].tolist()
            collect_train_pred[i] = [[data_epoch[i][0] for i in range(num_of_train)], [data_epoch[i][1] for i in range(num_of_train)]]
            collect_warped_grid[i] = [[grid_epoch[i][0] for i in range(num_of_grid)], [grid_epoch[i][1] for i in range(num_of_grid)]]

        axs[plot_spaces[grid_square][0],plot_spaces[grid_square][1]].plot(plotting_y_data[0], plotting_y_data[1], 'o', color='b')
        axs[plot_spaces[grid_square][0],plot_spaces[grid_square][1]].plot(collect_train_pred[-1][0], collect_train_pred[-1][0], 'o', color='r')
        axs[plot_spaces[grid_square][0],plot_spaces[grid_square][1]].plot(collect_warped_grid[-1][0], collect_warped_grid[-1][1], '.', color='k', markersize=0.8)
        axs[plot_spaces[grid_square][0],plot_spaces[grid_square][1]].set_title(plot_titles[grid_square])

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('plots/compare_gradient_descent.png')
    # show the plot once all spaces have been plotted
    plt.show()

if __name__ == '__main__':
    main()
