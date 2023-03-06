from shape_nn.invertible_nn import Invertible
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

grid_resolution = 20


def main():
    epoch = 2000
    # x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75], [0.5, 0.5]]
    # y_data = [[0.2, 0.2], [0.2, 0.7], [0.7, 0.2], [0.7, 0.7], [0.1, 0.1]]
    x_data = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    y_data = [[0.25, 0.3], [0.2, 0.6], [0.3, 0.25], [0.5, 0.5]]

    grid_points_x = [1/(grid_resolution-1)*i for i in range(grid_resolution)]
    grid_points_y = grid_points_x.copy()

    plotting_grid_points_y = [coord for coord in grid_points_y for i in range(grid_resolution)]
    plotting_grid_points_x = [coord for i in range(grid_resolution) for coord in grid_points_x]

    grid_points = [[plotting_grid_points_x[i], plotting_grid_points_y[i]] for i in range(len(plotting_grid_points_x))]

    plotting_x_data = [[x_data[i][0] for i in range(len(x_data))], [x_data[i][1] for i in range(len(x_data))]]
    plotting_y_data = [[y_data[i][0] for i in range(len(y_data))], [y_data[i][1] for i in range(len(y_data))]]
    
    plt.plot(plotting_y_data[0], plotting_y_data[1], 'o', color='b')
    plt.plot(plotting_x_data[0], plotting_x_data[1], 'o', color='r')
    plt.plot(plotting_grid_points_x, plotting_grid_points_y, '.', color='k', markersize=0.8)
    plt.legend(['target points', 'landmarks', '$\phi$ applied to evenly spaced mesh'], loc='upper right')
    plt.show()

    # Initialising the FeedFoward neural network from the class definition
    shape_model = Invertible(2, hidden_size=300, number_of_blocks=2)
    shape_model.initialise_inn()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shape_model.parameters(), lr=0.01)

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
        y_train_pred, dump = shape_model.inn(x_train)
        warped_grid_step, dump = shape_model.inn(grid_tensor)
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

    fig, ax = plt.subplots()
    ax.plot(plotting_y_data[0], plotting_y_data[1], 'o', color='b')
    # ax.set_xlim([0, 0.8])
    # ax.set_ylim([0, 0.8]) 
    line1, = ax.plot(plotting_x_data[0], plotting_x_data[1], 'o', color='r')
    line2, = ax.plot(plotting_grid_points_x, plotting_grid_points_y, '.', color='k', markersize=0.8)
    ax.legend(['target points', '$\phi$ applied to landmarks', '$\phi$ applied to evenly spaced mesh'], loc='upper right')

    def animate_shape(i):
        line1.set_data(collect_train_pred[i][0], collect_train_pred[i][1])
        line2.set_data(collect_warped_grid[i][0], collect_warped_grid[i][1])
        return (line1, line2)

    anim = FuncAnimation(fig, animate_shape,
                        frames = epoch, interval = 20, blit = True)

    writervideo = PillowWriter(fps = 10)
        
    anim.save('animations/warped_grid_inn.gif', 
            writer = writervideo)

if __name__ == '__main__':
    main()
