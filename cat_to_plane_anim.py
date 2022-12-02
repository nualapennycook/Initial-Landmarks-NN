from shape_nn.register_shape_data import RegisterShapeData
from shape_nn.rotate_shape import rotate_shape
from shape_nn.training_feed_forward_nn import train_network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

'''
Demo to animate the cat to plane registration.
'''

epoch = 1000

def main():
    # Firstly need to extract the training data from the text files
    # We train the network for a set shape to another set shape
    input_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/train_data/cat.txt')
    input_data.extract_shape_data()
    input_data.centre_shape()
    input_data.scale_shape()

    set_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/train_data/plane.txt')
    set_data.extract_shape_data()
    set_data.centre_shape()
    set_data.scale_shape()

    # Rotating the input data to the orientation of the set data
    x_data = rotate_shape(rotate_data=input_data.shape_data, fixed_data=set_data.shape_data)

    # Reorgainising/reshaping the data into pairs
    x_data = [[x_data[0][i], x_data[1][i]] for i in range(len(x_data[0]))]
    y_data = [[1.1*set_data.shape_data[0][i], 1.1*set_data.shape_data[1][i]] for i in range(len(set_data.shape_data[0]))]

    # Reducing the number of landmarks
    skip_step = 10
    reduction = int(len(x_data)/skip_step)
    x_data = [x_data[i*skip_step] for i in range(reduction)]
    y_data = [y_data[i*skip_step] for i in range(reduction)]

    # Training the neural network
    warped_x_data = train_network(x_data=x_data, y_data=y_data, epoch=epoch)

    plotting_x_data = [[x_data[i][0] for i in range(reduction)], [x_data[i][1] for i in range(reduction)]]
    plotting_y_data = [[y_data[i][0] for i in range(reduction)], [y_data[i][1] for i in range(reduction)]]

    fig, ax = plt.subplots()
    ax.plot(plotting_x_data[0], plotting_x_data[1], marker = 'x')
    ax.plot(plotting_y_data[0], plotting_y_data[1], marker = 'x')
    line, = ax.plot(plotting_x_data[0], plotting_x_data[1])
    ax.legend(['initial shape', 'target shape', 'result of NN'])

    def animate_shape(i):
        line.set_data(warped_x_data[i][0], warped_x_data[i][1])
        return line,

    anim = FuncAnimation(fig, animate_shape,
                        frames = epoch, interval = 20, blit = True)

    writervideo = PillowWriter(fps = 10)
        
    anim.save('animations/cat_to_plane.gif', 
            writer = writervideo)

if __name__ == '__main__':
    main()