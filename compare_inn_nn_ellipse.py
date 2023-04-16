from shape_nn.register_shape_data import RegisterShapeData
from shape_nn.rotate_shape import rotate_shape
from shape_nn.training_feed_forward_nn import train_network
from shape_nn.training_invertible_nn import train_network as train_inn_network
import matplotlib.pyplot as plt

'''
Script to compare the result of image registration using a feedforward and an invertible neural network.
Uses the ellipse to ellipse example.
Produces a plot comparing the test and training losses for feedforward and invertible neural networks.
'''

def compare_inn_nn(epoch, hidden_size, learning_rate):
    # Firstly need to extract the training data from the text files
    input_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/ellipse2.txt')
    input_data.extract_shape_data()
    input_data.centre_shape()
    input_data.scale_shape()

    set_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/ellipse2.txt')
    set_data.extract_shape_data()
    set_data.centre_shape()
    set_data.scale_shape()

    # Rotating the input data to the orientation of the set data
    x_data = rotate_shape(rotate_data=input_data.shape_data, fixed_data=set_data.shape_data)

    # Reshaping the data into coordinate pairs for input to the neural network
    x_data = [[x_data[0][i], x_data[1][i]] for i in range(len(x_data[0]))]
    y_data = [[1.1*set_data.shape_data[0][i], 1.1*set_data.shape_data[1][i]] for i in range(len(set_data.shape_data[0]))]

    # Reducing the number of landmarks to reduce the problem size
    skip_step = 10
    reduction = int(len(x_data)/skip_step)
    x_data = [x_data[i*skip_step] for i in range(reduction)]
    y_data = [y_data[i*skip_step] for i in range(reduction)]

    # Training the feedforward neural network
    y_landmarks, warped_x_data, warped_test_data, [train_loss, test_loss] = train_network(x_data=x_data, y_data=y_data, epoch=epoch, hidden_size = hidden_size, learning_rate=learning_rate)
    
    # Training the invertible neural network
    y_landmarks, warped_inn_x_data, warped_inn_test_data, [train_inn_loss, test_inn_loss] = train_inn_network(x_data=x_data, y_data=y_data, epoch=epoch, hidden_size=hidden_size, number_of_blocks=1, learning_rate=learning_rate)

    return train_loss, train_inn_loss, test_loss, test_inn_loss

def main():

    # Comparing effect of number of epochs
    number_of_epochs = [100, 500, 1000, 1500, 2000]
    collect_train_loss = [0 for epoch in number_of_epochs]
    collect_train_inn_loss = [0 for epoch in number_of_epochs]
    collect_test_loss = [0 for epoch in number_of_epochs]
    collect_test_inn_loss = [0 for epoch in number_of_epochs]

    # Evaulating the neural network for different numbers of epochs
    for i in range(len(number_of_epochs)):

        collect_train_loss[i], collect_train_inn_loss[i], collect_test_loss[i], collect_test_inn_loss[i] = compare_inn_nn(epoch = number_of_epochs[i], hidden_size=200, learning_rate=0.01)

    # Plotting a comparison of the test and training error for both neural networks
    fig, ax = plt.subplots()
    plt.plot(number_of_epochs, collect_train_loss)
    plt.plot(number_of_epochs, collect_train_inn_loss)
    plt.plot(number_of_epochs, collect_test_loss)
    plt.plot(number_of_epochs, collect_test_inn_loss)
    plt.legend(['Train loss:ffnn', 'Train loss: inn', 'Test loss: ffnn', 'Test loss: inn'])
    plt.title('Ellipse Feedforward and Invertible NN Registration Error')
    plt.savefig('animations/compare_ellipse_loss.png')
    plt.show()

if __name__ == '__main__':
    main()

