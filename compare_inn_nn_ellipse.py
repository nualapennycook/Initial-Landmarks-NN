from shape_nn.register_shape_data import RegisterShapeData
from shape_nn.rotate_shape import rotate_shape
from shape_nn.training_feed_forward_nn import train_network
from shape_nn.training_invertible_nn import train_network as train_inn_network
import matplotlib.pyplot as plt

def compare_inn_nn(epoch, hidden_size, learning_rate):
    # Firstly need to extract the training data from the text files
    # We train the network for a set shape to another set shape
    input_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/train_data/ellipse2.txt')
    input_data.extract_shape_data()
    input_data.centre_shape()
    input_data.scale_shape()

    set_data = RegisterShapeData(path_to_shape_data='shape_landmark_data/train_data/ellipse2.txt')
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
    y_landmarks, warped_x_data, warped_test_data, [train_loss, test_loss] = train_network(x_data=x_data, y_data=y_data, epoch=epoch, hidden_size = hidden_size, learning_rate=learning_rate)
    y_landmarks, warped_inn_x_data, warped_inn_test_data, [train_inn_loss, test_inn_loss] = train_inn_network(x_data=x_data, y_data=y_data, epoch=epoch, hidden_size=hidden_size, number_of_blocks=1, learning_rate=learning_rate)

    return train_loss, train_inn_loss, test_loss, test_inn_loss

    # Comparing effect of number of epochs
number_of_epochs = [100, 500, 1000, 1500, 2000]
collect_train_loss = [0 for epoch in number_of_epochs]
collect_train_inn_loss = [0 for epoch in number_of_epochs]
collect_test_loss = [0 for epoch in number_of_epochs]
collect_test_inn_loss = [0 for epoch in number_of_epochs]

for i in range(len(number_of_epochs)):

    collect_train_loss[i], collect_train_inn_loss[i], collect_test_loss[i], collect_test_inn_loss[i] = compare_inn_nn(epoch = number_of_epochs[i], hidden_size=200, learning_rate=0.01)

fig, ax = plt.subplots()
plt.plot(number_of_epochs, collect_train_loss)
plt.plot(number_of_epochs, collect_train_inn_loss)
plt.plot(number_of_epochs, collect_test_loss)
plt.plot(number_of_epochs, collect_test_inn_loss)
plt.legend(['Train loss:ffnn', 'Train loss: inn', 'Test loss: ffnn', 'Test loss: inn'])
plt.show()

    # # Plotting the data
    # fig, ax = plt.subplots()
    # plt.plot(plotting_y_data[0], plotting_y_data[1], marker='x')
    # plt.plot(plotting_x_data[0], plotting_x_data[1], marker='x')
    # plt.plot(reshaped_y_pred[0], reshaped_y_pred[1], marker='x')
    # for i in range(len(reshaped_y_pred[0])):
    #     plt.plot([plotting_y_data[0][i], reshaped_y_pred[0][i]], [plotting_y_data[1][i], reshaped_y_pred[1][i]], color='y')
    # plt.legend(['Target Shape', 'Template Shape', 'Mapped Template Shape'])
    # plt.show()

