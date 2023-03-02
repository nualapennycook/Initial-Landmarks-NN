from shape_nn.register_shape_data import RegisterShapeData
from shape_nn.rotate_shape import rotate_shape
from shape_nn.training_invertible_nn import train_network
import matplotlib.pyplot as plt

def main():
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
    y_landmarks, warped_x_data, warped_test_data = train_network(x_data=x_data, y_data=y_data, epoch=1000, hidden_size=200, number_of_blocks=2, learning_rate=0.01)

    plotting_x_data = [[x_data[i][0] for i in range(reduction)], [x_data[i][1] for i in range(reduction)]]
    plotting_y_data = [[y_data[i][0] for i in range(reduction)], [y_data[i][1] for i in range(reduction)]]
    plotting_y_landmarks = [[y_landmarks[i][0] for i in range(len(y_landmarks))], [y_landmarks[i][1] for i in range(len(y_landmarks))]]

    reshaped_y_pred = warped_x_data[0]

    # Plotting the data
    fig, ax = plt.subplots()
    plt.plot(plotting_y_data[0], plotting_y_data[1], marker='x')
    plt.plot(plotting_x_data[0], plotting_x_data[1], marker='x')
    plt.plot(reshaped_y_pred[0], reshaped_y_pred[1], marker='x')
    plt.plot(warped_test_data[0], warped_test_data[1], marker='x', linestyle='None')
    # Plotting lines between the target points and the corresponding mapped x points
    for i in range(len(reshaped_y_pred[0])):
        plt.plot([plotting_y_landmarks[0][i], reshaped_y_pred[0][i]], [plotting_y_landmarks[1][i], reshaped_y_pred[1][i]], color='y')
    plt.legend(['Target Shape', 'Template Shape', 'Mapped Template Shape: Train', 'Mapped Template Shape: Test'])
    plt.show()

if __name__ == '__main__':
    main()
