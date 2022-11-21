import torch
from shape_nn.register_shape_data import RegisterShapeData
from shape_nn.rotate_shape import rotate_shape
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from shape_nn.feed_forward_nn import Feedforward

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

# Plotting the input data before it is reshaped
fig, ax = plt.subplots()
# ax.plot(x_data[0], x_data[1], marker = 'x')
# line, = ax.plot(x_data[0], x_data[1])
# ax.plot(set_data.shape_data[0], set_data.shape_data[1], marker='x')

# Reorgainising/reshaping the data into pairs
x_data = [[x_data[0][i], x_data[1][i]] for i in range(len(x_data[0]))]
y_data = [[1.1*set_data.shape_data[0][i], 1.1*set_data.shape_data[1][i]] for i in range(len(set_data.shape_data[0]))]

# Reducing the number of landmarks
skip_step = 10
reduction = int(len(x_data)/skip_step)
x_data = [x_data[i*skip_step] for i in range(reduction)]
y_data = [y_data[i*skip_step] for i in range(reduction)]

plotting_x_data = [[x_data[i][0] for i in range(reduction)], [x_data[i][1] for i in range(reduction)]]
plotting_y_data = [[y_data[i][0] for i in range(reduction)], [y_data[i][1] for i in range(reduction)]]

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
shape_model = Feedforward(2, 300)
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
epoch = 2000
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


## Animation of the training
# Reshaped y_pred for plotting
y_train_pred_list = y_train_pred.tolist()
reshaped_y_pred = [[y_train_pred_list[i][0] for i in range(num_of_train)], [y_train_pred_list[i][1] for i in range(num_of_train)]]

for i in range(len(collect_train_pred)):
    data_epoch = collect_train_pred[i].tolist()
    collect_train_pred[i] = [[data_epoch[i][0] for i in range(num_of_train)], [data_epoch[i][1] for i in range(num_of_train)]]

def animate_shape(i):
    line.set_data(collect_train_pred[i][0], collect_train_pred[i][1])
    return line,

# anim = FuncAnimation(fig, animate_shape,
#                      frames = epoch, interval = 20, blit = True)

# writervideo = PillowWriter(fps = 10)
    
# anim.save('cat_to_plane.gif', 
#           writer = writervideo)


plt.plot(reshaped_y_pred[0], reshaped_y_pred[1], marker='x')
plt.plot(plotting_x_data[0], plotting_x_data[1], marker='x')
plt.plot(plotting_y_data[0], plotting_y_data[1], marker='x')
for i in range(len(reshaped_y_pred[0])):
    plt.plot([plotting_y_data[0][i], reshaped_y_pred[0][i]], [plotting_y_data[1][i], reshaped_y_pred[1][i]], color='y')
# plt.xlim([-60, 70])
# plt.ylim([-60, 70])
plt.show()