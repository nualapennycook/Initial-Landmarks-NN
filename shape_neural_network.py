import torch
from register_shape_data import RegisterShapeData
from rotate_shape import rotate_shape
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class Feedforward(torch.nn.Module):
    '''
    This class defines the basic structure of the pytorch neural network.
    This feed forward neural network has one hidden layer.
    This class is taken and modified from: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb.
    '''

    def __init__(self, input_size: int, hidden_size: int):
        '''
        :param input_size: number of neurons of the input layer.
        :param hidden_size: number of neurons in the hidden layer.
        '''
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        # Defining the output size of the output 
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()
             
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.tanh(hidden)
        output = self.fc2(relu)
        # output = self.sigmoid(output)
        return output

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

# Plotting the input data before it is reshaped
fig, ax = plt.subplots()
ax.plot(x_data[0], x_data[1])
line, = ax.plot(x_data[0], x_data[1])
ax.plot(set_data.shape_data[0], set_data.shape_data[1])

# Reorgainising/reshaping the data into pairs
x_data = [[x_data[0][i], x_data[1][i]] for i in range(len(x_data[0]))]
y_data = [[set_data.shape_data[0][i], set_data.shape_data[1][i]] for i in range(len(set_data.shape_data[0]))]

# Splitting the data into test and train sets
# 250 coords for the training and 50 for the test
x_train = x_data[:250]
x_test = x_data[250:]
y_train = y_data[:250]
y_test = y_data[250:]

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
epoch = 100
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
reshaped_y_pred = [[y_train_pred_list[i][0] for i in range(250)], [y_train_pred_list[i][1] for i in range(250)]]

for i in range(len(collect_train_pred)):
    data_epoch = collect_train_pred[i].tolist()
    collect_train_pred[i] = [[data_epoch[i][0] for i in range(250)], [data_epoch[i][1] for i in range(250)]]

def animate_shape(i):
    line.set_data(collect_train_pred[i][0], collect_train_pred[i][1])
    return line,

anim = FuncAnimation(fig, animate_shape,
                     frames = epoch, interval = 20, blit = True)

writervideo = PillowWriter(fps = 10)
    
anim.save('cat_to_plane.gif', 
          writer = writervideo)


# plt.plot(reshaped_y_pred[0], reshaped_y_pred[1])
# plt.xlim([-60, 70])
# plt.ylim([-60, 70])
# plt.show()