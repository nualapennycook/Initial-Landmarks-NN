import torch

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
        self.w = torch.empty(self.input_size, self.hidden_size)
        torch.nn.init.eye_(self.w)
        self.fc15 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc15.weight.data.copy_(torch.eye(self.hidden_size))
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        # Defining the output size of the output 
        self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)
        self.sigmoid = torch.nn.Sigmoid()
             
    def forward(self, x):
        # hidden1 = x@self.w
        hidden1 = self.fc1(x)
        hidden2 = self.fc15(hidden1)
        # hidden3 = self.fc15(hidden2)
        tahn = self.tanh(hidden2)
        output = self.fc2(tahn)
        # output = self.sigmoid(output)
        return output