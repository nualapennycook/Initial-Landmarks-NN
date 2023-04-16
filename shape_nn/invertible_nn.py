import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class Invertible(torch.nn.Module):
    '''
    This class defines the basic structure of an invertible neural network using FrEIA.
    FrEIA can be found at: https://github.com/vislearn/FrEIA
    The neural network is structured in blocks of simple subnetworks.
    The network can also be traversed backwards using an additional argument when the network is called.
    '''

    def __init__(self, input_size: int, hidden_size: int, number_of_blocks: int):
        '''
        :param input_size: number of neurons of the input layer.
        :param hidden_size: number of neurons in the hidden layer.
        :param number_of_blocks: number of coupling blocks, which are smaller feed-forward subnetworks
        '''
        super(Invertible, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.number_of_blocks = number_of_blocks

        # Selecting an activation function
        self.tanh = torch.nn.Tanh()
             
    def initialise_inn(self):
        '''
        Initialises the neural network, constructing a network with a specified number of subnet blocks.
        '''
        # Initialising invertible network
        self.inn = Ff.SequenceINN(self.input_size)
        # Adding layers
        for k in range(self.number_of_blocks):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc, permute_soft=False)
        
        return

    def subnet_fc(self,dims_in, dims_out):
        '''
        Constructs a subnet. This gives the basic structure of a layer of an invertible network.
        This is the fundamental building block of an INN.
        '''
        return torch.nn.Sequential(torch.nn.Linear(dims_in, self.hidden_size), self.tanh,
                         torch.nn.Linear(self.hidden_size,  dims_out))  