import torch

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class Invertible(torch.nn.Module):
    '''
    This class defines the basic structure of an invertible neural network using FrEIA.
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
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
             
    def initialise_inn(self):
        self.inn = Ff.SequenceINN(self.input_size)
        for k in range(self.number_of_blocks):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc, permute_soft=True)
        
        return

    # As far as I understand, this is the fundamental building block of the inn
    def subnet_fc(self,dims_in, dims_out):
        return torch.nn.Sequential(torch.nn.Linear(dims_in, self.hidden_size), self.tanh,
                         torch.nn.Linear(self.hidden_size,  dims_out))  