import torch

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class Invertible(torch.nn.Module):
    '''
    This class defines the basic structure of an invertible neural network using FrEIA.
    '''

    def __init__(self, input_size: int, hidden_size: int):
        '''
        :param input_size: number of neurons of the input layer.
        :param hidden_size: number of neurons in the hidden layer.
        '''
        super(Invertible, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.relu = torch.nn.ReLU()
             
    def initialise_inn(self):
        self.inn = Ff.SequenceINN(self.input_size)
        for k in range(8):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc, permute_soft=True)
        
        return

    # As far as I understand, this is the fundamental building block of the inn
    def subnet_fc(self,dims_in, dims_out):
        return torch.nn.Sequential(torch.nn.Linear(dims_in, self.hidden_size), self.relu,
                         torch.nn.Linear(self.hidden_size,  dims_out))  