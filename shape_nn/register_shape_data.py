'''
Class to read in shape data from text files.
'''
class RegisterShapeData:

    def __init__(self, path_to_shape_data) -> None:
        self.path_to_shape_data = path_to_shape_data
        self.shape_data = []

        self.centred_shape_data = []
        self.x_mean = None
        self.y_mean = None

        self.centred_scaled_data = []

        self.num_of_points = None

    def extract_shape_data(self) -> None:
        '''
        Method to extract shape data from text files as a list of x and y coordinates
        Sets the self.shape_data attribute as a list of x coords and a list of y coords
        '''
        with open(self.path_to_shape_data) as file:
            for line in file:
                self.shape_data.append(line)

            # Formatting the x coordinate data
            # Reads as single character strings, so concatenate the whole line
            self.shape_data[0] = ''.join(self.shape_data[0])
            # Then split according to spacing 
            self.shape_data[0] = self.shape_data[0].split(' ')
            # Convert to strings to floats
            self.shape_data[0] = [float(value.replace('-', '-')) for value in self.shape_data[0]]

            # Do the same for y
            self.shape_data[1] = ''.join(self.shape_data[1])
            # Then split according to spacing 
            self.shape_data[1] = self.shape_data[1].split(' ')
            # Convert to strings to floats
            self.shape_data[1] = [float(value.replace('-', '-')) for value in self.shape_data[1]]

        return 

    def centre_shape(self) -> None:
        '''
        Method to centre shape data about the origin using the translation method for 
        Procrustles alignment.
        Sets the self.centred_shape_data param with the centred x and y coordinates.
        '''
        self.num_of_points = len(self.shape_data[1])

        # Compute mean of x coords and y coords
        self.x_mean = sum(self.shape_data[0])/self.num_of_points
        self.y_mean = sum(self.shape_data[1])/self.num_of_points

        self.centred_shape_data = self.shape_data

        # Centre x and y coords in relation to the mean
        for i in range(self.num_of_points):
            self.centred_shape_data[0][i] = self.centred_shape_data[0][i] - self.x_mean
            self.centred_shape_data[1][i] = self.centred_shape_data[1][i] - self.y_mean
        
        return

    def scale_shape(self) -> None:
        if self.centred_shape_data == []:
            raise ValueError("Data not yet centred, please call self.centre_shape method first.")
        else:
            squared_coords = [coord**2 for coord in self.centred_shape_data[0] and self.centred_shape_data[1]]

            scale_param = (sum(squared_coords))**(1/2)/self.num_of_points

            self.centred_scaled_data = [[coord/scale_param for coord in self.centred_shape_data[0]], [coord/scale_param for coord in self.centred_shape_data[1]]]

        return