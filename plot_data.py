import matplotlib.pyplot as plt

'''
Script to read in shape data from text files and plot it.
'''

shape_data = []

# Extracting the data from the text file
with open('shape_landmark_data\data\shark.txt') as file:
    for line in file:
        shape_data.append(line)

    # Splitting into two lists for x and y coordinates
    [shape_data_x, shape_data_y] = shape_data

    # Reads as single character strings, so concatenate the whole line
    shape_data_x = ''.join(shape_data_x)
    # Then split according to spacing 
    shape_data_x = shape_data_x.split(' ')
    # Convert to strings to floats
    shape_data_x = [float(value.replace('-', '-')) for value in shape_data_x]

    # Do the same for y
    shape_data_y = ''.join(shape_data_y)
    shape_data_y = shape_data_y.split(' ')
    shape_data_y = [float(value.replace('-', '-')) for value in shape_data_y]

    print(len(shape_data_x))

    # Plotting the data
    plt.plot(shape_data_x, shape_data_y)
    plt.show()
