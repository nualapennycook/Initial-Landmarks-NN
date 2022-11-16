from register_shape_data import RegisterShapeData
from rotate_shape import rotate_shape
import matplotlib.pyplot as plt

shark_data = RegisterShapeData(path_to_shape_data='shape_landmark_data\data\shark.txt')

dog_data = RegisterShapeData(path_to_shape_data='shape_landmark_data\data\cat.txt')

shark_data.extract_shape_data()
dog_data.extract_shape_data()

shark_data.centre_shape()
dog_data.centre_shape()

shark_data.scale_shape()
dog_data.scale_shape()

# Plotting the data
plt.plot(shark_data.shape_data[0], shark_data.shape_data[1])
plt.plot(dog_data.shape_data[0], dog_data.shape_data[1])
plt.show()

# Plotting the rotated data

rotated_dog = rotate_shape(rotate_data=dog_data.shape_data, fixed_data=shark_data.shape_data)

plt.plot(shark_data.shape_data[0], shark_data.shape_data[1])
plt.plot(rotated_dog[0], rotated_dog[1])
plt.show()
