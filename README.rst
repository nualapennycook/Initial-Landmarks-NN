**Shape Modelling with Deep Neural Networks**

This repository contains implementations simple feedforward and invertible neural networks. The feedforward neural network is implemented in Pytorch and the invertible neural network is implemented using FrEIA (Framework for Easily Invertible Architectures) which can be found at https://github.com/vislearn/FrEIA. This repository supports my project report on Shape Modelling with Deep Neural Networks.

The dependencies for this project are:

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | >= 3.7                        |
+---------------------------+-------------------------------+
| Pytorch                   | >= 1.0.0                      |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.15.0                     |
+---------------------------+-------------------------------+
| Scipy                     | >= 1.5                        |
+---------------------------+-------------------------------+
| Matplotlib                | = 3.6.2                       |
+---------------------------+-------------------------------+
| FFmpeg                    | = 1.4                         |
+---------------------------+-------------------------------+

**Project Structure**

* The structure for the implementation and training of neural networks can be found in the folder 'shape_nn'.
* All demos can be found in the main project folder and generate plots or animations. 
* All the data required for these demos can be found in the file named 'shape_landmark_data'. 
* Pre-generated plots and animations from the demos can be found in the files names 'plots' and 'animations' respectively.
* The file named 'FrEIA' contains a slightly modified version of the FrEIA source code. This is to improve convergence for the specific application of invertible neural networks to the ellipse registration problem, and the registration more generally.

@software{freia,
  author = {Ardizzone, Lynton and Bungert, Till and Draxler, Felix and Köthe, Ullrich and Kruse, Jakob and Schmier, Robert and Sorrenson, Peter},
  title = {{Framework for Easily Invertible Architectures (FrEIA)}},
  year = {2018-2022},
  url = {https://github.com/vislearn/FrEIA}
}