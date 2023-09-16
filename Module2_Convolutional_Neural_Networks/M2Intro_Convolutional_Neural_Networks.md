# Module 2: Convolutional Neural Networks

## Neural Network Overview

We have seen how to build and optimize deep feed forward architectures consisting of linear & non-linear (e.g. ReLU) layers
- This can be generalized to arbitrary **computation graphs**
- **Backpropagation and automatic differentiation** can be used to optimize all parameters vis **gradient descent**

## Nodes with Local Receptive Fields
- But layers don't have to be fully connected - **other connectivity structures are possible**
- For images, it makes sense for output nodes to **consider only small patches of inputs**

## Convolution Operations
This operation, which can be another layer in the neural network, can be formalized as **convolution operations**.

This new convolution layer can take **anhy input 3D tensor (say, RGB image)** and **output another similarly-shaped output**
- Where **high values represent strong responses** of the learned features

![convoluation neural network](imgs/M2_Intro_01.png)

## References 
- Yann LeCun, Kevin Murphy
- Szegedy et al. Going deeper with convolutions
- Zeiler & Fergus, Visualizing and Understanding Convolutional Networks 
- Ren et al., Faster R-CNN