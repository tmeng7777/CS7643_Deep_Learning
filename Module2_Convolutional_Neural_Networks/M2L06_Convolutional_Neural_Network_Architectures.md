# Lesson 6: Convolutional Neural Network Architectures

## Backwards Pass for Convolution Layer

### Backwards Pass for Conv Layers
It is intructive to calculate __the backwards pass__ of a convolution layer
- Similar to fuly connected layer, will be __simple vecgtorized linear algebra operation!__
- We will see a __duality__ between cross-correlation and convolution

![img](imgs/M2L06_01.png)

### Recap: Cross-Correlation

![img](imgs/M2L06_02.png)

### Iterators
![img](imgs/M2L06_03.png)

__Some simplification:__ 1 channel input, 1 kernel (channel output), padding (here 2 pixels on right/bottom) to make output the same size

### Gradient Terms and Notation

![img](imgs/M2L06_04.png)

### Backpropagation Chain Rule

![img](imgs/M2L06_05.png)