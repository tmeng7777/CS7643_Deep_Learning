# Lesson 11: Introduction to Structured Representations

## Neural Network Overview
We have seen how to __build and optimize deep feedforward architectures__ consisting of linear & non-linear (e.g. ReLU) layers
- This can be generalized to __arbitrary computation graphs__
- __Back propagation and automatic differentiation__ can be used to optimize all parameters via __gradient descent__

![img](imgs/M3L11_01.png)

![img](imgs/M3L11_02.png)

## Convolutional Neural Networks

![img](imgs/M3L11_03.png)

## Deep Learning = Hierarchical Compositionality

![img](imgs/M3L11_04.png)

## Hierarchical Compositionality

![img](imgs/M3L11_05.png)

## Relationships are Important

![img](imgs/M3L11_06.png)

## Relationships are Everywhere

![img](imgs/M3L11_07.png)

## The Space of Architectures

![img](imgs/M3L11_08.png)

## Graph Embeddings

__Embedding:__ A learned map from entities to vectors of numbers that encodes similarity
- Word embeddings: word -> vector
- Graph embeddings: node -> vector

__Graph Embedding:__ Optimize the objective that __connected nodes have more similar embeddings__ than unconnected nodes via gradient descent.

![img](imgs/M3L11_09.png)

*More info in Q&A: __In the domain of deep learning, summarize key advances of graph embedding in the recent years. Provide references.__*

## Propagating Information

When representing structured information, several things are important:
- __State:__ Compactly representing all the data we have processed thus far
- __"Neighborhoods":__ What other elements to incorporate?
	- Can be seen as selecting from as set of elements
	- Typically done with some similarity measure or attention
- __Propagation of information:__ How to update information give nselected elemtns

![img](imgs/M3L11_10.png)

## Differentiably Selecting a Vector
- Given a set of vectors {__*u_1, ..., u_N*__} and a "query"
 vector __*q*__
 - We can select the most similar vector to __*q*__ via __*p = Softmax(Uq)*__

![img](imgs/M3L11_11.png)

## Example: Non-Local Neural Networks

![img](imgs/M3L11_12.png)

![img](imgs/M3L11_13.png)


## References and Links:

- Szegedy et al. Going deeper with convolutions
- Zeiler & Fergus 2013
- Marc'Aurelio Ranzato, Yann LeCun
- Xu et al., “Scene Graph Generation by Iterative Message Passing”, 2017
- https://github.com/facebookresearch/PyTorch-BigGraphLinks
- Lerer et al. 19
- Adam Lerer
- Wang et al., “Non-local Neural Networks”, 2017