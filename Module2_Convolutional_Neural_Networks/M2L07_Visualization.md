# Lesson 7: Visualization

## Visualization of Neural Networks

### Visualization Neural Networks

Given a __trained__ model, we'd like to understand what it learned
- Weights
- Activations
- Gradients
- Robustness

![img](imgs/M2L07_01.png)

### Visualizing Weights

__FC Layer:__ (if connected to image itself) Reshape weights for a node back into size of image, scale 0-255
![img](imgs/M2L07_02.png)

__Conv layers:__ For each kernel, scale values from 0-255 and visualize
![img](imgs/M2L07_03.png)

*Problem: 3x3 filters difficult to interpret!*

### Visualizing Output Maps

We can also produce __visualization output (aka activation/filter) maps__

*There are __larger__ early in the network*

![img](imgs/M2L07_04.png)

![img](imgs/M2L07_05.png)

### Activations - Small Output Sizes

![img](imgs/M2L07_06.png)

*Problem: Small conv outputs also hard to interpret*

### CNN101 and CNN Explainer

![img](imgs/M2L07_07.png)

- [CNN-Explainer](https://poloclub.github.io/cnn-explainer/)
- [CNN 101: Interactive Visual Learning for Convolutional Neural Networks](https://fredhohman.com/papers/cnn101)

### Dimensionality Reduction: t-SNE

We can take the activations of any layer (FC, conv, etc.) and __perform dimensionality reduction__
- Often reduce to two dimensions for plotting
- E.g. using Principle Component Analysis (PCA)

__t-SNE is most common__
- Performs non-linear mapping to preserve pair-wise distances

![img](imgs/M2L07_08.png)

### Visualizing Neural Networks

![img](imgs/M2L07_09.png)

### Summary & Caveats
While these methods provide __some__ visually interpretable representations, they can be misleading or uninformative (Adebayo et al., 2018)

Assessing interpretability is difficult
- Requires __user studies__ to show __usefulness__
- E.g. they allow a user to predict mistakes beforehand

Neural networks learn __distributed representations__
- (no one node represents a particular feature)
- This makes interpretation difficult

*Adebayo et al., “Sanity Checks for Saliency Maps”, 2018.*


### References and Links:
- Fei-Fei Li, Justin Johnson, Serena Yeung, from CS 231n
- Zeiler & Fergus, 2014
- Simonyan et al, 2013
- Hendrycks & Dietterich, 2019
- Yosinski et al., “Understanding Neural Networks Through Deep Visualization”, 2015
- https://poloclub.github.io/cnn-explainer/Links to an external site.
- https://fredhohman.com/papers/cnn101 Links to an external site.
- Van der Maaten & Hinton, “Visualizing Data using t-SNE”, 2008.
- Adebayo et al., “Sanity Checks for Saliency Maps”, 2018.
