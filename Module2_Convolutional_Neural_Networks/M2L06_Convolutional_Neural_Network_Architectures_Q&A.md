# Q&A

## During back propagation, for what functions would gradient multiplication be element-wise?

During backpropagation, gradient multiplication will be element-wise for activation functions applied element-wise to the input, i.e., they are applied independently to each element of the input tensor. Here are some common activation functions where gradient multiplication is element-wise:  
   
1. **ReLU (Rectified Linear Unit)**: As previously mentioned, the ReLU function is defined as `f(x) = max(0, x)`, and its derivative is 1 for positive input and 0 otherwise. During backpropagation, gradient multiplication is element-wise.  
   
2. **Sigmoid**: The sigmoid function is defined as `f(x) = 1 / (1 + exp(-x))`. Its derivative is `f'(x) = f(x) * (1 - f(x))`. When backpropagating through the sigmoid activation function, the gradient multiplication is element-wise.  
   
3. **Tanh (Hyperbolic Tangent)**: The tanh function is defined as `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Its derivative is `f'(x) = 1 - f(x)^2`. Similar to ReLU and Sigmoid, gradient multiplication for tanh is also element-wise during backpropagation.  
   
4. **Softplus**: The softplus function is defined as `f(x) = log(1 + exp(x))`. Its derivative is `f'(x) = 1 / (1 + exp(-x))`, which is the sigmoid function. When backpropagating through the softplus activation function, gradient multiplication is element-wise.  
   
In general, for activation functions that are applied to each element of the input independently, the gradient multiplication during backpropagation will be element-wise. This is because the local gradient of the activation function depends only on the value of the input at the corresponding element, and the chain rule is applied element-wise.


## Does multiplication associative law apply to matrix multiplication?
Yes, the associative law does apply to matrix multiplication. The associative law states that for any three matrices A, B, and C, the following relationship holds:  
   
```  
(A * B) * C = A * (B * C)  
```  
   
Matrix multiplication is associative, meaning that you can group the matrices in any order when performing multiplication without changing the result, as long as the dimensions of the matrices are compatible for multiplication. Note that this does not imply commutativity, as the order of matrices still matters:  
   
```  
A * B ≠ B * A  (in general)  
```  
   
Matrix multiplication is generally *not commutative*, meaning that changing the order of the matrices can result in different products. However, the associative property allows you to regroup matrices during multiplication without affecting the result.






