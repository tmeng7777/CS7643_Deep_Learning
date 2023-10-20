# Module 3: Structured Neural Representations

## The Space of Architectures

![img](imgs/M3Intro_01.png)

## Recurrent Neural Networks

![img](imgs/M3Intro_02.png)

## Enter the Transformer

Transformer [Vaswani et. al. 2017] is a multi-layer attention model that is currently state of the art in most language tasks (and in many other things!)

Has superior performance compared to previous attention based architecture via
- Multi-query hidden-state propagation ("self-attention")
- __Multi-head attention__
- Residual connections, LayerNorm

*More info in Q&A: __Summarize Vaswani et. al. 2017 paper about Transformer...__*

![img](imgs/M3Intro_03.png)

## Sequence Modeling

![img](imgs/M3Intro_04.png)

## Example Application: NLP

![img](imgs/M3Intro_05.png)

## Modeling Language as a Sequence

![img](imgs/M3Intro_06.png)

## Word Embeddings

__Word2vec: the Skip-gram model__
- The idea: use words to __predict__ their context words
- Context: a fixed window of size __*2m*__

![img](imgs/M3Intro_07.png)

## Graph Embeddings

__Embedding:__ A learned map from entities to vectors of numbers that encodes similarity
- Word embeddings: word -> vector
- Graph embeddings: node -> vector

__Graph Embedding:__ Optimize the objective that __connected nodes have more similar embeddings__ than unconnected nodes via gradient descent.

![img](imgs/M3Intro_08.png)

## Application: VideoSpace

![img](imgs/M3Intro_09.png)

## Machine Translation

![img](imgs/M3Intro_10.png)

## Attention in Natural Language PRocessing

__Alignment in machine translation:__ for each word in the target, get a distribution over words in the source [Brown et. al. 1993], (lots more)

![img](imgs/M3Intro_11.png)

## Sequence-to-Sequence Model

![img](imgs/M3Intro_12.png)

![img](imgs/M3Intro_13.png)

## Cross-Lingual Masked Language Modeling

![img](imgs/M3Intro_14.png)

*More info in Q&A: __What is Cross-Lingual Masked Language Modeling? ...__*

## Beam Search
- Search exponential space in linear time
- Beam size __*k*__ determines "width" of search
- At each step, extend each of __*k*__ elements by one token
- Top __*k*__ overall then become the hypotheses for next step

![img](imgs/M3Intro_15.png)

*More info in Q&A: __What is Cross-Lingual Masked Language Modeling? ...__*

## Differentiably Selecting a Vector from a Set
- Given a set of vectors {__*u_1, ..., u_N*__} and a "query" vector __*q*__
- We can select the most similar vector to __*q*__ via __*p = Softmax(Uq)*__

![img](imgs/M3Intro_16.png)
