# Q&A

## Summarize Vaswani et. al. 2017 paper about Transformer. What is transformer based on? What are the key innovations?

The paper "Attention Is All You Need" by Vaswani et al. in 2017 introduced the Transformer model, which has since become foundational for many modern natural language processing tasks. Here's a summary of the key points:

### What is the Transformer based on?
The Transformer model is based on self-attention mechanisms that can weigh input elements differently regardless of their positions in the sequence. This allows the model to draw global dependencies between input and output, making it especially effective for sequence-to-sequence tasks. Unlike RNNs or LSTMs, which process sequences iteratively, the Transformer processes all tokens in the input sequence in parallel, leading to efficiency gains.

### Key Innovations:

1. **Self-Attention Mechanism**: The Transformer introduces a novel self-attention mechanism that can relate different positions of a single sequence, allowing it to capture long-range dependencies in the data. Each token in the input can focus on different parts of the sequence, which is achieved by computing a weighted sum of all tokens in the sequence.

2. **Scaled Dot-Product Attention**: The Transformer uses a specific type of attention mechanism called the scaled dot-product attention, where the attention weights are computed as a dot product between the query and the key, followed by scaling and then applying the softmax function.

3. **Multi-Head Attention**: Instead of using one set of attention weights, the Transformer employs multiple sets, allowing the model to focus on different parts of the input for different tasks or reasons. Each "head" captures different types of relationships, and their outputs are concatenated and linearly transformed.

4. **Positional Encoding**: Since the Transformer doesn't have a built-in sense of position (like RNNs or LSTMs), it uses positional encodings to give the model information about the position of tokens in a sequence. These encodings are added to the embeddings at the bottoms of the encoder and decoder stacks.

5. **Stacked Layers**: Both the encoder and decoder consist of a stack of identical layers (default in the original paper was 6 layers for both encoder and decoder). This deep architecture enables the model to learn complex representations.

6. **Feed-Forward Networks**: In addition to the attention mechanism, each layer of the encoder and decoder contains a position-wise fully connected feed-forward network.

7. **Residual Connections**: To prevent the vanishing gradient problem in deep networks, the Transformer uses residual connections around each sub-layer (including self-attention and feed-forward neural network).

8. **Layer Normalization**: Each sub-layer in the model also employs layer normalization to stabilize the activations.

9. **Pointwise Feed-Forward Networks**: These are present in both the encoder and decoder and consist of two linear transformations with a ReLU activation in between.

10. **Encoder-Decoder Structure**: The model follows an encoder-decoder structure, where the encoder processes the input sequence, and the decoder produces the output sequence, making it suitable for tasks like machine translation.

The Transformer model's introduction was groundbreaking and led to the development of models like BERT, GPT, and many others that have defined the state of the art in numerous NLP tasks.

## What is Cross-Lingual Masked Language Modeling? How is it achieved?
Cross-Lingual Masked Language Modeling (XLM) is an approach to train language models that can understand and generate text across multiple languages. It's an advancement over monolingual models that typically work on a single language. The idea behind XLM is to leverage data from various languages to build models that can perform tasks in languages with limited training data by benefiting from the knowledge acquired from languages with abundant data.

### How is Cross-Lingual Masked Language Modeling achieved?

1. **Masked Language Modeling (MLM)**: The basic principle is borrowed from the BERT model, where some of the words in a sentence are masked (or hidden), and the model is trained to predict them. For cross-lingual modeling, sentences from multiple languages are taken, and words are masked in them. The model then learns to predict these masked words, gaining knowledge about the structure and semantics of multiple languages in the process.

2. **Translation Language Modeling (TLM)**: This is an extension of the MLM task. In TLM, a pair of sentences (one in the source language and its translation in the target language) are concatenated, and words from both sentences are masked. The model then predicts these masked words, learning about the relationships between languages and how concepts in one language map to another.

3. **Shared Vocabulary and Subword Units**: One challenge with cross-lingual models is handling the vast vocabulary of multiple languages. To address this, XLM often uses subword units (like Byte-Pair Encoding or BPE) to build a shared vocabulary that represents common subword units across languages. This allows the model to handle multiple languages without a prohibitively large vocabulary and also enables the model to generalize better to low-resource languages.

4. **Large-scale Multilingual Data**: For XLM to be effective, it requires training on data from multiple languages. Often, large multilingual corpora (like the Wikipedia data for various languages) are used. The more diverse the training data, the better the model's capability to generalize across languages.

5. **Fine-tuning on Specific Tasks**: Once the XLM is pre-trained, it can be fine-tuned on specific tasks, like cross-lingual classification, translation, or named entity recognition. The advantage is that even if fine-tuning data is available in one language, the model can potentially perform well on the same task in other languages.

Models like XLM-R (a variant of XLM) have been shown to achieve state-of-the-art results on several cross-lingual benchmarks. The beauty of such models is their ability to transfer knowledge from high-resource languages to low-resource ones, thereby democratizing access to state-of-the-art NLP technologies across languages.

## What is Beam Search?

Beam search is a search algorithm used in many sequence-to-sequence tasks, such as machine translation, speech recognition, and text generation. It's designed to improve the quality of the output sequences by exploring multiple possibilities.

### Basic Idea:
When generating sequences token by token (like translating a sentence word by word), a naive approach is to always pick the most probable next token at each step. This method is called "greedy decoding." However, this approach can lead to suboptimal results because an initially probable token might not lead to the best complete sequence.

Beam search addresses this problem by keeping track of multiple possible sequences (called "candidates" or "hypotheses") at each step and further expanding upon these candidates in the subsequent steps.

### How Beam Search Works:

1. **Initialization**: Start with an initial token, usually the "start of sequence" token.

2. **Expand**: For each candidate sequence in your list, predict the next possible tokens and their probabilities.

3. **Prune**: Retain only the top "beam width" (a pre-defined number) of these sequences based on their probabilities. This "beam width" determines how many candidates are kept at each step.

4. **Continue**: Repeat the expansion and pruning steps until a stopping criterion is met, such as reaching a maximum sequence length or all candidates have an "end of sequence" token.

5. **Select**: At the end of the process, the candidate with the highest probability is typically chosen as the output sequence.

### Pros and Cons of Beam Search:

**Pros**:
- Beam search often produces better results than greedy decoding because it explores multiple paths rather than committing to the first most probable token.
  
- It's a good compromise between exhaustive search (which considers all possible sequences) and greedy decoding (which considers only one sequence). Exhaustive search is computationally expensive and often infeasible for long sequences.

**Cons**:
- It's still an approximate search method. There's no guarantee that beam search will always find the most probable sequence, especially if the beam width is set too small.

- Increasing the beam width can lead to improved accuracy but at the cost of increased computational complexity.

- In practice, beam search can sometimes favor longer or shorter sequences disproportionately, which might require length normalization or other heuristics to address.

In summary, beam search is a heuristic algorithm that aims to balance the trade-off between computational efficiency and output quality in sequence-to-sequence tasks.

