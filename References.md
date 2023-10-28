# References

## What are some state of the art deep learning models for network modeling?

Deep learning models for network modeling, often referred to as graph neural networks (GNNs) or graph representation learning, have become an essential area of research due to the ubiquity of graph-structured data in various domains. Here are some state-of-the-art deep learning models and approaches designed for network or graph modeling:

1. **Graph Convolutional Networks (GCN)**:
   - Proposed by Thomas Kipf and Max Welling, GCNs use a form of convolution tailored to graph-structured data.
   - Reference: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In ICLR.

2. **GraphSAGE (Graph Sample and Aggregation)**:
   - This method learns a function to sample and aggregate features from a node's neighbors, allowing for inductive learning on large graphs.
   - Reference: Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In NeurIPS.

3. **Graph Attention Networks (GAT)**:
   - GATs use self-attention mechanisms to weigh neighbors' features in the aggregation process.
   - Reference: Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. In ICLR.

4. **Deep Graph Infomax (DGI)**:
   - DGI maximizes mutual information between patch representations and corresponding high-level summaries of graphs to obtain unsupervised node representations.
   - Reference: Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In ICLR.

5. **Graph Neural Networks with Jumping Knowledge (JK-Nets)**:
   - Addresses the challenge of deep GNNs by allowing information to be selectively used across layers.
   - Reference: Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?. In ICLR.

6. **ChebNet**:
   - Uses Chebyshev polynomials to generalize convolution operations on graphs.
   - Reference: Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. In NeurIPS.

7. **Message Passing Neural Network (MPNN)**:
   - MPNNs generalize various GNNs into a unified framework where nodes exchange information in the form of "messages".
   - Reference: Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In ICML.

8. **Heterogeneous Graph Neural Networks**:
   - Models like R-GCN, HAN, and MAGNN handle graphs with various types of nodes and edges.
   - References:
     - Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. In ESWC.
     - Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous graph attention network. In WWW.
     - Fu, R., Zhang, J., & Li, J. (2020). MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding. In WWW.

9. **Dynamic Graph Networks**:
   - EvolveGNN, DySAT, and DGCNs handle dynamic graphs where the topology or node/edge attributes change over time.
   - References:
     - Trivedi, R., Dai, H., Wang, Y., & Song, L. (2019). Representation learning over dynamic graphs. In AIES.

These models represent only a snapshot of the burgeoning field of graph representation learning. Researchers continue to introduce new methods to address challenges such as scalability, handling of heterogeneous graphs, dealing with dynamic graphs, and unsupervised or semi-supervised learning on graphs.

## Summarize key breakthroughs of deep learning for network modeling in recent decades. Provide references.

Deep learning for network modeling, particularly in the form of Graph Neural Networks (GNNs), has experienced significant advances in recent years. Here are key breakthroughs in this domain:

1. **Graph Convolutional Networks (GCN)**:
   - Introduced a novel convolution operation tailored for graph-structured data, enabling semi-supervised node classification on graphs.
   - Reference: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In ICLR.

2. **Graph Attention Networks (GAT)**:
   - Introduced an attention mechanism for graph nodes, allowing different weights for different neighboring nodes, thereby providing adaptive aggregation.
   - Reference: Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. In ICLR.

3. **Message Passing Neural Networks (MPNN)**:
   - A general framework for graph representation learning by defining a message-passing scheme across nodes.
   - Reference: Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In ICML.

4. **GraphSAGE (Graph Sample and Aggregation)**:
   - Addressed inductive learning challenges on graphs, enabling embeddings for nodes not seen during training.
   - Reference: Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In NeurIPS.

5. **Heterogeneous Graph Neural Networks**:
   - Models such as R-GCN, HAN, and MAGNN designed specifically for heterogeneous graphs with different node and edge types.
   - References:
     - Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). Modeling relational data with graph convolutional networks. In ESWC.
     - Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous graph attention network. In WWW.
     - Fu, R., Zhang, J., & Li, J. (2020). MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding. In WWW.

6. **Dynamic Graph Representation Learning**:
   - Addressed the challenge of graphs that change over time through evolving topologies or attributes.
   - Reference: Trivedi, R., Dai, H., Wang, Y., & Song, L. (2019). Representation learning over dynamic graphs. In AIES.

7. **Deep Graph Infomax (DGI)**:
   - Introduced an unsupervised learning approach for graphs, using mutual information maximization between local and global graph representations.
   - Reference: Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In ICLR.

8. **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**:
   - Designed for skeleton-based action recognition, capturing both spatial and temporal graph structures.
   - Reference: Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. In AAAI.

These breakthroughs have propelled GNNs to the forefront of network modeling, providing tools to handle a wide variety of graph-structured data and related challenges. The field continues to evolve with ongoing research addressing scalability, interpretability, and other domain-specific requirements.