# CAT

This is a PyTorch implementation of CAT which has been submitter to [Information Science](https://arxiv.org/abs/2312.08672).

## Abstract
Local Attention-guided Message Passing Mechanism (LAMP) adopted in Graph Attention Networks (GATs) is designed to adaptively learn the importance of neighboring nodes for better local aggregation on the graph, which can bring the representations of similar neighbors closer effectively, thus showing stronger discrimination ability. However, existing GATs suffer from a significant discrimination ability decline in heterophilic graphs because the high proportion of dissimilar neighbors can weaken the self-attention of the central node, jointly resulting in the deviation of the central node from similar nodes in the representation space. This kind of effect generated by neighboring nodes is called the Distraction Effect (DE) in this paper. To estimate and weaken the DE of neighboring nodes, we propose a Causally graph Attention network for Trimming heterophilic graph (CAT). To estimate the DE, since the DE are generated through two paths (grab the attention assigned to neighbors and reduce the self-attention of the central node), we use Total Effect to model DE, which is a kind of causal estimand and can be estimated from intervened data; To weaken the DE, we identify the neighbors with the highest DE (we call them Distraction Neighbors) and remove them. We adopt three representative GATs as the base model within the proposed CAT framework and conduct experiments on seven heterophilic datasets in three different sizes. Comparative experiments show that CAT can improve the node classification accuracy of all base GAT models. Ablation experiments and visualization further validate the enhancement of discrimination ability brought by CAT.
