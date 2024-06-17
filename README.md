This is a PyTorch implementation of [CAT](https://www.sciencedirect.com/science/article/pii/S0020025524008302) which has been accepted by Information Science.

## Abstract
The local attention-guided message passing mechanism (LAMP) adopted in graph attention networks (GATs) can adaptively learn the importance of neighboring nodes and perform local aggregation better, thus demonstrating a stronger discrimination ability. However, existing GATs suffer from significant discrimination ability degradations in heterophilic graphs. The reason is that a high proportion of dissimilar neighbors can weaken the self-attention of the central node, resulting in the central node deviating from its similar nodes in the representation space. This type of influence caused by neighboring nodes is referred to as Distraction Effect (DE) in this paper. To estimate and weaken the DE induced by neighboring nodes, we propose a Causal graph Attention network for Trimming heterophilic graphs (CAT). To estimate the DE, since DE is generated through two paths, we adopt the total effect as the metric for estimating DE; To weaken the DE, we identify the neighbors with the highest DE (we call them Distraction Neighbors) and remove them. We adopt three representative GATs as the base model within the proposed CAT framework and conduct experiments on seven heterophilic datasets of three different sizes. Comparative experiments show that CAT can improve the node classification accuracies of all base GAT models. Ablation experiments and visualization further validate the enhanced discrimination ability of CATs. In addition, CAT is a plug-and-play framework and can be introduced to any LAMP-driven GAT because it learns a trimmed graph in the attention-learning stage, instead of modifying the model architecture or globally searching for new neighbors.

## How to run

Thank you for your attention to our work. If this work is helpful to your research, please cite the following paper:
```
@article{HE2024120916,
title = {CAT: A Causal Graph Attention Network for Trimming Heterophilic Graphs},
journal = {Information Sciences},
pages = {120916},
year = {2024},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2024.120916},
url = {https://www.sciencedirect.com/science/article/pii/S0020025524008302},
author = {Silu He and Qinyao Luo and Xinsha Fu and Ling Zhao and Ronghua Du and Haifeng Li}
}
```
