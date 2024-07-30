# RELIEF
This repository contains the PyTorch implementation of the paper *RELIEF: Reinforcement Learning Empowered Graph Feature Prompt Tuning*. 

Inspired by the marginal effect of increasing prompt token length on performance improvement in LLMs, we propose **RELIEF**, a **RE**inforcement **L**earn**I**ng **E**mpowered graph **F**eature prompt tuning method. Our goal is to enhance the performance of pre-trained GNN models on downstream tasks by **incorporating only necessary and lightweight feature prompts** into input graphs.

## Installation

The following packages are required under `Python 3.9`.

```
pytorch 2.0.1
torch-geometric 2.3.1
torch-cluster 1.6.3+pt20cu117
torch-scatter 2.1.2+pt20cu117
torch-sparse 0.6.18+pt20cu117
torch-spline-conv 1.2.2+pt20cu117
rdkit 2022.3.4
scikit-learn 1.2.0
```

## Experiments

### Graph Classification

- **Datasets**: For graph-level tasks, we adopt Chemical datasets to pre-train GNN model. For downstream graph classification tasks, eight binary classification datasets for molecular property prediction are employed. All datasets are referenced from the paper *Strategies for Pre-training Graph Neural Networks* and are available at their [repositery](https://github.com/snap-stanford/pretrain-gnns/tree/master/chem). Please download the chemistry dataset, unzip it while retaining the `dataset` directory, and place it directly under the `graph_level` folder.

- **Pre-training**: We follow the training steps from the paper [*Strategies for Pre-training Graph Neural Networks*](https://github.com/snap-stanford/pretrain-gnns) and [*Graph Contrastive Learning with Augmentations*](https://github.com/Shen-Lab/GraphCL) to obtain four pre-trained GIN models, including Deep Graph Infomax, Attribute Masking, Context Prediction and Graph Contrastive Learning strategies. Pre-trained models are available at the [repository](https://github.com/LuckyTiger123/GPF/tree/main/chem/pretrained_models) of an exisiting baseline work.

- **Baselines**: Fine-tuning, [GPF](https://github.com/LuckyTiger123/GPF), [GPF-plus](https://github.com/LuckyTiger123/GPF), [SUPT<sub>soft</sub>](https://anonymous.4open.science/r/SUPT-F7B1), [SUPT<sub>hard</sub>](https://anonymous.4open.science/r/SUPT-F7B1) and [All in One](https://github.com/sheldonresearch/ProG). 

- **Running**: For each downstream dataset, the four pre-trained GIN models are used, forming 32 graph classification tasks. You can reproduce the experimental results presented in our paper by running `graph_level/run.sh`, where arguments and hyper-paramter settings are provided.


### Node Classification

- **Datasets**: For node-level tasks, we use Cora, Citeseer, Pubmed and Amazon-Co-Buy (Computer and Photo), which we have already saved as processed (SVD) datasets in the `node_level/dataset` folder. Since we extend feature prompt tuning approaches to node-level tasks by operating on the induced k-hop subgraphs of the target nodes, we provide the code for splitting training, validation, and testing subgraphs in `node_level/preprocess.py`. Note that since fine-tuning or other prompt-based methods do not incorporate subgraph-related designs, we also maintained node indices corresponding to training, validation, and testing node sets in the generated file saved within `node_level/subgraph/split_data`.

- **Pre-training**: We use two edge-level pre-training strategies employed in two pioneering work - [GPPT](https://github.com/MingChen-Sun/GPPT) and [GraphPrompt](https://github.com/Starlien95/GraphPrompt), respectively. GPPT used masked edge prediction which is a binary classification pretext task, whereas GraphPrompt used contrastive learning, which determines positive and negative node pairs based on edge connectivity. You can first pre-train GIN models by running `node_level/pretrain_gppt.py` and `node_level/pretrain_gprompt.py`, or directly use the models we provided in the `pretrained_models` directory. 

- **Baselines**: Fine-tuning, [GPPT](https://github.com/MingChen-Sun/GPPT), [GraphPrompt](https://github.com/Starlien95/GraphPrompt), GPF, GPF-plus, SUPT<sub>soft</sub> and SUPT<sub>hard</sub>.

- **Running**: For each downstream dataset, the two pre-trained GIN models are used, forming 10 node classification tasks. You can reproduce the experimental results presented in our paper by running `node_level/run.sh`.