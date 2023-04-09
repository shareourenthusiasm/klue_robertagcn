# Klue with RobertaGCN

## Result 

### Not fold valid.

|**Model** | **KLUE-ynat** |
| ------------ | ---- |
| *RoBERTaGCN* | **87.25** |

### 5-fold Avg. (date. 2023/04/09)

|**Model** | **KLUE-ynat** |
| ------------ | ---- |
| *BERT* | **86.31** |
| *RoBERTa* | **85.90** |
| *BertGCN* | **86.39** |
| *RoBERTaGCN* | **86.25** |
| *BertGCN_TBGC* | **86.99** |
| *RoBERTaGCN_TBGC* | **86.72** |

## How to Use

### Model

1. Run klue_data_convert.py

2. Run build_graph.py

3. Run robertagcn_klue.py

## Requirements

- `dgl-cu113 == 0.9.1.post1`
- `ignite == 1.1.0`
- `python == 3.6.9`
- `torch == 1.10.0+cu113`
- `scikit-learn =< 0.24.2`
- `transformers =< 4.18.0`
- `numpy =< 1.19.5`
- `networkx == 2.5.1`

## Problem

When constructing an adjacency matrix, there is a data leakage problem. (The paperwithcode doesn't seem to consider that issue.)

## Reference

### Backbone Model

- [klue/roberta-base](https://huggingface.co/klue/roberta-base)

### Dataset

- [KLUE-ynat](https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000066/data/ynat-v1.1.tar.gz)

### Citation

- [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727)
- [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/abs/2105.09680)

## Workers

### [노영준](https://github.com/youngjun-99)
- Change build graph method
- Preprocessing

### [박정원](https://github.com/jardin00)
- Finetune RoBERTaGCN
