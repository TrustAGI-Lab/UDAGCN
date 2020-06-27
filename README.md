## Unsupervised Domain Adaptive Graph Convolutional Networks

This repository contains the author's implementation in PyTorch for the paper "Unsupervised Domain Adaptive Graph Convolutional Networks".


## Dependencies

- Python (>=3.6)
- Torch  (>=1.2.0)
- numpy (>=1.16.4)
- torch_scatter (>= 1.3.0)
- torch_geometric (>= 1.3.0)

## Datasets
The data folder includes different domain data. The preprocessed data can be found in [Google Drive](https://drive.google.com/file/d/1DzQ3QN9yjQxU4vtYkXyCiJKFw7oCCPSM/view?usp=sharing). 

The orginal datasets can be founded from [here](https://www.aminer.cn/citation).

## Implementation

Here we provide the implementation of UDA-GCN, along with two domain datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for DBLP domain and ACM domian(can be found in [Google Drive](https://drive.google.com/file/d/1DzQ3QN9yjQxU4vtYkXyCiJKFw7oCCPSM/view?usp=sharing));
 - `dual_gnn/` contains the implementation of the Global GCN and Local GCN;

 Finally, `UDAGCN_demo.py` puts all of the above together and can be used to execute a full training run on the datasets.

## Process
 - Place the datasets in `data/`
 - Change the `dataset` in `UDAGCN_demo.py` .
 - Training/Testing:
 ```bash
 python UDAGCN_demo.py
 ```
# Citation
```
@inproceedings{wu2020UDAGCN
author={Man Wu and Shirui Pan and Chuan Zhou and Xiaojun Chang and Xingquan Zhu},
title={Unsupervised Domain Adaptive Graph Convolutional Networks},
journal={{WWW} '20: The Web Conference},
year={2020}
}
```