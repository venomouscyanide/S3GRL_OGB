S3GRL-OGB (Scalable Simplified Subgraph Representation Learning)
===============================================================================
S3GRL (Scalable Simplified Subgraph Representation Learning) is a subgraph representation learning (SGRL) framework aimed at faster link prediction. S3GRL introduces subgraph sampling and subgraph diffusion operator pairs that allow for fast precomputation leading to faster runtimes.

N.B: This repository holds the codes required for extending https://github.com/venomouscyanide/S3GRL for the OGB datasets. The original repository for S3GRL (and the first version of the paper) does not support running on the OGB datasets.

## Our Model (S3GRL, PoS Plus model) Results

|          	| ogbl-collab  	| ogbl-ddi     	| ogbl-vessel  	| ogbl-citation2 	| ogbl-ppa 	|
|----------	|--------------	|--------------	|--------------	|----------------	|----------	|
|          	| HR@50        	| HR@20        	| roc-auc      	| MRR            	| HR@100   	|
| PoS Plus 	| 66.83 ± 0.30 	| 22.24 ± 3.36 	| 80.56 ± 0.06 	| 88.14 ± 0.08    | *        	|

"*" - this experiment is in the works.

`Disclaimer`: It is to be noted that the validation dataset split and the training dataset split for some datasets are not consumed in full. Due to some computational constraints, and to speed up the overall runs, we sometimes only consume a percentage of the training and validation splits provided by OGB. However, it is to be noted that the testing dataset split is always consumed in full (i.e., 100% of the testing dataset is always used for inference of our learnt models).

Please see the values for `val_percent` and `train_percent` in our [configuration files](https://github.com/venomouscyanide/S3GRL_OGB/tree/main/configs/ogbl) provided. As stated above, `test_percent` is always 100%.

For reproducing the above results, run the following commands:

### ogbl-collab

`
python sgrl_run_manager.py --config configs/ogbl/ogbl_collab.json --results_json ogbl_collab_results.json
`

### ogbl-ddi

`
python sgrl_run_manager.py --config configs/ogbl/ogbl_ddi.json --results_json ogbl_ddi_results.json
`

### ogbl-vessel

`
python sgrl_run_manager.py --config configs/ogbl/ogbl_vessel_signk_3.json --results_json ogbl_vessel_results.json
`

`ogbl-vessel` requires a pre-trained embedding file (pretrained n2v embeddings). This embedding file is concatenated with the initial 3-d nodal embeddings. Please download the file from https://drive.google.com/file/d/1b_lDPUQKRuT51tlt39XDEVQ_g8-KW_zr/ and place it in the folder `Emb`, creating the folder `Emb` if it does not exist. To the best of my knowledge, ogbl-vessel is the only dataset where there is a need to download pretrained embeddings first, and then run our codes. 

This pretrained embedding file is created using [Julian McGinnis's](https://github.com/jqmcginnis) node2vec pretraining code available [here](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/vessel/node2vec.py). You can run this script and rename the output to `pretrained_n2v_ogbl_vessel.pt` and place it in `Emb` as well. 

See https://github.com/venomouscyanide/S3GRL_OGB/blob/main/sgrl_link_pred.py#L1093 for how pretrained n2v embeddings are consumed for `ogbl-vessel`.

### ogbl-citation2

`
python sgrl_run_manager.py --config configs/ogbl/ogbl_citation2.json --results_json ogbl_citation2_results.json
`

### ogbl-ppa

`
python sgrl_run_manager.py --config configs/ogbl/ogbl_ppa.json --results_json ogbl_ppa_results.json
`


## Citing Us/BibTex
Please cite our work if you find it useful in any way.

```
@misc{louis2023simplifying,
      title={Simplifying Subgraph Representation Learning for Scalable Link Prediction}, 
      author={Paul Louis and Shweta Ann Jacob and Amirali Salehi-Abari},
      year={2023},
      eprint={2301.12562},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The PDF of our preprint is available at https://arxiv.org/pdf/2301.12562.pdf.

## S3GRL architecture in brief
<img width="1260" alt="Screenshot 2023-01-28 at 3 19 09 PM" src="https://user-images.githubusercontent.com/14299839/215289015-e437d5d5-9df7-48b4-842a-932d4a0c7fc2.png">

Our S3GRL framework: In the preprocessing phase (shown by the shaded blue arrow), first multiple subgraphs are extracted around the target nodes $u$ and $v$ (shaded in blue) by various sampling strategies.  Diffusion matrices are then created from extracted subgraph adjacency matrices by predefined diffusion operators (e.g., powers of subgraphs in this figure). Each diffusion process involves the application of the subgraph diffusion matrix on its nodal features to create the matrix ![CodeCogsEqn](https://user-images.githubusercontent.com/14299839/215292449-7d2eb4ed-f482-4125-9a4f-6f0a0c7e7367.svg). The operator-level node representations of selected nodes (with a red border in raw data) are then aggregated for all subgraphs to form the joint ![CodeCogsEqn(4)](https://user-images.githubusercontent.com/14299839/215292690-ea451ed5-07a8-4b1c-b43a-40a868f23e3c.svg) matrix. The selected nodes in this example are the target nodes ![CodeCogsEqn(1)](https://user-images.githubusercontent.com/14299839/215292532-dd63afe8-5d7f-46bf-9a1e-a60894d4a7d6.svg)
, and their common neighbor $d$. In the learning phase (as shown by the shaded red arrow), the joint matrix ![CodeCogsEqn(4)](https://user-images.githubusercontent.com/14299839/215292690-ea451ed5-07a8-4b1c-b43a-40a868f23e3c.svg) undergoes dimensionality reduction followed by pooling using center pooling (highlighted by blue-border box) and common neighbor pooling (highlighted by purple-border box). Finally, the target representation ![CodeCogsEqn(2)](https://user-images.githubusercontent.com/14299839/215292664-86ab68a1-24f6-4f4e-9642-fcfb42f22755.svg)
 is transformed by an MLP to a link probability $P_{uv}$.


A higher-quality architecture diagram can be found here: [SG3RL_arch.pdf](https://github.com/venomouscyanide/S3GRL/files/10528180/SG3RL_arch.pdf)


## Getting Started with the dev environment

We use Python 3.8 for our experiments. Some notable Python libraries and their versions are as follows:

- pytorch==1.13.0
- scikit-learn==1.1.3
- scipy==1.9.3
- torch-cluster==1.6.0
- torch-geometric==2.1.0.post1
- torch-scatter==2.0.9
- torch-sparse==0.6.13
- ogb==1.3.5 (started out with 1.3.5 but we switched to 1.3.6 release during development)
- networkx==2.8.8
- ray==2.1.0
- gtrick==0.0.dev2 


Users can refer to the exact conda enviroment used for running all the experiments in https://github.com/venomouscyanide/S3GRL_OGB/blob/main/conda_env/s3grl_env.yml. If you have trouble setting up please raise an issue or reach out via email and we will be happy to assist. Also, if you have an M1-mac silicon please check out https://github.com/venomouscyanide/S3GRL_OGB/blob/main/quick_install.sh.

## Configuration on which all experiments are run on

For all the experiments, we use an Ubuntu server with 50-80 CPU cores, 11 Gb GTX 1080Ti GPU, 377 GB ram (with 500 Gigs of swap) and 2 Tb ROM. 

For supporting different hardware capacities, please try changing (to be higher or lower) the following parameters:

- `batch_size` - controls the number of subgraphs in each mini-batch.
- `hidden_channels` - controls the hidden dimensionality of the model.
- `num_workers` - controls the number of workers utilized in PyTorch's Dataloader.
- `num_hops` - controls the size of subgraphs extracted around each link.
- `sign_k` - controls the number of operators created for each subgraph, for each link.
- `cuda_device` - the index of your CUDA GPU that is to be used for training. In our case, we choose from 0, 1, 2 and 3 due to access to 4 NVIDIA GTX 1080 TIs. Depending upon your setup, number of GPUs and load on each GPU, you might want to revise the value of this parameter.  

All the above parameters will result in a higher computational load if increased.

## Running our codes
All our codes can be run by setting a JSON configuration file. Example configuration files can be found in `configs/` or [here](https://github.com/venomouscyanide/S3GRL_OGB/blob/main/test_config.json). In this example, we run 10 runs of Cora with random seeds ranging from 1-10, with hidden dimensionality of the model set to 256, batch size 32, with r=3, h=3 of the PoS model. Please see the next section for details on each argument.

Once you have the configuration JSON file setup, run our codes using `python sgrl_run_manager.py --config your_config_file_here.json --results_json your_result_file.json`. This command produces `your_result_file.json` which contains the average-out efficacy score using the configuration on the dataset the experiments are being run on. 


## Arguments supported

Currently, our S3GRL framework supports two instances, **PoS** and **SoP**. Both instances are available in `tuned_SIGN.py`. You can add your own instances by modifying this file.

Specific arguments related to our framework:
- `sign_k` - The number of diffusion operators to create (corresponds to r in our paper). Setting this to a higher value increases the time taken for training/prep.

- `sign_type` - Set the type of S3GRL model. Either PoS, SoP or hybrid. PoS is Powers of Subgraphs. SoP is Subgraphs of Powers. Hybrid is devised as a clever way to create `sign_k` number of operators for each hop in 1 to `num_hops` extractions. For example, if `sign_k` is 3 and `num_hops` is 3, for each hop subgraph, we create 3 operators each, for a total of 9 operators per subgraph. Currently, hybrid mode is only used with PoS style learning.

- `optimize_sign` - Choose to use optimized codes for running PoS or SoP. Ideally, always set this to True. Optimized formulations are a way to drop unnecessary rows in storage and are aimed at being faster ways of PoS/SoP learning with virtually no loss of generalization.

- `init_features` - Initialize features for non-attributed datasets. Our S3GRL models require initial node features to work. Choose between 'degree', 'eye' or 'n2v'. This is mainly aimed at the non-attributed datasets. Unlike SEAL, we do not support learnable node embeddings (e.g., DRNL is a trainable embedding lookup). As a result, the user needs to supply the nodes with initial node embeddings for our models. 

- `k_heuristic` - Choose to use `CCN` (center-common-neighbor pooling) for the nodes other than source-target nodes. This boolean decides whether the `CCN` pooling is enabled for PoS or SoP. If `k_heuristic` is set to True, we also calculate the pooled values of the common neighbors for making the link prediction. 

- `k_node_set_strategy` - How to choose the nodes for `CCN`. Either intersection or union. Intersection means you choose the common neighbors. Works when `k_heuristic` is True. Even though the name given is common center pooling, you can set it between `union` or `intersection`. I.e., the intersection of the neighbors or unions of the neighbors of source-target nodes will be taken. In our paper, we use `CCN` which is synonymous with setting `k_node_set_strategy` to `intersection` (common neighbors are just an intersection of the set of connections to source and target nodes). Users can choose to use 'union' if common neighbors are not what they have in mind. Also, for this argument to be considered, you need to set `k_heuristic` to True to enable the Plus versions of PoS/SoP.

- `k_pool_strategy` - How to pool the nodes other than source-target in the forward pass for `CCN` (the nodes you select using `k_node_set_strategy`).  Either mean, sum or max pooling. Works when `k_heuristic` is True. This decides how you want to pool the `union`/`intersection` of source and target nodes' neighbors. We use PyG's mean, max or sum pooling for this. `mean` and `sum` is usually consumed.

- `init_representation` - Use an unsupervised model to train the initial features before running S3GRL. Choose between 'GIC', 'ARGVA', 'GAE', 'VGAE'.
If you want to run an unsupervised model to further refine the initial nodal embeddings before running our models, use this. This is adapted from https://github.com/DaDaCheng/WalkPooling.

- `dynamic_train`, `dynamic_val` and `dynamic_test` - This setting makes the code do the subgraph extractions and S3GRl's operators creation on the fly. This utilizes PyTorch's multiprocessing to do all calculations using multiprocessing. This means no preprocessing and saving S3GRL models data. This is aimed at being faster to see results per epoch. However, if you want to train for a large number of epochs, repeatedly calculating all operators per epoch could be taxing (you are essentially computing the same data for each epoch in dynamic mode). 

- `normalize_feats` - A boolean that determines whether initial nodal features should be row normalized or not.

- `edge_feature` - Uses the gtrick library to inject edge features. Can take values `cn`, `aa`, `ra` and `ad`. They correspond to common neighbor, adamic adar, resource allocation and anchor distance values for all edges.

- `use_mlp` - Boolean value that determines whether to use a single weight or an MLP to learn the subgraph operators of S3GRL.

- `split_by_year` - Boolean value that determines whether to take only training edges after the year 2010 in ogbl-collab. Only used when the dataset is ogbl-collab.

- `node_label` - Which labeling trick to be used to augment the nodal features while creating SoP or PoS operators. Choose between `zo` and `drnl`. `zo` is the zero-one labeling trick, whereas, `drnl` is the double-radius node labeling trick. 

- `use_valedges_as_input` - Boolean value that is used to consume validation edges as part of the training set of edges. Ideally, this boolean results in 100% on the validation dataset split accuracy. Also, only really used for Planetoid and ogbl-collab datasets.

- `delete_dataset` - Delete the dataset created for the run upon finishing. Helps save up disk space.

- `model` - Underlying neural model for training subgraphs. Choose `SIGN` for our S3GRL framework.

- `profile` - Boolean value, which when set will run the PyG profiler during training and inference. Must be training on a CUDA device for this. Expect some significant slowdowns due to overheads related to profiling.


Most of the other arguments like learning rate, epochs, batch size, num hops etc. are pretty self-explanatory. In case you need us to document more arguments, please let us know. Finally, please note that this work is a fork of https://github.com/facebookresearch/SEAL_OGB and carries over some of the original authors' arguments.



## Supported Datasets

We support any PyG dataset. However, the below list covers all the datasets we use in our paper's experiments:

1) Planetoid Dataset (Cora, PubMed, CiteSeer) from "Revisiting Semi-Supervised Learning with Graph Embeddings
    <https://arxiv.org/abs/1603.08861>"
    
2) SEAL datasets (USAir, Yeast etc. introduced in the original paper) from "Link prediction based on graph neural networks https://arxiv.org/pdf/1802.09691.pdf"

3) The OGB Dataset, for which this repo was mainly written.

For datasets 1 and 2, we recommend running using https://github.com/venomouscyanide/S3GRL using the reproduction commands.


## Reporting Issues and Improvements
We currently don't have an issue/PR template. However, if you find an issue in our code please create an issue in GitHub. It would be great if you could give as much information regarding the issue as possible (what command was run, what are the Python package versions, providing full stack trace etc.).  

If you have any further questions, you can reach out to us (the authors) via email and we will be happy to have a conversation. 
[Paul Louis](mailto:paul.louis@ontariotechu.net), [Shweta Ann Jacob](mailto:shweta.jacob@ontariotechu.net).


## Reproducing the Paper's Tabular data

### Reproducing Tables 2, 3 and 5

Please check out https://github.com/venomouscyanide/S3GRL for reproducing Tables 2, 3 and 5 (Table 5 is just a reference to Table 4 in that repo).

### Reproducting Table 4

Table 4 is the primary reason why this repo exists. I.e, for running PoS Plus on the OGB datasets and Planetoid datasets in the fashion of BUDDY (https://github.com/melifluos/subgraph-sketching). 

For running PoS Plus on the Planetoid datasets under the experimental settings set by Chamberlain et.al please use

```
python sgrl_run_manager.py --configs configs/planetoid/cora_citeseer_pubmed.json 
```

For running PoS Plus on any of the OGB datasets use the following template command


```
python sgrl_run_manager.py --configs configs/ogbl/ogbl_*.json
```

Where * is any of the ogb dataset. See the [conf](https://github.com/venomouscyanide/S3GRL_OGB/tree/main/configs/ogbl) folder for all configs. Notable, we have 2 instances for OGB-Vessel. The `ogbl_vessel_signk_3.json` was found to have higher efficacy and is included with the initial configuration `ogbl_vessel.json`, which performs slightly worse in comparison.

## Acknowledgements

The code for S3GRL is based on a clone of SEAL-OGB by Zhang et al. (https://github.com/facebookresearch/SEAL_OGB) and ScaLed by Louis et al. (https://github.com/venomouscyanide/ScaLed). 

The baseline softwares used are adapted from: 
- GIC by Mavromatis et al. (https://github.com/cmavro/Graph-InfoClust-GIC)
- WalkPool by Pan et al. (https://github.com/DaDaCheng/WalkPooling). 
- BUDDY/ELPH by Chamberlain et al. (https://github.com/melifluos/subgraph-sketching). We also used our own fork of this codebase to run BUDDY on ogbl-vessel available at https://github.com/venomouscyanide/subgraph-sketching.

There are also some baseline model codes taken from :
- OGB implementations (https://github.com/snap-stanford/ogb) 
- and other Pytorch Geometric implementations (https://github.com/pyg-team/pytorch_geometric).

Some other libraries/projects that has helped us
- https://github.com/chuanqichen/cs224w
- https://github.com/lustoo/OGB_link_prediction
- https://github.com/zhitao-wang/PLNLP
- https://github.com/yao8839836/ogb_report/
- https://github.com/sangyx/gtrick
- https://github.com/snap-stanford/distance-encoding
- https://github.com/zhf3564859793/GDNN

This repository also uses multiple methods and practices from other developers and researchers. We are thankful for their contributions to this field (and for releasing the codes). Whenever possible, we include a comment linking the sources used.
