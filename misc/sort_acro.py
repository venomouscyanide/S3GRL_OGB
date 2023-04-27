def sort_acro(acro_raw):
    split = acro_raw.split('\n')
    split = list(filter(lambda x: x.strip() != "", split))
    sorted_acro = sorted(split, key=lambda x: x.split('{\dotfill')[1].strip()[:-1])
    print("\n".join(sorted_acro).replace('\\\\', '\\'))


if __name__ == '__main__':
    acro_raw = r"""
    \acro{SoP}{\dotfill Subgraphs of Powers}
  \acro{S3GRL}{\dotfill Scalable Simplified  Subgraph Representation Learning}
  \acro{PoS}{\dotfill Powers of Subgraphs}
  \acro{SGRL}{\dotfill Subgraph Representation Learning}
  \acro{SGRLs}{\dotfill Subgraph Representation Learning approaches}
  \acro{GRL}{\dotfill Graph Representation Learning}
  \acro{MLP}{\dotfill Multilayer Perceptron}
  \acro{GNNs}{\dotfill Graph Neural Networks}
  \acro{MPGNN}{\dotfill Message Passing Graph Neural Networks}
  \acro{CNN}{\dotfill Convolutional Neural Network}
  \acro{RNN}{\dotfill Recurrent Neural Network}
  \acro{i.i.d}{\dotfill Independent and Identically Distributed}
  \acro{AA}{\dotfill Adamic Adar}
  \acro{PA}{\dotfill Preferential Attachment}
  \acro{RA}{\dotfill Resource Allocation}
  \acro{SGD}{\dotfill Stochastic Gradient Descent}
  \acro{NLP}{\dotfill Natural Language Processing}
  \acro{BFS}{\dotfill Breadth-first Search}
  \acro{DFS}{\dotfill Depth-first Search}
  \acro{WL-Test}{\dotfill Weisfeiler-Lehman Isomorphism Test}
  \acro{DRNL}{\dotfill Double Radius Node Labeling}
  \acro{GCN}{\dotfill Graph Convolutional Networks}
  \acro{GAT}{\dotfill Graph Attention Networks}
  \acro{GIN}{\dotfill Graph Isomorphism Network}
  \acro{DE-GNN}{\dotfill Distance Encoding Graph Neural Network}
  
  \acro{MRR}{\dotfill Mean Reciprocal Rank}
  \acro{HR}{\dotfill Hit Rate}
  \acro{roc-auc}{\dotfill Area under the ROC Curve}
  \acro{AUC}{\dotfill Area under the ROC Curve}
    """
    sort_acro(acro_raw)
