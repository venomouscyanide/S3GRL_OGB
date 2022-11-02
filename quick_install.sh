#!/bin/sh
# this installation is very specific to Mac M1 chip, however, simple pip installs should work on any silicon
conda create --name sweal_3_9 python=3.9 # don't use python 3.10 as it breaks the PyCharm debugger flow: https://youtrack.jetbrains.com/issue/PY-52137
conda activate sweal_3_9
conda install pytorch torchvision -c pytorch
export CC=/opt/homebrew/Cellar/llvm/13.0.1_1/bin/clang # point to the brew clang compiler
pip install torch-sparse==0.6.13
pip install torch-scatter torch-sparse  torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install matplotlib
pip install ogb
pip install networkx
pip install pytorch_memlab
pip install class_resolver
pip install fast-pagerank
pip install sklearn
pip install graphistry
pip install scipy