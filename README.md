# MoINN

MoINN, short for Moiety Identification Neural Network, is designed to automate the 
identification of chemical moieties (fundamental building blocks of molecules)
from machine learned representations. It has been shown that MoINN is suitable for
the automatic construction of coarse-grained force fields, selecting representative 
entries in chemical databases, as well as the identification of reaction coordinates. 
The design of MoINN (differantiable, transferable w. r. t. molecule size) makes it very 
versatile and paves the way for many other interesting applications.


##### Requirements:
- python 3.8
- Atomic Simulation Environment (ASE)
- NumPy 1.23.3
- PyTorch 1.13.1
- H5py
- tensorboard
- tqdm
- RDKit
- NetworkX
- SchNetPack 1.0

_**Note: We recommend using a GPU for training the neural networks.**_


## Installation


You can install the most recent code from our repository:

```
git clone https://github.com/jnsLs/MoINN.git
cd MoINN
pip install .
```


## Getting started

In the following, we show the workflow of training and evaluating MoINN. Here,
we focus on using the CLI to train on the QM9 dataset, but the same procedure 
applies for other datasets as well. First, create a working directory, where 
all data and runs will be stored:

```
mkdir moinn_workdir
cd moinn_workdir
```

Then, the training of a pretrained MoINN model with default settings for QM9 can be started by:

```
moinn_train.py --model_dir ./run0 --datapath ./data/qm9.db --tradeoff_warmup_epochs 0 0 --split 1000 100 --clustering_mode pretrained --rep_model_dir /directory/of/MPNN/model --manual_seed 3 --cuda
```

The dataset will be downloaded automatically to `spk_workdir/data`, if it does not exist yet.
Then, the training will be started. We provide a pretrained SchNet model, which can serve as a 
representation model for the training of a pretrained MoINN model. It is stored in ```trained_models/schnet_model```.

Training of an end-to-end MoINN model is performed with the following command:

```
moinn_train.py --model_dir ./run1 --datapath ./data/qm9.db --tradeoff_warmup_epochs 100 130 --split 1000 100 --clustering_mode end_to_end --manual_seed 3 --cuda
```

Finally, we can evaluate the respective MoINN model:

```
moinn_eval_qm9.py --model_dir ./run0 --datapath ./data/qm9.db
```
The evaluation results will be stored in the model directory in a folder named ```eval```.

## References

* [1] Lederer, J., Gastegger, M., Schütt, K. T., Kampffmeyer, M., Müller, K.-R., & Unke, O. T.  
*Automatic identification of chemical moieties.*
Physical Chemistry Chemical Physics (2023). (https://doi.org/10.1039/D3CP03845A)
