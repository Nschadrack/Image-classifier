# Image-classifier
This repo contains files training the neural network model, predicting flowers using neural network trained

### train.py
This script is used for training the network model and saves the checkpoint
The scripts requires some arguments(parameters) and optional arguments

You can type ***python train.py -h*** to see the usage of this script

**N.B:** data_diractory(data_dir) must contain three sub_directories, which are:
1. train
2. valid
3. test

In each sub-directory 
is where the images datasets stored, training dataset, validation dataset, and testing dataset respectively.

You must also provide save_directory, this is the directory where the model checkpoint will be saved.

### predict.py
This script is used for predicting the flower using trained model saved in the checkpoint
The script requires some positional arguments(parameter) and optional arguments

You can type ***python predict.py -h*** to see the usage of this script

### helper.py
This is script contains functions for helping the **train.py** and **predict.py** scripts

### workspace_utils.py
This is script contains functions to help keep kernel keep running
