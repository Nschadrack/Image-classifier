"""
This module contains functions for helping the project
for plotting, getting commandline arguments
"""

import torch
import numpy as np
from torchvision import  transforms
from workspace_utils import active_session
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transforms_ = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stds)])

    # Process a PIL image for use in a PyTorch model
    with Image.open(image) as pil_image:

        tensor_image = transforms_(pil_image)

    return tensor_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title is not None:
        ax.set_title(title)

    return ax



def train_arguments():
    """
    The function for processing training commandline arguments
    """
    parser = argparse.ArgumentParser(description="The program for receiving arguments to train the model")

    parser.add_argument("data_dir", action="store", help="Directory which contains images for testing and training")
    parser.add_argument("--save_dir", action="store", dest="save_dir", help="Directory where the checkpoint will be saved")
    parser.add_argument("--arch", action="store", dest="arch", default="vgg16", help="The model architecture for transfering network")
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", type=float, default=0.001,
                        help="The learning rate for optimizing the network")
    parser.add_argument("--hidden_units", action="append", dest="hidden_units", type=int, default=[],
                        help="The hidden layers for the network")
    parser.add_argument("--epochs", action="store", dest="epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--gpu", action="store_true", dest="gpu", default=False, help="do you want to use GPU")
    parser.add_argument("--output_size", action="store", default=102, dest="output_size",
                        type=int, help="the number of output neurons")
    parser.add_argument("--dropout", action="store", default=0.01, dest="dropout",
                        type=float, help="probability dropout")
    parser.add_argument("--print_every", action="store", dest="print_every", default=45,
                        type=int, help="at which every step to print report of accuray, validation and training rate")

    return parser.parse_args()



def predict_arguments():
    """
    The function for processing the predicting commandline argument
    """
    parser = argparse.ArgumentParser(description="The program for receiving arguments to predict the input")

    parser.add_argument("input", action="store", help="The file path to the image to predict")
    parser.add_argument("checkpoint", action="store", help="The file path to the checkpoint of the model to use for predicting")
    parser.add_argument("--top_k", action="store", dest="top_k", type=int, default=5,
                        help="The number of top classes and probabilities to return")
    parser.add_argument("--category_names", action="store", dest="category_names",
                        help="The file path to the json file contains labels mapping to the names of flowers")
    parser.add_argument("--gpu", action="store_true", default=False, help="do you want to use GPU")


    return parser.parse_args()
