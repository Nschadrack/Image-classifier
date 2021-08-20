import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models, datasets, transforms
from torch import optim
from workspace_utils import active_session
import os

from helper import train_arguments


class FlowerClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.shape[0], -1)
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def build_model(arch_model='vgg16', hidden_layers=[2048, 1024, 512], output_size=102, dropout=0.2, lr=0.001):

    """
    The function for building the model

    parameters
    ----------
    arch_model: prettrained model architecture to use
    input_size: number of inputs at the first hidden layer(number of nodes in the first hidden layer)
    output_size: number of output classes
    hidden_layers: The list of hidden layers inputs

    Example: [256, 128, 64]: hidden layers


    dropout: probaility of dropout
    lr: learning rate

    return
    -------
    returns model, optimizer, criterion
    """

    arch_models = {
                    "vgg16": [25088, "classifier"],
                    "alexnet": [9216, "classifier"],
                    "resnet101": [2048, "fc"],
                    "densenet121": [1024, "classifier"]
                }

    if arch_model in arch_models:

        classifier = FlowerClassifier(arch_models[arch_model][0], output_size, hidden_layers, dropout)

        model = models.__dict__[arch_model](pretrained=True)

        if arch_model in ["vgg16", "alexnet", "densenet121"]:
            model.classifier = classifier

        elif arch_model == "resnet101":
            model.fc = classifier


        for param in model.parameters():
            param.require_grad = False


        if arch_model in ["vgg16", "alexnet", "densenet121"]:
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

        elif arch_model == "resnet101":
            optimizer = optim.Adam(model.fc.parameters(), lr=lr)

        criterion = nn.NLLLoss()

        return arch_model, model, optimizer, criterion

    return None, None, None, None


def validate(model, test_validation_data, criterion, device_mode):
    """
    the function for the testing the network

    parameters
    -----------
    model: the network model to test
    test_validation_data: the data for testing/validating the model
    criterion: loss function object

    return
    ------
    The function returns the accuracy and Testing/validation loss
    """
    test_or_validation_loss = 0
    accuracy = 0

    if device_mode:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device) #switching the model to the appropriate mode

    with torch.no_grad():
        model.eval() # the set model to the validating or testing mode
        for images, labels in test_validation_data:
            images, labels = images.to(device), labels.to(device)
            lgps = model.forward(images)
            test_or_validation_loss += float(criterion(lgps, labels))
            ps = torch.exp(lgps)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()

    return (test_or_validation_loss / len(test_validation_data)) * 100, (accuracy / len(test_validation_data)) * 100



def train(model, train_data, test_data, optimizer, criterion, device_mode, epochs=1, print_at_every=45):
    """
    The function for training new network

    parameters:
    -----------
    model: the network model to train
    train_data: the data for training the network and it must be torch loader
    test_data: the data for the testing/validating the network
    optimizer: the optimizer to the network
    criterion: loss object for anyway of Torch Loss functions
    example: nn.NLLLoss()
    epochs: number of epochs for training
    print_at_every: at what number of batches to print accuracy rate, validation loss and training loss
    device: the working environment mode which is either GPU which is cuda or CPU
    """
    running_loss = 0

    if device_mode:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    for epoch in range(epochs):
        step = 0
        for images, labels in train_data:
            step +=1
            images, labels = images.to(device), labels.to(device)
            lgps = model.forward(images)
            loss = criterion(lgps, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % print_at_every == 0:
                validation_loss, accuracy = validate(model, test_data, criterion, device_mode)

                print("""
                    Epoch: {} / {} ... Training loss: {:.2f}% ... Validation loss: {:.2f}% ... Validation accuracy: {:.2f}%
                """.format(epoch+1, epochs,
                          (running_loss / print_at_every * 100),
                          validation_loss, accuracy))

                running_loss = 0


def save_checkpoint(model, optimizer, image_datasets, save_path, arch_model="vgg16", output_size=102, epochs=1, dropout=0.2, lr=0.001):
    """
    The function for saving a checkpoint
    """
    if save_path is None:
        raise Exception("You didn't provide the directory where to save the checkpoint")

    model.class_to_idx = image_datasets.class_to_idx
    if arch_model in ["vgg16", "alexnet", "densenet121"]:
        hidden_layers = [each.out_features for each in model.classifier.hidden_layers]

    elif arch_model == "resnet101":
        hidden_layers = [each.out_features for each in model.fc.hidden_layers]


    checkpoint = {
        "epochs": 1,
        "class_to_idx": model.class_to_idx,
        "optimizer": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
        "dropout": dropout,
        "hidden_layers": hidden_layers,
        "output_size": output_size,
        "learning_rate": lr,
        "arch_model": arch_model
    }

    save_path2 = os.path.join(save_path, "checkpoint.pth")
    if "checkpoint.pth" in os.listdir(save_path):
        os.remove(save_path2)

    torch.save(checkpoint, save_path2)



if __name__ == "__main__":
    args = train_arguments()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomRotation(45),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "valid": transforms.Compose([transforms.Resize(250),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "test": transforms.Compose([transforms.Resize(250),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                      }


    # Load the datasets with ImageFolder
    image_datasets = { x: datasets.ImageFolder(data_dir + "/" + x,
                                               transform=data_transforms[x]) for x in ["train", "valid", "test"] }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True) for x in ["train", "valid", "test"]}

    with active_session():
        arch_model, model, optimizer, criterion = build_model(arch_model=args.arch,
                                                              hidden_layers=args.hidden_units,
                                                              output_size=args.output_size,
                                                              dropout=args.dropout,
                                                              lr=args.learning_rate)

        train(model, dataloaders["train"], dataloaders["valid"], optimizer, criterion, device_mode=args.gpu, epochs=args.epochs, print_at_every=args.print_every)


        save_checkpoint(model, optimizer, image_datasets["train"], args.save_dir, arch_model)
