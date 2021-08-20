"""
This module contains functions for helping to predict the image and it's class
"""
import json
import torch
from workspace_utils import active_session
import matplotlib.pyplot as plt
from helper import predict_arguments
from train import build_model
from helper import process_image, imshow




def load_checkpoint(filepath):
    """
    The function for loading the checkpoint

    parameter
    ---------
    filepath: is the path to the file checkpoint to load in

    returns
    --------
    model: the model built to use for predicting
    optimizer: the optimizer for optimizing the network
    criterion: the loss function for checking the loss of the model
    epochs: the epochs used for training the network
    """
    checkpoint = torch.load(filepath)
    arch_model, model, optimizer, criterion = build_model(checkpoint["arch_model"],
                                                checkpoint["hidden_layers"],
                                                checkpoint["output_size"],
                                                checkpoint["dropout"],
                                                checkpoint["learning_rate"])

    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


    return model, optimizer, criterion, checkpoint["epochs"]



def predict(image_path, model, divice_mode, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implementing the code to predict the class from an image file
    if divice_mode:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        image = process_image(image_path)

        image = image.unsqueeze(0)
        model.eval()
        model.to(device)
        image = image.to(device)
        lgps = model.forward(image)
        ps = torch.exp(lgps)
        top_probs, top_classes = ps.topk(5, dim=1)
        arr_classes = top_classes.cpu().numpy().reshape(-1)

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [ k for k in arr_classes if str(k) in idx_to_class.values()]


    return top_probs, classes



# Display an image along with the top 5 classes
def display_image_predicted(image_path, model, actual_names, divice_mode, topk=5):

    top_probs, top_classes = predict(image_path, model, divice_mode, topk)
    image =   process_image(image_path)

    imshow(image, title=actual_names[str(top_classes[0])])

    top_probs = top_probs.cpu().numpy().reshape(-1)
    top_names = [ actual_names[str(k)] for k in top_classes ]

    fig, ax = plt.subplots()

    ax.barh(top_names, top_probs , align="center")

    plt.show()



if __name__ == "__main__":
    args = predict_arguments()

    model, optimizer, criterion, epochs = load_checkpoint(args.checkpoint)

    if args.category_names is None:
        raise Exception("You didn't provide the file path which contains the file which mapps labels to the names of the flowers")


    with open(args.category_names) as f_names:
        category_names = json.load(f_names)

    if category_names is None or len(category_names) < 1:
        raise Exception("You have provided wrong file for mapping labels to the names of the flowers")

    with active_session():
        display_image_predicted(args.input, model, category_names, args.gpu, args.top_k)
