"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_layer import Linear
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils.data_loader import load_data

from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-w_p", "--wandb_project", default="da6401_assignment_1-1-src")
    parser.add_argument("-m", "--model_path", default="src")

    parser.add_argument("-d", "--dataset", default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=15)
    parser.add_argument("-b", "--batch_size", type=int, default=128)

    parser.add_argument("-l", "--loss", default="cross_entropy")
    parser.add_argument("-o", "--optimizer", default="nadam")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 64])
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)

    parser.add_argument("-a", "--activation", default="tanh")
    parser.add_argument("-w_i", "--weight_init", default="xavier")
    
    return parser.parse_args()


def load_model(model_path, args):
    """
    Load trained model from disk.
    """
    weights = np.load(model_path, allow_pickle=True).item() # weights loaded from .npy file created during training

    model = NeuralNetwork(args) # create model architecture based on args (hidden layers, activation, etc.)

    #idx=0
    #for layer in model.layers:   # weights got from the model saved in .npy file are assigned to the corresponding layers in the model architecture
        #if isinstance(layer,Linear):
            #layer.W = weights[idx]["W"]
            #layer.b = weights[idx]["b"]
            #idx += 1
    model.set_weights(weights) # weights got from the model saved in .npy file are assigned to the corresponding layers in the model architecture

    return model
        

def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test) # get model predictions (logits)

    y_pred_labels = np.argmax(logits, axis=1) # convert logits to predicted class labels
    y_true_labels = np.argmax(y_test, axis=1) # convert one-hot encoded true labels to class labels

    # Metrics calculation for evaluation

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')

    loss = model.loss.forward(y_test, logits) # calculate loss using the model's loss function

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Load architecture from best_config.json if it exists
    import os, json
    config_path = args.model_path.replace("best_model.npy", "best_config.json")
    if not os.path.exists(config_path):
        save_dir = os.path.dirname(args.model_path) or "."
        config_path = os.path.join(save_dir, "best_config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            args.hidden_size = config.get("hidden_size", args.hidden_size)
            args.num_layers = config.get("num_layers", args.num_layers)
            args.activation = config.get("activation", args.activation)
            args.loss = config.get("loss", args.loss)

    args.hidden_sizes = args.hidden_size

    _, _, X_test, y_test = load_data(args.dataset)

    args.input_size = X_test.shape[1] # set input size based on test data features
    args.output_size = y_test.shape[1] # set output size based on test data labels
    
    model = load_model(args.model_path, args) # load trained model

    results = evaluate_model(model, X_test, y_test) # evaluate model on test data

    print("Evaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")


if __name__ == '__main__':
    main()
