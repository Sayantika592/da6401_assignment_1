"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np

from ann.neural_layer import Linear
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import f1_score

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-l", "--loss", required=True)
    parser.add_argument("-o", "--optimizer", required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_sizes", type=int, nargs="+", required=True)
    parser.add_argument("-a", "--activation", required=True)
    parser.add_argument("-w_i", "--weight_init", required=True)
    
    return parser.parse_args()

def compute_f1_score(model, X, y):
    """
    Compute F1 score for model predictions.
    
    Args:
        model: Trained neural network model
        X: Input data
        y: True labels (one-hot encoded)
    Returns:
        F1 score for the model's predictions
    """
    y_pred = model.forward(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)
    return f1_score(y_true_labels, y_pred_labels, average='weighted')

def main():
    """
    Main training function.
    """
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    args.input_size = X_train.shape[1]
    args.output_size = y_train.shape[1]

    model = NeuralNetwork(args)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train(X_train, y_train, 1, args.batch_size)  # Train for one epoch
        accuracy = model.evaluate(X_test, y_test)
        print(f"Epoch {epoch+1}/{args.epochs}, Accuracy: {accuracy:.4f}")
        f1 = compute_f1_score(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{args.epochs}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

            weights = [] # save model weights in .npy file but only for those with best f1 score
            for layer in model.layers:
                if isinstance(layer, Linear):
                    weights.append({"W": layer.W, "b": layer.b})

            np.save("best_model.npy", weights)

            config = {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_sizes": args.hidden_sizes,
                "activation": args.activation,
                "weight_init": args.weight_init
            }

            with open("best_config.json", "w") as f:
                json.dump(config, f)

            print(f"New best model saved with F1 Score: {best_f1:.4f}")

    print("Training complete!")


if __name__ == '__main__':
    main()
