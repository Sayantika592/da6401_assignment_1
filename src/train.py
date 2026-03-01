"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np

from ann.neural_layer import Linear
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import f1_score
import wandb

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
    
    parser.add_argument("-w_p", "--wandb_project", default=None)
    parser.add_argument("-m", "--model_path", required=True)
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
    args = parse_arguments()

    np.random.seed(40)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # path handling for model saving
    save_dir = args.model_path
    if save_dir.endswith(".npy") or save_dir.endswith(".json"):
        save_dir = os.path.dirname(save_dir) or "."
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "best_config.json")
    model_path = os.path.join(save_dir, "best_model.npy")

    best_f1 = -1.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                best_f1 = json.load(f).get("best_f1", -1.0)
                print(f"Loaded previous best F1: {best_f1:.4f}")
        except:
            print("Could not read previous config. Starting fresh.")

    # Data loading
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    args.input_size = X_train.shape[1]
    args.output_size = y_train.shape[1]

    model = NeuralNetwork(args)

    # training loop
    N = X_train.shape[0]

    for epoch in range(args.epochs):

        perm = np.random.permutation(N)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, args.batch_size):

            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            # forward
            logits = model.forward(X_batch)

            # loss
            loss = model.loss.forward(y_batch, logits)
            epoch_loss += loss
            num_batches += 1

            # backward
            model.backward(y_batch, logits)

            # gradient norm (first layer)
            first_linear = None
            for layer in model.layers:
                if isinstance(layer, Linear):
                    first_linear = layer
                    break

            grad_norm = np.linalg.norm(first_linear.grad_W)

            # dead neuron ratio (ReLU) or saturation ratio (Tanh) for activation analysis

            dead_ratio = None
            tanh_saturation = None

            if hasattr(model, "cached_activations") and len(model.cached_activations) > 0:
                act = model.cached_activations[0]  # first hidden layer

                if args.activation == "relu":
                    dead_ratio = np.mean(act == 0)

                elif args.activation == "tanh":
                    tanh_saturation = np.mean(np.abs(act) > 0.95)

            # dead neuron ratio (ReLU only as asked in the question)
            dead_ratio = None
            if args.activation == "relu" and hasattr(model, "cached_activations"):
                act = model.cached_activations[0]
                dead_ratio = np.mean(act == 0)

            # W&B logging for batch metrics
            if args.wandb_project:
                log_dict = {
                    "batch_loss": loss,
                    "grad_norm_layer1": grad_norm
                }
                if dead_ratio is not None:
                    log_dict["dead_neuron_ratio"] = dead_ratio

                if tanh_saturation is not None:
                    log_dict["tanh_saturation_ratio"] = tanh_saturation

                wandb.log(log_dict)
                
            # update
            model.update_weights()

        epoch_loss /= num_batches

        # evaluate on test set
        accuracy = model.evaluate(X_test, y_test)
        f1 = compute_f1_score(model, X_test, y_test)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

        # W&B epoch logging
        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "test_accuracy": accuracy,
                "test_f1": f1
            })

        # saving best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1

            best_weights = model.get_weights()
            np.save(model_path, best_weights)

            config = vars(args).copy()
            config["best_f1"] = best_f1

            with open(config_path, "w") as f:
                json.dump(config, f)

            print(f"New best model saved with F1 Score: {best_f1:.4f}")

            if args.wandb_project:
                wandb.save(model_path)
                wandb.save(config_path)

    print("Training complete!")

if __name__ == '__main__':
    main()
