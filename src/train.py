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

def main():
    args = parse_arguments()

    args.hidden_sizes = args.hidden_size

    # Convert hidden_sizes argument into a list of integers regardless of how it was passed
    #if isinstance(args.hidden_sizes, str):
    #    args.hidden_sizes = list(map(int, args.hidden_sizes.split()))
    #elif isinstance(args.hidden_sizes, list):
    #    args.hidden_sizes = [int(x) for x in args.hidden_sizes]
    #else:
    #    args.hidden_sizes = [int(args.hidden_sizes)]

    # if args.wandb_project:
    #     wandb.init(project=args.wandb_project, config=vars(args))

    # If running inside a W&B sweep, override CLI args with sweep config
    if args.wandb_project and wandb.run and wandb.run.sweep_id:
        sweep_cfg = wandb.config

        args.learning_rate = sweep_cfg.learning_rate
        args.batch_size = sweep_cfg.batch_size
        args.optimizer = sweep_cfg.optimizer
        args.activation = sweep_cfg.activation
        args.weight_init = sweep_cfg.weight_init
        args.loss = sweep_cfg.loss
        args.hidden_sizes = args.hidden_size
        args.num_layers = sweep_cfg.num_layers
        args.dataset = sweep_cfg.dataset
        args.epochs = sweep_cfg.epochs
        args.model_path = sweep_cfg.model_path
        args.weight_decay = sweep_cfg.weight_decay
        args.wandb_project = sweep_cfg.wandb_project

    if args.num_layers != len(args.hidden_sizes):
        raise ValueError("num_layers must match length of hidden_sizes list")

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
    np.random.seed(40)
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

            # update
            model.update_weights()

        epoch_loss /= num_batches

        # evaluate on train set
        train_accuracy, train_f1 = model.evaluate(X_train, y_train)

        # evaluate on test set
        accuracy, f1 = model.evaluate(X_test, y_test)
        val_logits = model.forward(X_test)
        val_loss = model.loss.forward(y_test, val_logits)

        # convert to scalar if it’s a numpy array
        if hasattr(val_loss, "item"):
            val_loss = val_loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Val Acc: {accuracy:.4f}, Val F1: {f1:.4f}")

        # W&B epoch logging
        # if args.wandb_project:
        #     wandb.log({
        #         "epoch": epoch + 1,
        #         "train_loss": epoch_loss,
        #         "train_accuracy": train_accuracy,
        #         "train_f1": train_f1,
        #         "val_loss": val_loss,
        #         "test_accuracy": accuracy,
        #         "test_f1": f1
        #     })

        # saving best model based on F1 score compared to previous best F1 score, if improved, save model weights and config with new best F1 score
        if f1 > best_f1:
            best_f1 = f1

            best_weights = model.get_weights()
            np.save(model_path, best_weights)

            config = vars(args).copy()
            config["best_f1"] = best_f1

            with open(config_path, "w") as f:
                json.dump(config, f)

            print(f"New best model saved with F1 Score: {best_f1:.4f}")

            # if args.wandb_project:
            #     wandb.save(model_path)
            #     wandb.save(config_path)

    # if args.wandb_project:
    #     wandb.finish()

    print("Training complete!")

if __name__ == '__main__':
    main()
