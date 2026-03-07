# DA6401 Assignment 1 — Neural Network from Scratch

This repository contains an implementation of a **Multi-Layer Perceptron (MLP)** built completely from scratch using **NumPy**.

---

## Features

The implementation includes:

- Forward and backward propagation
- Multiple activation functions  
  - ReLU  
  - Sigmoid  
  - Tanh  

- Multiple optimizers  
  - SGD  
  - Momentum  
  - NAG  
  - RMSProp  
  - Adam  
  - Nadam  

- Weight initialization strategies  
  - Zeros  
  - Xavier  

- Loss functions  
  - Cross Entropy  
  - Mean Squared Error  

- Gradient monitoring and neuron analysis

- Experiment tracking using **Weights & Biases (W&B)**

---

## Experiments Conducted

The following analyses were performed:

1. Optimizer comparison
2. Vanishing gradient analysis
3. Dead neuron investigation
4. Loss function comparison
5. Hyperparameter search
6. Global performance analysis
7. Error analysis using confusion matrices
8. Weight initialization symmetry analysis
9. Transfer learning experiment on Fashion-MNIST

---

## Repository Structure

```
da6401_assignment_1-1
│
├── models/                # Saved trained models
│   └── best_model.npy
│
├── notebooks/             # W&B visualization notebook
│   └── wandb_demo.ipynb
│
├── src/
│   ├── ann/               # Neural network implementation
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   │
│   ├── utils/
│   │   └── data_loader.py
│   │
│   ├── train.py           # Training script
│   └── inference.py       # Model inference
│
├── requirements.txt
├── sweep.yaml
└── README.md
```

---

## W&B Experiment Dashboard

All experiments and visualizations are available here:

W&B Report:  https://wandb.ai/me22b190-indian-institute-of-technology-madras/da6401_assignment_1-1-src/reports/DA6401-Assignment-1-Report--VmlldzoxNjEyNDg5Mw?accessToken=gbc6eqkuso1iotd882fib02qp64s0zx96mcsycf3i8hlpryt0701zy8ma84lyqcc

## GitHub Repository

https://github.com/Sayantika592/da6401_assignment_1


---

## How to Run

Example training command:

bash: 
python src/train.py \
-w_p da6401 \
-m src \
-d mnist \
-e 15 \
-b 128 \
-l cross_entropy \
-o nadam \
-lr 0.001 \
-nhl 2 \
-sz "128 64" \
-a tanh \
-w_i xavier


---

# Datasets

-MNIST
-Fashion-MNIST

Sayantika Chakraborty
ME22B190
