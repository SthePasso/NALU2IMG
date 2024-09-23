import argparse
import logging
import pickle
import os
import math
import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from mxnet import gluon
from mxnet.gluon.data import DataLoader, Dataset
from mxnet.gluon.data.vision import transforms
import numpy as np

from nalu_try import NAC, NALU, NALU2M, NALUIG, NALU2MIG

# Define activations
activations = {
    'ReLU':    nn.Activation('relu'),
    'Sigmoid': nn.Activation('sigmoid'),
}

# experiment setting
HIDDEN_DIM = 64
OUTPUT_DIM = 10  # For 10 classes in MNIST

# Load the MNIST dataset with train, validation, and test splits
def load_mnist_data(batch_size, val_split=0.1):
    # Define the transformation for MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)
    ])

    # Load and transform MNIST data
    mnist_train = gluon.data.vision.datasets.MNIST(train=True).transform_first(transform)
    mnist_test = gluon.data.vision.datasets.MNIST(train=False).transform_first(transform)

    # Calculate the lengths of train and validation sets
    train_len = int(len(mnist_train) * (1 - val_split))
    val_len = len(mnist_train) - train_len

    # Shuffle the data indices
    indices = np.random.permutation(len(mnist_train))

    # Split the indices into training and validation
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # Create training and validation datasets using Subset
    train_data = gluon.data.SimpleDataset([mnist_train[i] for i in train_indices])
    val_data = gluon.data.SimpleDataset([mnist_train[i] for i in val_indices])

    # DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Evaluate model accuracy
def evaluate_accuracy(net, data_iter, ctx):
    acc = mx.metric.Accuracy()
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc.update(preds=output, labels=label)
    return acc.get()[1]

# Calculate errors (MAE, MSE, RMSE, MAD)
def calculate_errors(y_true, y_pred):
    # Convert predicted probabilities to class labels
    y_pred_class = np.argmax(y_pred, axis=1)

    mae = np.mean(np.abs(y_true - y_pred_class))
    mse = np.mean((y_true - y_pred_class) ** 2)
    rmse = np.sqrt(mse)
    mad = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae, mse, rmse, mad


# Get predictions for calculating errors
def get_predictions(net, data_iter, ctx):
    all_labels = []
    all_preds = []
    for data, label in data_iter:
        data = data.as_in_context(ctx)
        output = net(data)
        all_preds.append(output.asnumpy())  # Predicted probabilities
        all_labels.append(label.asnumpy())  # Actual class labels
    return np.concatenate(all_labels), np.concatenate(all_preds)


# Training the network with time tracking
def train_static(net_type, net, train_data, val_data, test_data, ctx, params):
    batch_size = params['batch_size']
    trainer = gluon.Trainer(net.collect_params(), optimizer=params['optimizer'], optimizer_params=params['optimizer_params'])
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    print_every = 5
    start_time = time.time()  # Start time for training

    for epoch in range(params['n_epoch']):
        for data, label in train_data:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

        if epoch % print_every == 0:
            val_acc = evaluate_accuracy(net, val_data, ctx)
            logging.info(f"Epoch {epoch}: Validation Accuracy = {val_acc * 100:.2f}%")

    end_time = time.time()  # End time for training
    training_time = end_time - start_time  # Calculate training time
    return training_time

# Build the network
def build_network(net_type):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2))
        net.add(nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2))
        net.add(nn.Flatten())  # Flatten the 2D data to 1D

        # Now add NALU layers (which expect 1D data after flattening)
        net.add(nn.Dense(HIDDEN_DIM, activation='relu'))
        
        if net_type in activations:
            net.add(activations[net_type])
        elif net_type == 'NAC':
            net.add(NAC(HIDDEN_DIM, OUTPUT_DIM))
        elif net_type == 'NALU':
            net.add(NALU(HIDDEN_DIM, OUTPUT_DIM))
        elif net_type == 'NALU2M':
            net.add(NALU2M(HIDDEN_DIM, OUTPUT_DIM))
        elif net_type == 'NALUIG':
            net.add(NALUIG(HIDDEN_DIM, OUTPUT_DIM))
        elif net_type == 'NALU2MIG':
            net.add(NALU2MIG(HIDDEN_DIM, OUTPUT_DIM))
        else:
            raise ValueError("Invalid Network Type")

        net.add(nn.Dense(OUTPUT_DIM))
    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="ReLU")
    parser.add_argument("--n-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ctx = mx.cpu()

    networks = ['ReLU', 'Sigmoid', 'NAC', 'NALU', 'NALU2M', 'NALUIG', 'NALU2MIG']

    params = {
        'n_epoch': args.n_epoch,
        'batch_size': args.batch_size,
        'optimizer': 'adam',
        'optimizer_params': {'learning_rate': args.learning_rate},
    }

    # Load MNIST data
    train_data, val_data, test_data = load_mnist_data(params['batch_size'])

    # Matrix to store results for each network
    results_matrix = []

    # Train and evaluate each network
    for net_type in networks:
        logging.info(f"Training network: {net_type}")
        net = build_network(net_type)
        if isinstance(net_type, (NALUIG, NALU2MIG)):  # For NALU layers expecting 1D data
            net.initialize(mx.init.Normal(), ctx=ctx)  # Use a normal initializer as an example for these layers
        else: 
            net.initialize(mx.init.Xavier(), ctx=ctx)

        # Train the network and capture training time
        training_time = train_static(net_type, net, train_data, val_data, test_data, ctx, params)

        # Final validation accuracy
        test_acc = evaluate_accuracy(net, test_data, ctx)
        logging.info(f"Final Test Accuracy for {net_type}: {test_acc * 100:.2f}%")

        # Get predictions and calculate metrics on validation set
        y_true, y_pred = get_predictions(net, test_data, ctx) # changed val_data to test_data
        mae, mse, rmse, mad = calculate_errors(y_true, y_pred)

        # Store the metrics and training time for this model
        results_matrix.append([net_type, mae, mse, rmse, mad, test_acc, training_time])

    # Print out the results matrix
    print("\nMetrics for each network:")
    print("Network | MAE | MSE | RMSE | MAD | Accuracy | Training Time (s)")
    for result in results_matrix:
        print(f"{result[0]} | {result[1]:.4f} | {result[2]:.4f} | {result[3]:.4f} | {result[4]:.4f} | {result[5] * 100:.2f}% | {result[6]:.2f}s")
