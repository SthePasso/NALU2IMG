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

from nalu import NAC, NALU, NALU2M, NALUIG, NALU2MIG

functions = {
    'a + b':   lambda a, b: a + b,
    'a - b':   lambda a, b: a - b,
    'a x b':   lambda a, b: a * b,
    'a / b':   lambda a, b: a / b,
    'a ^ 2':   lambda a, b: a**2,
    'sqrt(a)': lambda a, b: np.sqrt(a),
}

activations = {
    'ReLU':    nn.Activation('relu'),
    'Tanh':    nn.Activation('tanh'),
    'Sigmoid': nn.Activation('sigmoid'),
}

# experiment setting
IN_DIM = 100
HIDDEN_DIM = 2
N, M = 0, 10
P, Q = 70, 100


def load_data(op, scale, train_size, test_size):
    np.random.seed(0)

    filename = "data/" + (("%s_%.2f_%d_%d.cache" % (op, scale, train_size, test_size)).replace('/', '_div_').replace(' ',''))

    if os.path.exists(filename):
        logging.info("Load cache from %s" % filename)
        x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e, random_rmse_i, random_rmse_e =\
                pickle.load(open(filename, "rb"))
    else:
        func = functions[op]

        x_train = np.random.uniform(0, scale, size=(train_size, IN_DIM))
        y_train = func(np.sum(x_train[:, N:M], axis=1), np.sum(x_train[:, P:Q], axis=1))

        a = np.sum(x_train[:, N:M], axis=1)
        b = np.sum(x_train[:, P:Q], axis=1)

        a_max = np.max(np.sum(x_train[:, N:M], axis=1))
        b_max = np.max(np.sum(x_train[:, P:Q], axis=1))
        y_max = np.max(y_train)

        x_test_i, y_test_i = [], []
        x_test_e, y_test_e = [], []

        i_len = 0
        e_len = 0

        while i_len < test_size:
            x_test = np.random.uniform(0, scale, size=(test_size, IN_DIM))
            a = np.sum(x_test[:, N:M], axis=1)
            b = np.sum(x_test[:, P:Q], axis=1)
            y_test = func(a, b)

            in_range = (a < a_max) & (b < b_max) & (y_test < y_max)
            out_range = ~in_range
            x_test_i.append(x_test[in_range])
            y_test_i.append(y_test[in_range])

            i_len += np.sum(in_range)

        while e_len < test_size:
            x_test = np.random.uniform(0, scale * 4, size=(test_size, IN_DIM))
            a = np.sum(x_test[:, N:M], axis=1)
            b = np.sum(x_test[:, P:Q], axis=1)
            y_test = func(a, b)

            in_range = (a < a_max) & (b < b_max) & (y_test < y_max)
            out_range = ~in_range
            x_test_e.append(x_test[out_range])
            y_test_e.append(y_test[out_range])

            e_len += np.sum(out_range)

        x_test_i = np.concatenate(x_test_i)[:test_size].astype(np.float32)
        y_test_i = np.concatenate(y_test_i)[:test_size].astype(np.float32)
        x_test_e = np.concatenate(x_test_e)[:test_size].astype(np.float32)
        y_test_e = np.concatenate(y_test_e)[:test_size].astype(np.float32)

        random_rmse_i, random_rmse_e = get_random_baseline(op, x_test_i, y_test_i, x_test_e, y_test_e)

        pickle.dump((x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e, random_rmse_i, random_rmse_e),
                    open(filename, "wb"))

    logging.info("random I: %.2f\trandom E: %.2f" % (random_rmse_i, random_rmse_e))

    return [np.array(x, dtype=np.float32) for x in [x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e]] +\
            [random_rmse_i, random_rmse_e]


def evaluate_rmse(net, data_iter, ctx):
    rmse = mx.metric.RMSE()
    for data in data_iter:
        data = [x.as_in_context(ctx) for x in data]
        output = net(*data[:-1])
        rmse.update(preds=output, labels=data[-1])
    return rmse.get()[1]


def get_random_baseline(op, x_test_i, y_test_i, x_test_e, y_test_e, n_repeat=20):
    batch_size = 1024
    test_i_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test_i, y_test_i),
                                        batch_size, shuffle=False)
    test_e_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test_e, y_test_e),
                                        batch_size, shuffle=False)

    total_rmse_i, total_rmse_e = 0, 0
    for i in range(n_repeat):
        net = nn.Sequential()
        net.add(nn.Dense(HIDDEN_DIM))
        net.add(nn.Dense(1))
        net.collect_params().initialize(mx.init.Uniform(1), ctx=ctx)
        rmse_i, rmse_e = evaluate_rmse(net, test_i_data, ctx), evaluate_rmse(net, test_e_data, ctx)
        total_rmse_i += rmse_i
        total_rmse_e += rmse_e

    return total_rmse_i / n_repeat, total_rmse_e / n_repeat


def calculate_errors(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mean_true = np.mean(y_true)
    mad = np.mean(np.abs(y_true-mean_true)) 
    return mae, mse, rmse, mad

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    return np.mean(np.abs(y_true - y_pred) < threshold)

def train_static(op, net_type, net, x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e,
                 random_i, random_e, params):
    batch_size = params['batch_size']

    filename = "models/" + ((op + "." + net_type + ".params").replace('/', '_div_').replace(' ',''))

    if args.cont:
        logging.info("Continue training from %s" % filename)
        if os.path.exists(filename):
            net.load_parameters(filename)

    # data iterator
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train),
                                       batch_size, shuffle=True)
    test_i_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test_i, y_test_i),
                                        batch_size * 10, shuffle=False)
    test_e_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test_e, y_test_e),
                                        batch_size * 10, shuffle=False)

    trainer = gluon.Trainer(net.collect_params(), optimizer=params['optimizer'], optimizer_params=params['optimizer_params'])
    l2loss = gluon.loss.L2Loss()
    print_every = 5
    moving_loss = None

    best_loss = 1e9
    best_round = 0

    # Initial evaluations
    start_time_i = time.time()
    i_rmse = evaluate_rmse(net, test_i_data, ctx) 
    end_time_i = time.time()
    training_time_i = end_time_i - start_time_i

    start_time_e = time.time()
    e_rmse = evaluate_rmse(net, test_e_data, ctx) 
    end_time_e = time.time()
    training_time_e = end_time_e - start_time_e

    logging.info("Epoch: %2d\tBatch: %d\tI MSE: %.6f\tE MSE: %.6f\tI score: %.2f\tE score: %.2f" %
                 (0, 0, i_rmse, e_rmse, i_rmse / random_i * 100, e_rmse / random_e * 100))

    start_time = time.time()  # Start measuring training time
    for epoch in range(params['n_epoch']):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx).reshape((-1, 1))
            with autograd.record():
                output = net(data)
                loss = l2loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            num_batch = i + 1

            if moving_loss is None:
                moving_loss = np.mean(loss.asnumpy())
            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(loss.asnumpy())

        if epoch % print_every == 0:
            start_time_i = time.time()
            i_rmse = evaluate_rmse(net, test_i_data, ctx)
            end_time_i = time.time()
            training_time_i += (end_time_i - start_time_i)

            start_time_e = time.time()
            e_rmse = evaluate_rmse(net, test_e_data, ctx)
            end_time_e = time.time()
            training_time_e += (end_time_e - start_time_e)

            logging.info("Epoch: %2d\tBatch: %d\tI MSE: %.6f\tE MSE: %.6f\tI score: %.2f\tE score: %.2f\tMoving loss: %.6f" %
                         (epoch + 1, num_batch, i_rmse, e_rmse, i_rmse / random_i * 100, e_rmse / random_e * 100, moving_loss))
            if i_rmse + e_rmse < best_loss:
                best_loss = i_rmse + e_rmse
                best_round = epoch + 1
                net.save_parameters(filename)

    end_time = time.time()  # End measuring training time
    training_time = end_time - start_time

    # Calculate accuracy
    # Compute final predictions and errors
    y_test_i_pred = net(nd.array(x_test_i, ctx=ctx)).asnumpy()
    y_test_e_pred = net(nd.array(x_test_e, ctx=ctx)).asnumpy()
    
    accuracy_i = calculate_accuracy(y_test_i, y_test_i_pred)
    accuracy_e = calculate_accuracy(y_test_e, y_test_e_pred)

    mae_i, mse_i, rmse_i, mad_i = calculate_errors(y_test_i, y_test_i_pred)
    mae_e, mse_e, rmse_e, mad_e = calculate_errors(y_test_e, y_test_e_pred)
    
    return {
        'net': net,
        'i_mae': mae_i,
        'i_mse': mse_i,
        'i_rmse': rmse_i,
        'i_mad': mad_i,
        'e_mae': mae_e,
        'e_mse': mse_e,
        'e_rmse': rmse_e,
        'e_mad': mad_e,
        'accuracy_i': accuracy_i,
        'accuracy_e': accuracy_e,
        'training_time': training_time,
        'training_time_i': training_time_i,
        'training_time_e': training_time_e
    }



def results_to_markdown(eval_results, metrics):
    # Create header row
    res_str = "|       |"
    for net_type in networks:
        res_str += f"  {net_type}  |" * len(metrics)
    res_str += "\n| --- " + " | --- " * len(networks) * len(metrics) + "|\n|       |"

    # Create sub-header row
    for net_type in networks:
        for metric in metrics:
            res_str += f" {metric} |"
    res_str += "\n"

    # Create rows for each operator
    for op in operators:
        res_str += f"| {op} |"
        for net_type in networks:
            results = eval_results.get((op, net_type), (np.nan, np.nan, np.nan))
            res_str += " | ".join([f"{result:.2f}" for result in results]) + " |"
        res_str += "\n"

    return res_str
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--op", type=str)
    parser.add_argument("--log", type=str, default='default')
    parser.add_argument("--simple", action='store_true')
    parser.add_argument("--cont", action='store_true')
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--init-scale", type=float, default=1e-2)
    parser.add_argument("--n-epoch", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ctx = mx.cpu()

    if args.network is None:
        networks = ['ReLU', 'Sigmoid', 'NAC', 'NALU', 'NALU2M', 'NALUIG', 'NALU2MIG'] #['ReLU', 'Sigmoid', 'NAC', 'NALU', 'NALUIG']
    else:
        networks = [args.network]

    if args.op is None:
        operators = ['a + b', 'a - b', 'a x b', 'a / b', 'a ^ 2', 'sqrt(a)']
    else:
        operators = [args.op]

    scale = 0.5
    train_size, test_size = 40000, 2000
    params = {
        'n_epoch': args.n_epoch,
        'early_stopping': 50,
        'batch_size': 64,
        'optimizer': 'adam',
        'optimizer_params': {'learning_rate': args.learning_rate},
    }

    print(params)

    logging.info(args)
    logging.info(params)

    eval_results_i = {}
    eval_results_e = {}

    # train and evaluate
    for op in operators:
        x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e, random_rmse_i, random_rmse_e = \
            load_data(op, scale, train_size, test_size)
        for net_type in networks:
            mx.random.seed(0)

            net = nn.Sequential()
            with net.name_scope():
                if net_type in ['ReLU', 'Sigmoid']:
                    net.add(nn.Dense(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(activations[net_type])
                    net.add(nn.Dense(units=1))
                    net.collect_params().initialize(mx.init.Uniform(1), ctx=ctx)
                elif net_type == 'NAC':
                    net.add(NAC(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(NAC(in_units=HIDDEN_DIM, units=1))
                    net.collect_params().initialize(mx.init.Uniform(args.init_scale), ctx=ctx)
                elif net_type == 'NALU':
                    net.add(NALU(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(NALU(in_units=HIDDEN_DIM, units=1))
                    net.collect_params().initialize(mx.init.Uniform(args.init_scale), ctx=ctx)
                elif net_type == 'NALU2M':
                    net.add(NALU2M(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(NALU2M(in_units=HIDDEN_DIM, units=1))
                    net.collect_params().initialize(mx.init.Uniform(args.init_scale), ctx=ctx)
                elif net_type == 'NALUIG':
                    net.add(NALUIG(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(NALUIG(in_units=HIDDEN_DIM, units=1))
                    net.collect_params().initialize(mx.init.Uniform(args.init_scale), ctx=ctx)
                elif net_type == 'NALU2MIG':
                    net.add(NALU2MIG(in_units=IN_DIM, units=HIDDEN_DIM))
                    net.add(NALU2MIG(in_units=HIDDEN_DIM, units=1))
                    net.collect_params().initialize(mx.init.Uniform(args.init_scale), ctx=ctx)
                else:
                    raise ValueError("Invalid Network: " + net_type)

            logging.info("Learn %s with %s" % (op, net_type))
            errors = train_static(op, net_type, net, x_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e,
                                  random_rmse_i, random_rmse_e, params)
            
            eval_results_i[(op, net_type)] = (errors['i_mae'], errors['i_mse'], errors['i_rmse'],errors['accuracy_i'], errors['training_time_i'])
            eval_results_e[(op, net_type)] = (errors['e_mae'], errors['e_mse'], errors['e_rmse'],errors['accuracy_e'], errors['training_time_e'])

    print("### Interpolation")
    print(results_to_markdown(eval_results_i, ['MAE', 'MSE', 'RMSE','MAD','ACCURACY','TRAINING TIME']))

    print("### Extrapolation")
    print(results_to_markdown(eval_results_e, ['MAE', 'MSE', 'RMSE','MAD','ACCURACY','TRAINING TIME']))
