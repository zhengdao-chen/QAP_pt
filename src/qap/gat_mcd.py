from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pyGAT_code.utils import load_data, accuracy
from pyGAT_code.models import GAT, SpGAT

from data_generator_mod import Generator
from Logger import Logger

from losses import compute_loss_multiclass, compute_accuracy_multiclass

parser = argparse.ArgumentParser()

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.8)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.03)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='ErdosRenyi')
parser.add_argument('--iterations', nargs='?', const=1, type=int,
                    default=int(60000))
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)
parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
parser.set_defaults(eval_vs_train=False)
parser.add_argument('--subtract_mean', dest='bn', action='store_false')
parser.set_defaults(bn=True)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers_test', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_iterations_avg', nargs='?', const=1, type=int,
                    default=5)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=2)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=50)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=50)

# Training settings
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if (args.eval_vs_train):
    print ('eval')
else:
    print ('train')
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=1,
                nhid=args.num_features,
                nclass=args.n_classes,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
else:
    model = GAT(nfeat=1,
                nhid=args.num_features,
                nclass=args.n_classes,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha,
                bn_or_not=args.bn)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(model, epoch): #, inputs_batch, labels_batch, W_batch):
    loss_batch = 0
    acc_batch = 0
    start = time.time()

    for i in range(args.batch_size):

        model.train()
        optimizer.zero_grad()
        # print ('features', np.max(features.data.numpy()))
        # print (np.where(features>0))
        # print ('features', features.shape)
        # print ('adj', adj.shape)
        # output = model(features, adj)

        inputs_batch, labels_batch, W_batch = gen.sample_batch_bcd(args.batch_size, cuda=torch.cuda.is_available())

        features = inputs_batch[1][i, :]
        adj = Variable(torch.from_numpy(W_batch[i, :, :]), volatile=False)
        labels = labels_batch[i, :].type(dtype_l)
        if (torch.cuda.is_available()):
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.cpu().numpy() == -1):
                labels = (labels + 1)/2
        else:
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.numpy() == -1):
                labels = (labels + 1)/2

        if (torch.cuda.is_available()):
            adj = adj.cuda()

        # print ('labels', labels)
        output = model(features, adj)
        # output = model(inputs[1], W)
        # print ('output', output[:3, :])
        # print ('labels', labels.shape)
        # loss_train = F.nll_loss(output, labels)
        loss_train = compute_loss_multiclass(output.unsqueeze(0), labels.unsqueeze(0), args.n_classes)
        # acc_train = accuracy(output, labels)
        acc_train = compute_accuracy_multiclass(output.unsqueeze(0), labels.unsqueeze(0), args.n_classes)
        loss_train.backward()
        optimizer.step()

        loss_batch += loss_train
        acc_batch += acc_train

    elapsed = time.time() - start

    if (epoch % args.print_freq == 0) and (epoch > 0):
        info = ['epoch', 'loss', 'accuracy', 'edge_density',
                'noise', 'model', 'elapsed']
        out = [epoch, loss_train.item(), acc_train / args.batch_size, args.edge_density,
               args.noise, 'GAT', elapsed]
        print(template1.format(*info))
        print(template2.format(*out))
    # print (info)
    # print (out)

        # if not args.fastmode:
        #     # Evaluate validation set performance separately,
        #     # deactivates dropout during validation run.
        #     model.eval()
        #     output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.data[0]),
        #       'acc_train: {:.4f}'.format(acc_train.data[0]),
        #       'loss_val: {:.4f}'.format(loss_val.data[0]),
        #       'acc_val: {:.4f}'.format(acc_val.data[0]),
        #       'time: {:.4f}s'.format(time.time() - t))

    return loss_train


def compute_test(model):
    loss_total = 0
    acc_total = 0

    inputs_batch, labels_batch, W_batch = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    if (args.eval_vs_train):
        model.eval()
    for i in range(args.num_examples_test):
        start = time.time()
        features = inputs_batch[1][i, :]
        adj = Variable(torch.from_numpy(W_batch[i, :, :]), volatile=False)
        labels = labels_batch[i, :].type(dtype_l)
        if (torch.cuda.is_available()):
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.cpu().numpy() == -1):
                labels = (labels + 1)/2
        else:
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.numpy() == -1):
                labels = (labels + 1)/2

        if (torch.cuda.is_available()):
            adj = adj.cuda()
        output = model(features, adj)

        loss_test = compute_loss_multiclass(output.unsqueeze(0), labels.unsqueeze(0), args.n_classes)
        acc_test = compute_accuracy_multiclass(output.unsqueeze(0), labels.unsqueeze(0), args.n_classes)

        elapsed = time.time() - start

        info = ['epoch', 'loss', 'accuracy', 'edge_density',
        'noise', 'model', 'elapsed']
        out = [-1, loss_test, acc_test, args.edge_density,
               args.noise, 'GAT', elapsed]
        print(template1.format(*info))
        print(template2.format(*out))

        loss_total += loss_test.item() / args.num_examples_test
        acc_total += acc_test / args.num_examples_test

    elapsed = time.time() - start

    info = ['epoch', 'average loss', 'average accuracy', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [-1, loss_total, acc_total, args.edge_density,
           args.noise, 'GAT', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))


# load or generate data
gen = Generator(args.path_dataset)
# generator setup
gen.num_examples_train = args.num_examples_train
gen.num_examples_test = args.num_examples_test
gen.J = args.J
gen.N_train = args.N_train
gen.N_test = args.N_test
gen.edge_density = args.edge_density
gen.p_SBM = args.p_SBM
gen.q_SBM = args.q_SBM
gen.random_noise = args.random_noise
gen.noise = args.noise
gen.noise_model = args.noise_model
gen.generative_model = args.generative_model
gen.n_classes = args.n_classes
# load dataset
# print(gen.random_noise)
gen.load_dataset()

# inputs_batch, labels_batch, W_batch = gen.sample_batch_bcd(args.batch_size, cuda=torch.cuda.is_available())

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

if (torch.cuda.is_available()):
    model = model.cuda()

for epoch in range(args.epochs):
    loss_values.append(train(model, epoch))#, inputs_batch, labels_batch, W_batch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)



files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test(model)
if args.bn:
    print ('bn2d')
else:
    print ('sub mean')
# print ('subtract mean')
