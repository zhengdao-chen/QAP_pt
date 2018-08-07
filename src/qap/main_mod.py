#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator_mod import Generator
from model import Siamese_GNN, GNN
from Logger import Logger
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
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
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers_test', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=50)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=50)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'

def compute_loss(pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return criterion(pred, labels)

def train(siamese_gnn, logger, gen):
    labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
              gen.N_train)).type(dtype_l))
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=1e-3)
    for it in range(args.iterations):
        start = time.time()
        input = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        pred = siamese_gnn(*input)
        loss = compute_loss(pred, labels)
        siamese_gnn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(siamese_gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy(pred, labels)
        elapsed = time.time() - start
        if it % logger.args['print_freq'] == 0:
            logger.plot_train_accuracy()
            logger.plot_train_loss()
            loss = loss.data.cpu().numpy()#[0]
            info = ['iteration', 'loss', 'accuracy', 'edge_density',
                    'noise', 'model', 'elapsed']
            out = [it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
                   args.noise, args.generative_model, elapsed]
            print(template1.format(*info))
            print(template2.format(*out))
            # test(siamese_gnn, logger, gen)
        if it % logger.args['save_freq'] == 0:
            logger.save_model(siamese_gnn)
            logger.save_results()
    print('Optimization finished.')

def test(siamese_gnn, logger, gen):
    labels = (Variable(torch.arange(0, gen.N_test).unsqueeze(0).expand(batch_size,
              gen.N_test)).type(dtype_l))
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=1e-3)
    for it in range(1):
        start = time.time()
        input = gen.sample_batch(batch_size, cuda=torch.cuda.is_available(), is_training=False)
        pred = siamese_gnn(*input)
        loss = compute_loss(pred, labels)
        siamese_gnn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(siamese_gnn.parameters(), args.clip_grad_norm)
        # optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy(pred, labels)
        elapsed = time.time() - start

        logger.plot_train_accuracy()
        logger.plot_train_loss()
        loss = loss.data.cpu().numpy()#[0]
        info = ['iteration', 'loss', 'accuracy', 'edge_density',
                'noise', 'model', 'elapsed']
        out = [it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
               args.noise, args.generative_model, elapsed]
        print(template1.format(*info))
        print(template2.format(*out))

    print('Optimization finished.')

def extend_gnn_append_random(siamese_gnn):
    gnn = siamese_gnn.gnn
    gnn_new = GNN(args.num_features, args.num_layers_test, args.J+2)
    gnn_new.cuda()
    gnn_new.layer0 = gnn.layer0
    for i in range(gnn.num_layers):
        gnn_new._modules['layer{}'.format(i+1)] = gnn._modules['layer{}'.format(i+1)]
    siamese_gnn_new = Siamese_GNN(args.num_features, args.num_layers_test, args.J+2, gnn=gnn_new)
    return siamese_gnn_new

def extend_gnn_append(siamese_gnn):
    gnn = siamese_gnn.gnn
    gnn_new = GNN(args.num_features, args.num_layers_test, args.J+2)
    gnn_new.cuda()
    gnn_new.layer0 = gnn.layer0
    for i in range(args.num_layers_test):
        gnn_new._modules['layer{}'.format(i+1)] = gnn._modules['layer{}'.format(1 + i % args.num_layers)]
    gnn_new.layerlast = gnn.layerlast
    siamese_gnn_new = Siamese_GNN(args.num_features, args.num_layers_test, args.J+2, gnn=gnn_new)
    return siamese_gnn_new

def extend_gnn_repeat(siamese_gnn):
    gnn = siamese_gnn.gnn
    gnn_new = GNN(args.num_features, args.num_layers_test, args.J+2)
    gnn_new.cuda()
    gnn_new.layer0 = gnn.layer0
    num_repeat = np.ceil(args.num_layers_test / args.num_layers)
    for i in range(args.num_layers_test):
        gnn_new._modules['layer{}'.format(i+1)] = gnn._modules['layer{}'.format(np.floor(i / num_repeat)+1)]
    gnn_new.layerlast = gnn.layerlast
    siamese_gnn_new = Siamese_GNN(args.num_features, args.num_layers_test, args.J+2, gnn=gnn_new)
    return siamese_gnn_new


if __name__ == '__main__':
    logger = Logger(args.path_logger)
    logger.write_settings(args)
    siamese_gnn = Siamese_GNN(args.num_features, args.num_layers, args.J + 2)
    if torch.cuda.is_available():
        siamese_gnn.cuda()
    gen = Generator(args.path_dataset)
    # generator setup
    gen.num_examples_train = args.num_examples_train
    gen.num_examples_test = args.num_examples_test
    gen.J = args.J
    gen.N_train = args.N_train
    gen.N_test = args.N_test
    gen.edge_density = args.edge_density
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    # load dataset
    # print(gen.random_noise)
    gen.load_dataset()
    if args.mode == 'train':
        train(siamese_gnn, logger, gen)
    # elif args.mode == 'test':

    test(siamese_gnn, logger, gen)
    print ('Finished testing the trained GNN')

    siamese_gnn_test = extend_gnn_append(siamese_gnn)
    test(siamese_gnn_test, logger, gen)
    print ('Finished testing the extended GNN')
