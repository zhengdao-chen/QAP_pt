#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator_mod import Generator
from model import GNN, Gconv, GNN_bcd
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
from losses import compute_loss_multiclass, compute_accuracy_multiclass
# import losses

from Logger import compute_accuracy_bcd

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
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10.5f} {:<10.5f} \n'

def compute_loss(pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return criterion(pred, labels)

def compute_cosine_loss_bcd(pred, labels):
    # pred = pred.view(-1, pred.size()[-1])
    # labels = labels.view(-1)
    if (pred.data.shape[0] == 1):
        print (pred)
        print (labels)
        pred = pred.squeeze(0)
        labels_01_1 = ((labels + 1) / 2).squeeze(0)
        labels_01_2 = ((-1 * labels + 1) / 2).squeeze(0)
        print (labels_01_1)
        # print (labels_01_1.type(dtype_l))
        # print (labels_01_1.type(dtype_l).shape)
        loss = pred.dot(labels_01_1.type(dtype)).pow(2) * (-1)
    else:
        raise ValueError('batch size greater than 1')
    return loss

def compute_loss_bcd(pred, labels):
    # pred = pred.view(-1, pred.size()[-1])
    # labels = labels.view(-1)
    if (pred.data.shape[0] == 1):
        pred = pred.squeeze(0)
        labels_01_1 = ((labels + 1) / 2).squeeze(0)
        labels_01_2 = ((-1 * labels + 1) / 2).squeeze(0)
        # print (labels_01_1.type(dtype_l))
        # print (labels_01_1.type(dtype_l).shape)
        loss1 = criterion(pred, labels_01_1.type(dtype_l))
        loss2 = criterion(pred, labels_01_2.type(dtype_l))
        loss = torch.min(loss1, loss2)
    else:
        # print ('pred', pred)
        # print ('labels' labels)
        loss = 0
        for i in range(pred.data.shape[0]):
            pred_single = pred[i, :, :]
            labels_single = labels[i, :]
            # pred = pred.squeeze(0)
            labels_01_1 = ((labels_single + 1) / 2)#.squeeze(0)
            labels_01_2 = ((-1 * labels_single + 1) / 2)#.squeeze(0)
            # print (labels_01_1.type(dtype_l))
            # print (labels_01_1.type(dtype_l).shape)
            loss1 = criterion(pred_single, labels_01_1.type(dtype_l))
            loss2 = criterion(pred_single, labels_01_2.type(dtype_l))
            loss_single = torch.min(loss1, loss2)
            loss += loss_single
    return loss

def train(siamese_gnn, logger, gen):
    labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
              gen.N_train)).type(dtype_l))
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=args.lr)
    for it in range(args.iterations):
        start = time.time()
        input = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        # print ('input', input)
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

def train_bcd(gnn, logger, gen, iters=args.iterations):
    # labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
    #           gen.N_train)).type(dtype_l))
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    for it in range(iters):
        start = time.time()
        input, labels, W = gen.sample_batch_bcd(batch_size, cuda=torch.cuda.is_available())
        # print ('input', input)
        pred = gnn(input)
        # print ('input', input[0])
        # print (pred)
        labels = labels.type(dtype_l)
        # print ('pred', pred)
        # print ('labels', labels)
        loss = compute_loss_bcd(pred, labels)
        gnn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy_bcd(pred, labels)
        elapsed = time.time() - start
        print_freq = logger.args['print_freq']
        if (it % print_freq == 0) and (it > 0):
            # print ('pred', pred)
            # print ('labels', labels)
            logger.plot_train_accuracy()
            logger.plot_train_loss()
            loss = loss.data.cpu().numpy()#[0]
            info = ['iteration', 'loss', 'accuracy', 'edge_density',
                    'noise', 'model', 'elapsed']
            # out = [it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
            #        args.noise, args.generative_model, elapsed]
            out = [it, sum(logger.loss_train[-print_freq:]).item() / print_freq, sum(logger.accuracy_train[-print_freq:]).item() / print_freq, args.edge_density,
                   args.noise, args.generative_model, elapsed]
            print(template1.format(*info))
            print(template2.format(*out))
            # test(gnn, logger, gen)
        if it % logger.args['save_freq'] == 0:
            logger.save_model(gnn)
            logger.save_results()
    print('Optimization finished.')

def train_mcd(gnn, logger, gen, n_classes, iters=args.iterations):
    # labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
    #           gen.N_train)).type(dtype_l))
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    for it in range(iters):
        start = time.time()
        input, labels, W = gen.sample_batch_bcd(batch_size, cuda=torch.cuda.is_available())
        # print ('input', input)
        pred = gnn(input)
        # print ('input', input[0])
        # print (pred)
        labels = labels.type(dtype_l)
        # print ('pred', pred)
        # print ('labels', labels)
        # print ('n classes', n_classes)
        loss = compute_loss_multiclass(pred, labels, n_classes)
        gnn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy_mcd(pred, labels, n_classes)
        elapsed = time.time() - start
        print_freq = logger.args['print_freq']
        if (it % print_freq == 0) and (it > 0):
            # print ('pred', pred)
            # print ('labels', labels)
            logger.plot_train_accuracy()
            logger.plot_train_loss()
            loss = loss.data.cpu().numpy()#[0]
            info = ['iteration', 'loss', 'accuracy', 'edge_density',
                    'noise', 'model', 'elapsed']
            # out = [it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
            #        args.noise, args.generative_model, elapsed]
            out = [it, sum(logger.loss_train[-print_freq:]).item() / print_freq, sum(logger.accuracy_train[-print_freq:]).item() / print_freq, args.edge_density,
                   args.noise, args.generative_model, elapsed]
            print(template1.format(*info))
            print(template2.format(*out))
            # test(gnn, logger, gen)
        if it % logger.args['save_freq'] == 0:
            logger.save_model(gnn)
            logger.save_results()
    print('Optimization finished.')

def test(siamese_gnn, logger, gen):
    labels = (Variable(torch.arange(0, gen.N_test).unsqueeze(0).expand(batch_size,
              gen.N_test)).type(dtype_l))
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=args.lr)
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

def test_bcd(gnn, logger, gen, eval_vs_train=True):
    # labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
    #           gen.N_train)).type(dtype_l))
    if eval_vs_train:
        gnn.eval()
        print ('status eval')
    else:
        print ('status train')
        gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    print ('########################### testing')
    print ('batch prediction:')
    start = time.time()
    input, labels, W = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    pred = gnn(input)
    labels = labels.type(dtype_l)
    # print ('input', input)
    # print ('pred', pred)
    # print ('labels', labels)


    # print ('pred', pred)
    # print ('labels', labels)
    # loss = compute_loss_bcd(pred, labels)
    # gnn.zero_grad()
    # loss.backward()
    # nn.utils.clip_grad_norm(gnn.parameters(), args.clip_grad_norm)
    # optimizer.step()
    # logger.add_test_loss(loss)
    # print ('pred', pred)
    # print ('labels', labels)
    # print ('#################################3')
    # print ('input', input)
    # for i in range(labels.data.shape[0]):
    #     print ('input WW', input[0][i])
    #     print ('input X', input[1][i])
    #     print ('labels_pred', pred[i, :].squeeze(0))
    #     print ('labels_true', labels[i])
    logger.add_test_accuracy_bcd(pred, labels)
    elapsed = time.time() - start
    print_freq = args.num_examples_test-1

    # print ('loss', loss)
    # if (it == print_freq):
    logger.plot_test_accuracy()
    # logger.plot_test_loss()
    # loss = loss.data.cpu().numpy()#[0]
    info = ['iteration', 'loss', 'accuracy', 'edge_density',
            'noise', 'model', 'elapsed']
    # out = [it, sum(logger.loss_test[-print_freq:]).item() / print_freq, sum(logger.accuracy_test[-print_freq:]).item() / print_freq, args.edge_density,
    #        args.noise, args.generative_model, elapsed]
    out = [0, -1 , logger.accuracy_test[-1].item(), args.edge_density,
           args.noise, args.generative_model, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))
        # test(gnn, logger, gen)
    # if it % logger.args['save_freq'] == 0:
    #     logger.save_model(gnn)
    #     logger.save_results()

    print ('one-by-one prediction:')
    avg_acc = 0
    for itit in range(args.num_examples_test):
        input_single = [input[0][itit, :, :, :].unsqueeze(0), input[1][itit, :, :].unsqueeze(0)]
        labels_single = labels[itit, :].unsqueeze(0)

    # print ('input', input[0])
        pred_single = gnn(input_single)
        logger.add_test_accuracy_bcd(pred_single, labels_single)
        # print ('input single', itit, input_single)
        print ('pred_singe', itit, pred_single)
        print ('label single', itit, labels_single)
        print ('accuracy of a single example:', logger.accuracy_test[-1].item())
        avg_acc += logger.accuracy_test[-1].item()
    print ('accuracy averaged', avg_acc / args.num_examples_test)

    gnn.eval()
    print ('batch prediction again:')
    start = time.time()
    input, labels, W = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    pred = gnn(input)
    labels = labels.type(dtype_l)
    logger.add_test_accuracy_bcd(pred, labels)
    elapsed = time.time() - start
    print_freq = args.num_examples_test-1

    logger.plot_test_accuracy()

    info = ['iteration', 'loss', 'accuracy', 'edge_density',
            'noise', 'model', 'elapsed']

    out = [0, -1 , logger.accuracy_test[-1].item(), args.edge_density,
           args.noise, args.generative_model, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    # if it % logger.args['save_freq'] == 0:
    #     logger.save_model(gnn)
    #     logger.save_results()

def test_mcd(gnn, logger, gen, n_classes, eval_vs_train=True):
    if eval_vs_train:
        gnn.eval()
        print ('status eval')
    else:
        print ('status train')
        gnn.train()
    print ('########################### testing')
    print ('batch prediction:')
    start = time.time()
    input, labels, W = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    pred = gnn(input)
    labels = labels.type(dtype_l)
    logger.add_test_accuracy_mcd(pred, labels, n_classes)
    elapsed = time.time() - start
    print_freq = args.num_examples_test-1

    # print ('loss', loss)
    # if (it == print_freq):
    logger.plot_test_accuracy()
    # logger.plot_test_loss()
    # loss = loss.data.cpu().numpy()#[0]
    info = ['iteration', 'loss', 'accuracy', 'edge_density',
            'noise', 'model', 'elapsed']
    # out = [it, sum(logger.loss_test[-print_freq:]).item() / print_freq, sum(logger.accuracy_test[-print_freq:]).item() / print_freq, args.edge_density,
    #        args.noise, args.generative_model, elapsed]
    out = [0, -1 , logger.accuracy_test[-1].item(), args.edge_density,
           args.noise, args.generative_model, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))
        # test(gnn, logger, gen)
    # if it % logger.args['save_freq'] == 0:
    #     logger.save_model(gnn)
    #     logger.save_results()

    print ('one-by-one prediction:')
    avg_acc = 0
    for itit in range(args.num_examples_test):
        input_single = [input[0][itit, :, :, :].unsqueeze(0), input[1][itit, :, :].unsqueeze(0)]
        labels_single = labels[itit, :].unsqueeze(0)

    # print ('input', input[0])
        pred_single = gnn(input_single)
        logger.add_test_accuracy_mcd(pred_single, labels_single, n_classes)
        # print ('input single', itit, input_single)
        # print ('pred_singe', itit, torch.t(pred_single))
        # print ('label single', itit, torch.t(labels_single))
        # print ('accuracy of a single example:', logger.accuracy_test[-1].item())
        avg_acc += logger.accuracy_test[-1].item()
    print ('accuracy averaged', avg_acc / args.num_examples_test)

    gnn.eval()
    print ('batch prediction again:')
    start = time.time()
    input, labels, W = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    pred = gnn(input)
    labels = labels.type(dtype_l)
    logger.add_test_accuracy_mcd(pred, labels, n_classes)
    elapsed = time.time() - start
    print_freq = args.num_examples_test-1

    logger.plot_test_accuracy()

    info = ['iteration', 'loss', 'accuracy', 'edge_density',
            'noise', 'model', 'elapsed']

    out = [0, -1 , logger.accuracy_test[-1].item(), args.edge_density,
           args.noise, args.generative_model, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    # if it % logger.args['save_freq'] == 0:
    #     logger.save_model(gnn)
    #     logger.save_results()

def compute_test(model):
    loss_total = 0
    acc_total = 0
    
    inputs_batch, labels_batch, W_batch = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
    if (args.eval_vs_train):
        model.eval()
    for i in range(args.num_examples_test):
        start = time.time()
        features = inputs_batch[1][i, :].unsqueeze(0)
        adj = Variable(torch.from_numpy(W_batch[i, :, :]).unsqueeze(0), volatile=False)
        labels = labels_batch[i, :].type(dtype_l)
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


def avgiter(W):
    # init_vec = np.ones(W.shape[0])
    # init_vec[(args.N_test // 2):] = -1
    # init_vec = np.random.permutation(init_vec)
    # vec = init_vec
    vec = np.random.binomial(1, 0.5, (args.N_test)) * 2 - 1
    # print ('W', W)
    for it in range(args.num_iterations_avg):
        vec = W.dot(vec)
        # print ('middle vec', vec)
        vec = np.sign(vec)
    return vec

def test_avgiter_bcd(logger, gen):
    start = time.time()
    test_acc_lst = []
    for it in range(args.num_examples_test):
        input, labels, W = gen.sample_batch_bcd(1, cuda=torch.cuda.is_available(), is_training=False)
        # print ('input', input)
        acc_minilst = []
        # W_numpy = W.cpu().numpy().squeeze(0)
        W_numpy = W
        for attempt in range(5):
            pred = avgiter(W_numpy)
            acc = compute_accuracy_bcd(Variable(torch.from_numpy(pred).type(dtype)), labels)
            acc_minilst.append(acc)
            # print ('labels', labels, 'pred', pred, 'acc', acc)
        acc = max(acc_minilst)
        test_acc_lst.append(acc)

    elapsed = time.time() - start

    info = ['accuracy', 'model', 'elapsed']
    out = [sum(test_acc_lst) / args.num_examples_test, args.generative_model, elapsed]
    # print(template3.format(*info))
    # print(template4.format(*out))
        # test(gnn, logger, gen)
    for i in range(3):
        print (info[i], out[i])


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
        gnn_new._modules['layer{}'.format(i+1)] = gnn._modules['layer{}'.format(int(np.floor(i / num_repeat))+1)]
    gnn_new.layerlast = gnn.layerlast
    siamese_gnn_new = Siamese_GNN(args.num_features, args.num_layers_test, args.J+2, gnn=gnn_new)
    return siamese_gnn_new

def add_linear_modules(dim1, dim2, linear1, linear2, weight1, weight2):
    if(True):
        new_linear = torch.nn.Linear(dim1, dim2)
        new_linear.weight = torch.nn.Parameter(linear1.weight.data * weight1 + linear2.weight.data * weight2)
        new_linear.bias = torch.nn.Parameter(linear1.bias.data * weight1 + linear2.bias.data * weight2)
        return new_linear
    else:
        return None

def extend_gnn_interpolate(siamese_gnn):
    gnn = siamese_gnn.gnn
    J = args.J+2
    gnn_new = GNN(args.num_features, args.num_layers_test, J)
    gnn_new.cuda()
    gnn_new.layer0 = gnn.layer0
    num_repeat = np.ceil(args.num_layers_test / args.num_layers)
    featuremap_mi = [args.num_features, args.num_features, args.num_features]
    dim1 = J*featuremap_mi[0]
    dim2 = featuremap_mi[2] // 2
    for i in range(args.num_layers_test):
        j = i * args.num_layers / args.num_layers_test
        if (j == np.floor(j)):
            gnn_new._modules['layer{}'.format(i+1)] = gnn._modules['layer{}'.format(int(j)+1)]
        else:
            layer_new = Gconv(featuremap_mi, J)

            # layer_new.fc1 = gnn._modules['layer{}'.format(int(np.floor(j))+1)].fc1 * (np.ceil(j) - j) + gnn._modules['layer{}'.format(int(np.ceil(j))+1)].fc1 * (j - np.floor(j))
            # layer_new.fc2 = gnn._modules['layer{}'.format(int(np.floor(j))+1)].fc2 * (np.ceil(j) - j) + gnn._modules['layer{}'.format(int(np.ceil(j))+1)].fc2 * (j - np.floor(j))
            layer_new.fc1 = add_linear_modules(dim1, dim2, gnn._modules['layer{}'.format(int(np.floor(j))+1)].fc1, gnn._modules['layer{}'.format(int(np.ceil(j))+1)].fc1, np.ceil(j) - j, j - np.floor(j))
            layer_new.fc2 = add_linear_modules(dim1, dim2, gnn._modules['layer{}'.format(int(np.floor(j))+1)].fc2, gnn._modules['layer{}'.format(int(np.ceil(j))+1)].fc2, np.ceil(j) - j, j - np.floor(j))
            gnn_new._modules['layer{}'.format(i+1)] = layer_new
    gnn_new.layerlast = gnn.layerlast
    siamese_gnn_new = Siamese_GNN(args.num_features, args.num_layers_test, args.J+2, gnn=gnn_new)
    return siamese_gnn_new


if __name__ == '__main__':
    # print (args.eval_vs_train)

    logger = Logger(args.path_logger)
    logger.write_settings(args)

    # ## One fixed generator
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
    # load dataset
    # print(gen.random_noise)
    gen.load_dataset()
    #

    torch.backends.cudnn.enabled=False

    if (args.mode == 'test'):
        print ('In testing mode')
        # filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(gen.N_test) + '_it' + str(args.iterations)
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
        else:
            print ('No such a gnn exists; creating a brand new one')
            if (args.generative_model == 'SBM'):
                gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2)
            elif (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(args.iterations)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins')
            # train_bcd(gnn, logger, gen)
            # print ('Saving gnn ' + filename)
            # if torch.cuda.is_available():
            #     torch.save(gnn.cpu(), path_plus_name)
            #     gnn.cuda()
            # else:
            #     torch.save(gnn, path_plus_name)

    elif (args.mode == 'train'):
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(args.iterations)
            path_plus_name = os.path.join(args.path_gnn, filename)
        # if (os.path.exists(path_plus_name)):
        #     print ('loading gnn ' + filename)
        #     gnn = torch.load(path_plus_name)
        # else:
        else:
            print ('No such a gnn exists; creating a brand new one')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(args.iterations)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM'):
                gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2)
            elif (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM'):
            train_bcd(gnn, logger, gen)
        elif (args.generative_model == 'SBM_multiclass'):
            train_mcd(gnn, logger, gen, args.n_classes)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)

    elif (args.mode == 'search_opt_iters'):
        iters_per_check = 600

        for check_pt in range(int(args.iterations / iters_per_check) + 1):
            filename = args.filename_existing_gnn
            path_plus_name = os.path.join(args.path_gnn, filename)
            if ((filename != '') and (os.path.exists(path_plus_name))):
                print ('Loading gnn ' + filename)
                gnn = torch.load(path_plus_name)
                # gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2)

                filename = filename + '_Ntr' + str(gen.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(iters_per_check * (check_pt))
                path_plus_name = os.path.join(args.path_gnn, filename)
            else:
                print ('creating a brand new gnn')
                if (args.generative_model == 'SBM'):
                    gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2)
                elif (args.generative_model == 'SBM_multiclass'):
                    gnn = GNN_bcd(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
                filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(gen.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(iters_per_check * (check_pt))
                path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins for num_iters = ' + str(iters_per_check * (check_pt)))
            if (args.generative_model == 'SBM'):
                train_bcd(gnn, logger, gen)
            elif (args.generative_model == 'SBM_multiclass'):
                train_mcd(gnn, logger, gen, args.n_classes)
            # print ('Saving gnn ' + filename)
            # if torch.cuda.is_available():
            #     torch.save(gnn.cpu(), path_plus_name)
            #     gnn.cuda()
            # else:
            #     torch.save(gnn, path_plus_name)

            print ('Testing the GNN at check_pt ' + str(check_pt))
            if (args.generative_model == 'SBM'):
                test_bcd(gnn, logger, gen)
            else:
                test_mcd(gnn, logger, gen, args.n_classes)
    #
    # if torch.cuda.is_available():
    #     gnn.cuda()
    #
    # train_bcd(gnn, logger, gen)
    #
    # if ((args.mode == 'train') or (not os.path.exists(path_plus_name))):
    #     print ('saving gnn ' + filename)
    #     torch.save(gnn, path_plus_name)

    print ('Testing the GNN:')
    # test(siamese_gnn, logger, gen)
    if (args.generative_model == 'SBM'):
        test_bcd(gnn, logger, gen, args.eval_vs_train)
    else:
        test_mcd(gnn, logger, gen, args.n_classes, args.eval_vs_train)
    # test_mcd(gnn, logger, gen, args.n_classes, args.eval_vs_train)
    # compute_test(gnn)

    # print ('Testing the GNN extended by appending random layers after itself')
    # siamese_gnn_test = extend_gnn_append_random(siamese_gnn)
    # test(siamese_gnn_test, logger, gen)

    # print ('Testing the GNN extended by appending itself after itself')
    # siamese_gnn_test = extend_gnn_append(siamese_gnn)
    # test(siamese_gnn_test, logger, gen)

    # print ('Testing the GNN extended by repeating each of its layers twice')
    # siamese_gnn_test = extend_gnn_repeat(siamese_gnn)
    # test(siamese_gnn_test, logger, gen)

    # print ('Testing the GNN extended by interpolation')
    # siamese_gnn_test = extend_gnn_interpolate(siamese_gnn)
    # test(siamese_gnn_test, logger, gen)

    ################## Sequence of generators ######################
    # gen_lst = []
    # N_lst = [10, 20, 40, 80]
    #
    # for i in range(len(N_lst)):
    #     gen = Generator(args.path_dataset + '_N' + str(N_lst[i]))
    #     gen.num_examples_train = args.num_examples_train
    #     gen.num_examples_test = args.num_examples_test
    #     gen.J = args.J
    #     gen.N_train = N_lst[i]
    #     gen.N_test = N_lst[i]
    #     gen.p_SBM = args.p_SBM
    #     gen.q_SBM = args.q_SBM
    #     gen.generative_model = args.generative_model
    #     gen_lst.append(gen)
    #
    # for i in range(len(N_lst)):
    #
    #     print ('Training the gnn on graphs with N = ' + str(N_lst[i]))
    #     train_bcd(gnn, logger, gen_lst[i])
    #
    #     for j in range(len(N_lst)):
    #
    #         print ('Testing the trained GNN on graphs with N = ' + str(N_lst[j]))
    #         # test(siamese_gnn, logger, gen)
    #         test_bcd(gnn, logger, gen_lst[j])
    ######################################### avg iteration #######################

    # print ('Testing the avg iteration method')
    # test_avgiter_bcd(logger, gen)
