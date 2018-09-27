#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator_mod import Generator
from load import get_lg_inputs
from model import lGNN_multiclass
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



def train_mcd(gnn, logger, gen, n_classes, iters=args.iterations):
    # labels = (Variable(torch.arange(0, gen.N_train).unsqueeze(0).expand(batch_size,
    #           gen.N_train)).type(dtype_l))
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    for it in range(iters):
        start = time.time()
        input, labels, W = gen.sample_batch_bcd(batch_size, cuda=torch.cuda.is_available())
        # x = input[1]
        # P = get_P(W.squeeze(0)).unsqueeze(0)
        WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)
        print ('WW', WW.shape)
        print ('WW_lg', WW_lg.shape)
        if (torch.cuda.is_available()):
            WW.cuda()
            x.cuda()
            WW_lg.cuda()
            y.cuda()
            P.cuda()
        # print ('input', input)
        pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
        # print ('input', input[0])
        # print (pred)
        labels = labels.type(dtype_l)
        if (torch.cuda.is_available()):
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.cpu().numpy() == -1):
                labels = (labels + 1)/2
        else:
            if (args.generative_model == 'SBM_multiclass') and (torch.min(labels).data.numpy() == -1):
                labels = (labels + 1)/2
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


# def test_mcd(gnn, logger, gen, n_classes, eval_vs_train=True):
#     if eval_vs_train:
#         gnn.eval()
#         print ('model status: eval')
#     else:
#         print ('model status: train')
#         gnn.train()
#     # print ('########################### testing')
#     # print ('batch prediction:')
#     start = time.time()
#     # input, labels, W = gen.sample_batch_bcd(args.num_examples_test, cuda=torch.cuda.is_available(), is_training=False)
#     # pred = gnn(input)
#
#
#     acc_total = 0
#     loss_total = 0
#     for it in range(args.num_examples_test):
#         inputs, labels, W = gen.sample_single(it, cuda=torch.cuda.is_available(), is_training=False)
#         labels = labels.type(dtype_l)
#         if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
#             labels = (labels + 1)/2
#         WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)
#         print ('WW', WW.shape)
#         print ('WW_lg', WW_lg.shape)
#         if (torch.cuda.is_available()):
#             WW.cuda()
#             x.cuda()
#             WW_lg.cuda()
#             y.cuda()
#             P.cuda()
#         # print ('input', input)
#         pred_single = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
#
#         # input_single = [input[0][itit, :, :, :].unsqueeze(0), input[1][itit, :, :].unsqueeze(0)]
#         labels_single = labels
#
#     # print ('input', input[0])
#         # pred_single = gnn(input_single)
#
#         loss_test = compute_loss_multiclass(pred_single, labels_single, n_classes)
#         acc_test = compute_accuracy_multiclass(pred_single, labels_single, n_classes)
#         # logger.add_test_accuracy_mcd(pred_single, labels_single, n_classes)
#         # print ('input single', itit, input_single)
#         # print ('pred_singe', itit, torch.t(pred_single[0, :, :]))
#         # print ('label single', itit, torch.t(labels_single[0, :]))
#         # print ('accuracy of a single example:', logger.accuracy_test[-1].item())
#         acc_total += acc_test
#         loss_total += loss_test
#
#         if (torch.cuda.is_available()):
#             WW.cpu()
#             x.cpu()
#             WW_lg.cpu()
#             y.cpu()
#             P.cpu()
#         del WW
#         del WW_lg
#         del x
#         del y
#         del P
#
#     elapsed = time.time() - start
#     # if (it % args.print_freq == 0):
#     #     info = ['iter', 'avg loss', 'avg acc', 'edge_density',
#     #             'noise', 'model', 'elapsed']
#     #     out = [it, loss_test, acc_test, args.edge_density,
#     #            args.noise, 'lGNN', elapsed]
#     #     print(template1.format(*info))
#     #     print(template2.format(*out))
#
#     elapsed = time.time() - start
#
#     if(torch.cuda.is_available()):
#         loss_value = float(loss_total.data.cpu().numpy())
#     else:
#         loss_value = float(loss_total.data.numpy())
#
#     info = ['epoch', 'avg loss', 'avg acc', 'edge_density',
#             'noise', 'model', 'elapsed']
#     out = [-1, loss_value / args.num_examples_test, acc_total / args.num_examples_test, args.edge_density,
#            args.noise, 'lGNN', elapsed]
#     print(template1.format(*info))
#     print(template2.format(*out))

def train_mcd_single(gnn, optimizer, logger, gen, n_classes, it):
    start = time.time()
    W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
    labels = labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2

    WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)

    # print ('WW', WW.shape)
    # print ('WW_lg', WW_lg.shape)

    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()
        WW_lg.cuda()
        y.cuda()
        P.cuda()
    # print ('input', input)
    pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))

    loss = compute_loss_multiclass(pred, labels, n_classes)
    gnn.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc = compute_accuracy_multiclass(pred, labels, n_classes)

    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())

    info = ['epoch', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [it, loss_value, acc, args.edge_density,
           args.noise, 'lGNN', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    return loss_value, acc

def train(gnn, logger, gen, n_classes=args.n_classes, iters=args.iterations):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = train_mcd_single(gnn, optimizer, logger, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
    print ('Avg train loss', np.mean(loss_lst))
    print ('Avg train acc', np.mean(acc_lst))

def test_mcd_single(gnn, logger, gen, n_classes, iter):

    start = time.time()
    W, labels = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())
    labels = labels.type(dtype_l)
    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2
    WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)

    print ('WW', WW.shape)
    print ('WW_lg', WW_lg.shape)

    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()
        WW_lg.cuda()
        y.cuda()
        P.cuda()
    # print ('input', input)
    pred_single = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
    labels_single = labels

    loss_test = compute_loss_multiclass(pred_single, labels_single, n_classes)
    acc_test = compute_accuracy_multiclass(pred_single, labels_single, n_classes)

    elapsed = time.time() - start
    # if (it % args.print_freq == 0):
    #     info = ['iter', 'avg loss', 'avg acc', 'edge_density',
    #             'noise', 'model', 'elapsed']
    #     out = [it, loss_test, acc_test, args.edge_density,
    #            args.noise, 'lGNN', elapsed]
    #     print(template1.format(*info))
    #     print(template2.format(*out))

    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss_test.data.cpu().numpy())
    else:
        loss_value = float(loss_test.data.numpy())

    info = ['epoch', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [iter, loss_value, acc_test, args.edge_density,
           args.noise, 'lGNN', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    return loss_value, acc_test

def test(gnn, logger, gen, n_classes, iters=args.num_examples_test):
    gnn.train()
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        # inputs, labels, W = gen.sample_single(it, cuda=torch.cuda.is_available(), is_training=False)
        loss_single, acc_single = test_mcd_single(gnn, logger, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
    print ('Avg test loss', np.mean(loss_lst))
    print ('Avg test acc', np.mean(acc_lst))



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
    gen.n_classes = args.n_classes
    # load dataset
    # print(gen.random_noise)
    # gen.load_dataset()
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
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2)
            elif (args.generative_model == 'SBM_multiclass'):
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(args.iterations)
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
            filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(args.iterations)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM'):
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2)
            elif (args.generative_model == 'SBM_multiclass'):
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM'):
            train_bcd(gnn, logger, gen)
        elif (args.generative_model == 'SBM_multiclass'):
            # train_mcd(gnn, logger, gen, args.n_classes)
            train(gnn, logger, gen, args.n_classes)
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
                    gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2)
                elif (args.generative_model == 'SBM_multiclass'):
                    gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
                filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(gen.N_train) + '_num' + str(args.num_examples_train) + '_it' + str(iters_per_check * (check_pt))
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
    if args.eval_vs_train:
        gnn.eval()
        print ('model status: eval')
    else:
        print ('model status: train')
        gnn.train()
    # test(siamese_gnn, logger, gen)
    # for i in range(100):
    # test_mcd(gnn, logger, gen, args.n_classes, args.eval_vs_train)
    test(gnn, logger, gen, args.n_classes)
