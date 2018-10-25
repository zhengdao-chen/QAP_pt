#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx

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


class Generator(object):
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.num_examples_train = 10e6
        self.num_examples_test = 10e4
        self.data_train = []
        self.data_test = []
        self.J = 3
        self.N_train = 50
        self.N_test = 100
        # self.generative_model = 'ErdosRenyi'
        self.generative_model = 'SBM'
        self.edge_density = 0.2
        self.random_noise = False
        self.noise = 0.03
        self.noise_model = 2
        self.p_SBM = 0.8
        self.q_SBM = 0.2
        self.n_classes = 5

    def SBM(self, p, q, N):
        W = np.zeros((N, N))

        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        n = N // 2

        W[:n, :n] = np.random.binomial(1, p, (n, n))
        W[n:, n:] = np.random.binomial(1, p, (N-n, N-n))
        W[:n, n:] = np.random.binomial(1, q, (n, N-n))
        W[n:, :n] = np.random.binomial(1, q, (N-n, n))
        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        blockA = perm < n
        labels = blockA * 2 - 1

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def SBM_multiclass(self, p, q, N, n_classes):
        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        prob_mat = np.ones((N, N)) * q_prime

        n = N // n_classes

        for i in range(n_classes):
            prob_mat[i * n : (i+1) * n, i * n : (i+1) * n] = p_prime

        # print ('prob mat', prob_mat)

        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        # blockA = perm < n
        # labels = blockA * 2 - 1
        labels = (perm // n)

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def ErdosRenyi(self, p, N):
        W = np.zeros((N, N))
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                add_edge = (np.random.uniform(0, 1) < p)
                if add_edge:
                    W[i, j] = 1
                W[j, i] = W[i, j]
        return W

    def ErdosRenyi_netx(self, p, N):
        g = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph_netx(self, p, N):
        """ Generate random regular graph """
        d = p * N
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def compute_operators(self, W):
        N = W.shape[0]
        # print ('W', W)
        # print ('W size', W.size())
        if (self.generative_model == 'ErdosRenyi') or (self.generative_model == 'SBM') or (self.generative_model == 'SBM_multiclass'):
            # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
            d = W.sum(1)
            D = np.diag(d)
            QQ = W.copy()
            WW = np.zeros([N, N, self.J + 2])
            WW[:, :, 0] = np.eye(N)
            for j in range(self.J):
                WW[:, :, j + 1] = QQ.copy()
                # QQ = np.dot(QQ, QQ)
                QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
            WW[:, :, self.J] = D
            WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
            WW = np.reshape(WW, [N, N, self.J + 2])
            x = np.reshape(d, [N, 1])
        elif self.generative_model == 'Regular':
            # operators: {Id, A, A^2}
            ds = []
            ds.append(W.sum(1))
            QQ = W.copy()
            WW = np.zeros([N, N, self.J + 2])
            WW[:, :, 0] = np.eye(N)
            for j in range(self.J):
                WW[:, :, j + 1] = QQ.copy()
                # QQ = np.dot(QQ, QQ)
                QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
                ds.append(QQ.sum(1))
            d = ds[1]
            D = np.diag(ds[1])
            WW[:, :, self.J] = D
            WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
            WW = np.reshape(WW, [N, N, self.J + 2])
            x = np.reshape(d, [N, 1])
        else:
            raise ValueError('Generative model {} not implemented'
                             .format(self.generative_model))
        # print ('J', self.J)
        # print ('WW size', WW.size())
        return WW, x

    def compute_example_train(self):
        example = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.N_train)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.N_train)
        elif self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_train)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N_train)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N_train)
            noise2 = self.ErdosRenyi_netx(pe2, self.N_train)
            print (W)
            print (1-W)
            print (noise1)
            print (noise2)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        WW, x = self.compute_operators(W)
        WW_noise, x_noise = self.compute_operators(W_noise)
        example['WW'], example['x'] = WW, x
        example['WW_noise'], example['x_noise'] = WW_noise, x_noise
        return example

    def compute_example_train_bcd(self):
        example = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.N_train)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.N_train)
        elif self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_train)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, self.N_train, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N_train)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N_train)
            noise2 = self.ErdosRenyi_netx(pe2, self.N_train)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        WW, x = self.compute_operators(W)
        WW_noise, x_noise = self.compute_operators(W_noise)
        example['WW'], example['x'] = WW, x
        example['WW_noise'], example['x_noise'] = WW_noise, x_noise
        example['labels'] = labels
        example['W'] = W
        return example

    def compute_example_test(self):
        example = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.N_test)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.N_test)
        elif self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_test)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_test, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N_test)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N_test)
            noise2 = self.ErdosRenyi_netx(pe2, self.N_test)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        WW, x = self.compute_operators(W)
        WW_noise, x_noise = self.compute_operators(W_noise)
        example['WW'], example['x'] = WW, x
        example['WW_noise'], example['x_noise'] = WW_noise, x_noise
        return example

    def compute_example_test_bcd(self):
        example = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.N_test)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.N_test)
        elif self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_test)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, self.N_test, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N_test)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N_test)
            noise2 = self.ErdosRenyi_netx(pe2, self.N_test)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        WW, x = self.compute_operators(W)
        WW_noise, x_noise = self.compute_operators(W_noise)
        example['WW'], example['x'] = WW, x
        example['WW_noise'], example['x_noise'] = WW_noise, x_noise
        example['labels'] = labels
        example['W'] = W
        return example

    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example_train_bcd()
            self.data_train.append(example)

    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example_test_bcd()
            self.data_test.append(example)

    def load_dataset(self):
        # load train dataset
        if self.random_noise:
            filename = 'QAPtrain_RN.np'
        else:
            filename = ('QAPtrain_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading training dataset at {}'.format(path))
            self.data_train = np.load(open(path, 'rb'))
        else:
            print('Creating training dataset.')
            self.create_dataset_train()
            print('Saving training datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_train)
        # load test dataset
        if self.random_noise:
            filename = 'QAPtest_RN.np'
        else:
            filename = ('QAPtest_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading testing dataset at {}'.format(path))
            self.data_test = np.load(open(path, 'rb'))
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_test)

    def sample_batch(self, num_samples, is_training=True,
                     cuda=True, volatile=False):
        if is_training:
            WW_size = self.data_train[0]['WW'].shape
            x_size = self.data_train[0]['x'].shape
        else:
            WW_size = self.data_test[0]['WW'].shape
            x_size = self.data_test[0]['x'].shape

        WW = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X = torch.zeros(x_size).expand(num_samples, *x_size)
        WW_noise = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X_noise = torch.zeros(x_size).expand(num_samples, *x_size)

        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        for b in range(num_samples):
            if is_training:
                ind = np.random.randint(0, len(dataset))
            else:
                ind = b
            ww = torch.from_numpy(dataset[ind]['WW'])
            x = torch.from_numpy(dataset[ind]['x'])
            WW[b] = ww
            X[b] = x
            ww_noise = torch.from_numpy(dataset[ind]['WW_noise'])
            x_noise = torch.from_numpy(dataset[ind]['x_noise'])
            WW_noise[b] = ww_noise
            X_noise[b] = x_noise

        WW = Variable(WW, volatile=volatile)
        X = Variable(X, volatile=volatile)
        WW_noise = Variable(WW_noise, volatile=volatile)
        X_noise = Variable(X_noise, volatile=volatile)

        # print ('WW size', WW.size())
        # print ('X size', X.size())
        # print ('WW noise size', WW_noise.size())
        # print ('X noise size', X_noise.size())

        if cuda:
            return [WW.cuda(), X.cuda()], [WW_noise.cuda(), X_noise.cuda()]
        else:
            return [WW, X], [WW_noise, X_noise]

    def sample_batch_bcd(self, num_samples, is_training=True,
                     cuda=True, volatile=False):
        if is_training:
            WW_size = self.data_train[0]['WW'].shape
            x_size = self.data_train[0]['x'].shape
            labels_size = self.data_train[0]['labels'].shape
            W_size = self.data_train[0]['W'].shape
        else:
            WW_size = self.data_test[0]['WW'].shape
            x_size = self.data_test[0]['x'].shape
            labels_size = self.data_test[0]['labels'].shape
            W_size = self.data_test[0]['W'].shape

        # WW = torch.zeros(WW_size).expand(num_samples, *WW_size)
        # X = torch.zeros(x_size).expand(num_samples, *x_size)
        # Labels =  torch.zeros(labels_size).expand(num_samples, *labels_size)
        # W = torch.zeros(W_size).expand(num_samples, *W_size)

        WW = np.zeros([num_samples, WW_size[0], WW_size[1], WW_size[2]])
        X = np.zeros([num_samples, x_size[0], x_size[1]])
        Labels = np.zeros([num_samples, labels_size[0]])
        W = np.zeros([num_samples, W_size[0], W_size[1]])

        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        for b in range(num_samples):
            if is_training:
                ind = np.random.randint(0, len(dataset))
            else:
                ind = b

            # ww = torch.from_numpy(dataset[ind]['WW'])
            # x = torch.from_numpy(dataset[ind]['x'])
            # labels = torch.from_numpy(dataset[ind]['labels'])
            ww = dataset[ind]['WW']
            x = dataset[ind]['x']
            labels = dataset[ind]['labels']

            w = dataset[ind]['W']

            # print ('b', b)
            # print ('bth slice', WW[b, :, :, :])
            WW[b, :, :, :] = ww
            X[b, :, :] = x
            Labels[b, :] = labels
            # print ('b', b)
            # print ('ind', ind)
            # print ('ww', WW[b])
            # print ('X', X[b])
            W[b, :, :] = w
        #     print ('ww', ww)
        #     print ('xx', x)
        #     print ('WWfrom sample batch', WW)
        #     print ('X from sample batch', X)
        # print ('WWfrom sample batch', WW)
        # print ('X from sample batch', X)

        WW = Variable(torch.from_numpy(WW).float(), volatile=volatile)
        X = Variable(torch.from_numpy(X).float(), volatile=volatile)
        Labels = Variable(torch.from_numpy(Labels), volatile=volatile)
        # WW = Variable(WW, volatile=volatile)
        # X = Variable(X, volatile=volatile)
        # Labels = Variable(Labels, volatile=volatile)
        # W = Variable(W, volatile=volatile)

        # print ('WWfrom sample batch', WW)
        # print ('X from sample batch', X)

        # print ('WW size', WW.size())
        # print ('X size', X.size())
        # print ('WW noise size', WW_noise.size())
        # print ('X noise size', X_noise.size())

        if cuda:
            # print ('WW.cuda() from sample batch', WW.cuda())
            # print ('X.cuda() from sample batch', X.cuda())
            return [WW.cuda(), X.cuda()], Labels.cuda(), W
        else:
            return [WW, X], Labels, W

    def sample_single(self, ind, is_training=True, cuda=True, volatile=False):

        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test

        # ww = torch.from_numpy(dataset[ind]['WW'])
        # x = torch.from_numpy(dataset[ind]['x'])
        # labels = torch.from_numpy(dataset[ind]['labels'])
        WW = dataset[ind]['WW']
        X = dataset[ind]['x']
        Labels = dataset[ind]['labels']

        W = dataset[ind]['W']

        # print ('b', b)
        # print ('bth slice', WW[b, :, :, :])
        #     print ('ww', ww)
        #     print ('xx', x)
        #     print ('WWfrom sample batch', WW)
        #     print ('X from sample batch', X)
        # print ('WWfrom sample batch', WW)
        # print ('X from sample batch', X)

        if is_training:
            volatile = False
        else:
            volatile = True

        WW = Variable(torch.from_numpy(WW).float(), volatile=volatile).unsqueeze(0)
        X = Variable(torch.from_numpy(X).float(), volatile=volatile).unsqueeze(0)
        Labels = Variable(torch.from_numpy(Labels), volatile=volatile).unsqueeze(0)
        W = np.expand_dims(W, 0)
        # WW = Variable(WW, volatile=volatile)
        # X = Variable(X, volatile=volatile)
        # Labels = Variable(Labels, volatile=volatile)
        # W = Variable(W, volatile=volatile)

        # print ('WWfrom sample batch', WW)
        # print ('X from sample batch', X)

        # print ('WW size', WW.size())
        # print ('X size', X.size())
        # print ('WW noise size', WW_noise.size())
        # print ('X noise size', X_noise.size())

        if cuda:
            return [WW.cuda(), X.cuda()], Labels.cuda(), W
        else:
            return [WW, X], Labels, W

    def sample_batch_bcd_new(self, num_samples, is_training=True,
                     cuda=True, volatile=False):
        if is_training:
            WW_size = self.data_train[0]['WW'].shape
            x_size = self.data_train[0]['x'].shape
            labels_size = self.data_train[0]['labels'].shape
            N = self.data_train[0]['W'].shape[0]
        else:
            WW_size = self.data_test[0]['WW'].shape
            x_size = self.data_test[0]['x'].shape
            labels_size = self.data_test[0]['labels'].shape
            N = self.data_test[0]['W'].shape[0]

        WW = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X = torch.zeros(x_size).expand(num_samples, *x_size)
        Labels =  torch.zeros(labels_size).expand(num_samples, *labels_size)
        # W = torch.zeros(W_size).expand(num_samples, *W_size)

        W = np.zeros(num_samples, N, N)

        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        for b in range(num_samples):
            if is_training:
                ind = np.random.randint(0, len(dataset))
            else:
                ind = b
            ww = torch.from_numpy(dataset[ind]['WW'])
            x = torch.from_numpy(dataset[ind]['x'])
            labels = torch.from_numpy(dataset[ind]['labels'])
            W = dataset[ind]['W']
            WW[b] = ww
            X[b] = x
            Labels[b] = labels
            W[b, :, :] = W

        WW = Variable(WW, volatile=volatile)
        X = Variable(X, volatile=volatile)
        Labels = Variable(Labels, volatile=volatile)
        # W = Variable(W, volatile=volatile)

        # print ('WW size', WW.size())
        # print ('X size', X.size())
        # print ('WW noise size', WW_noise.size())
        # print ('X noise size', X_noise.size())

        if cuda:
            return [WW.cuda(), X.cuda()], Labels.cuda(), W
        else:
            return [WW, X], Labels, W

    def sample_otf_single(self, is_training=True, cuda=True):
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, self.N_train)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, self.N_train, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N_train)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N_train)
            noise2 = self.ErdosRenyi_netx(pe2, self.N_train)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'.format(self.noise_model))
        labels = np.expand_dims(labels, 0)
        labels = Variable(torch.from_numpy(labels), volatile=not is_training)
        W = np.expand_dims(W, 0)
        return W, labels



if __name__ == '__main__':
    ###################### Test Generator module ##############################
    path = '/home/chenzh/tmp/'
    gen = Generator(path)
    gen.num_examples_train = 10
    gen.num_examples_test = 10
    gen.N = 50
    # gen.generative_model = 'Regular'
    gen.generative_model = 'SBM'
    gen.load_dataset()
    g1, g2 = gen.sample_batch(32, cuda=False)
    print(g1[0].size())
    print(g1[1][0].data.cpu().numpy())
    W = g1[0][0, :, :, 1]
    W_noise = g2[0][0, :, :, 1]
    print(W, W.size())
    print(W_noise.size(), W_noise)
    ################### Test graph generators networkx ########################
    # path = '/home/anowak/tmp/'
    # gen = Generator(path)
    # p = 0.2
    # N = 50
    # # W = gen.ErdosRenyi_netx(p, N)
    # W = gen.RegularGraph_netx(3, N)
    # G = networkx.from_numpy_matrix(W)
    # networkx.draw(G)
    # # plt.draw(G)
    # plt.savefig('/home/anowak/tmp/prova.png')
    # print('W', W)
