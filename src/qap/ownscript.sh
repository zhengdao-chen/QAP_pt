#!/bin/bash

python main.py --path_dataset './data/dataset1' \
--path_logger '' \
--num_examples_train 20000 \
--num_examples_test 1000 \
--edge_density 0.2 \
--random_noise \
--noise 0.03 \
--noise_model 2 \
--generative_model 'ErdosRenyi' \
--iterations 60000 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_features 20 \
--num_layers 20 \
--J 4 \