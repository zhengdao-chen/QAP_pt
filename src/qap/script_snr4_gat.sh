#!/bin/bash

# all commands that start with SBATCH contain commands that are just used by SLURM forscheduling
#################
# set a job name
#SBATCH --job-name=snr4_gat_nf2_nl30_seeded
#################
# a file for job output, you can check job progress
#SBATCH --output=snr4_gat_nf2_nl30_seeded.out
#################
# a file for errors from the job
#SBATCH --error=snr4_gat_nf2_nl30_seeded.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the
# fasteryour job will run.
# Default is one hour, this example will run in less that 5 minutes.
#SBATCH --time=04:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
# We are submitting to the batch partition
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=10000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=zc1216@nyu.edu

source activate py36_torch4
python3 gat_mcd_otf.py --path_dataset '/data/chenzh/QAP/data/dataset11_J2' \
--path_logger '' \
--path_gnn '/data/chenzh/QAP/models' \
--filename_existing_gnn '' \
--num_examples_train 6000 \
--num_examples_test 100 \
--p_SBM 0.0055 \
--q_SBM 0.0005 \
--generative_model 'SBM_multiclass' \
--iterations 6000 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_layers 30 \
--num_features 2 \
--N_train 1000 \
--N_test 1000 \
--print_freq 1 \
--n_classes 2 \
--dropout 0 \
--nb_heads 1 \
--epochs 6000 \
--lr 0.004 \
