#!/bin/bash

# all commands that start with SBATCH contain commands that are just used by SLURM forscheduling
#################
# set a job name
#SBATCH --job-name=disac5_otf_gat_nf2_nl30_seeded
#################
# a file for job output, you can check job progress
#SBATCH --output=disac5_otf_gat_nf2_nl30_seeded.out
#################
# a file for errors from the job
#SBATCH --error=disac5_otf_gat_nf2_nl30_seeded.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the
# fasteryour job will run.
# Default is one hour, this example will run in less that 5 minutes.
#SBATCH --time=03:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:4
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

module load python-3.6
source activate py36_torch4

python3 gat_mcd_otf.py --path_dataset '/data/chenzh/QAP/data/dataset25_J2' \
--path_logger '' \
--path_gnn '' \
--filename_existing_gnn '' \
--num_examples_train 3200 \
--num_examples_test 100 \
--p_SBM 0.0 \
--q_SBM 0.045 \
--random_noise \
--noise 0.03 \
--noise_model 2 \
--generative_model 'SBM_multiclass' \
--iterations 3200 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_layers 30 \
--num_features 2 \
--J 2 \
--N_train 400 \
--N_test 400 \
--print_freq 10 \
--n_classes 5 \
--lr 0.004 \
--nb_heads 1 \
--dropout 0.0 \
--patience 3200 \
--epochs 3200 \
