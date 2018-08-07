#!/bin/bash

# all commands that start with SBATCH contain commands that are just used by SLURM forscheduling
#################
# set a job name
#SBATCH --job-name=qap10
#################
# a file for job output, you can check job progress
#SBATCH --output=qap10.out
#################
# a file for errors from the job
#SBATCH --error=qap10.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the
# fasteryour job will run.
# Default is one hour, this example will run in less that 5 minutes.
#SBATCH --time=00:10:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:2
# We are submitting to the batch partition
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=40000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=zc1216@nyu.edu

module load python-3.6
srun python3 main_mod.py --path_dataset './data/dataset2' \
--path_logger '' \
--num_examples_train 20000 \
--num_examples_test 1000 \
--edge_density 0.2 \
--random_noise \
--noise 0.03 \
--noise_model 2 \
--generative_model 'ErdosRenyi' \
--iterations 300 \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_features 20 \
--num_layers 20 \
--num_layers_test 40 \
--J 4 \
--N_train 50 \
--N_test 100 \
