#!/bin/bash

##################### Shell arguments descriptions #####################
# $1: should be the path to config.yaml
########################################################################

##################### slurm Arguments ##################################
#SBATCH --array=0-2
#SBATCH --job-name=narval_visda_ucon_sfda
#SBATCH --account=def-boyuwang
#SBATCH --time=00-05:55   # time (DD-HH:MM)
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=xugezheng@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --output=../slurm_output/2025_iclr_formal/visda2017-slurm-%A-%x-%a.out
#######################################################################


ENVDIR=~/envs/dl
module load StdEnv/2020 cuda scipy-stack python/3.8
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

PROJECT_DIR=~/scratch/sfda_lln_extend/UCon_SFDA

############# DATA pre ###########
DATASOURCE_DIR=~/scratch/sfda_lln_extend/DATASOURCE

cd $SLURM_TMPDIR 
mkdir DATASOURCE
cd DATASOURCE

echo "copy begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
cp $DATASOURCE_DIR/VISDA-C.zip $SLURM_TMPDIR/DATASOURCE
echo "copy end ... $(date +%Y-%m-%d" "%H:%M:%S)"

echo "unzip begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
unzip VISDA-C.zip -d $SLURM_TMPDIR/DATASOURCE
echo "unzip end ... $(date +%Y-%m-%d" "%H:%M:%S)"

cd VISDA-C 
echo "unzip train begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
tar xvf train.tar -C $SLURM_TMPDIR/DATASOURCE/VISDA-C > /dev/null
echo "unzip end ... $(date +%Y-%m-%d" "%H:%M:%S)"

echo "unzip validation begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
tar xvf validation.tar -C $SLURM_TMPDIR/DATASOURCE/VISDA-C > /dev/null
echo "unzip end ... $(date +%Y-%m-%d" "%H:%M:%S)"
echo Current dir is `pwd`
ls


SEED=`expr $SLURM_ARRAY_TASK_ID + 2019`
wandb offline

cd $PROJECT_DIR
echo Current dir is `pwd`
echo $(date +%Y-%m-%d" "%H:%M:%S)
python train_target_UCon_SFDA.py \
 --seed $SEED \
 --dset VISDA-C \
 --max_epoch 30 --interval 30 \
 --num_workers 6 \
 --source Tr --target Val \
 --net resnet101 --net_mode fbc \
 --root $SLURM_TMPDIR/DATASOURCE \
 --model_root ~/scratch/sfda_lln_extend/Models/shot \
 --output_dir ~/scratch/sfda_lln_extend/output/partial_label \
 --log_dir ~/scratch/sfda_lln_extend/logs/partial_label \
 --lr_decay False --lr_F_coef 0.05 \
 --K 5 --alpha 1 --beta 5 --alpha_decay True \
 --dc_loss_type ce --dc_coef 1 --dc_temp 1.0 \
 --partial_coef 0.01 --partial_k 1 --sample_selection_R 1.1 \
 --warmup_eval_iter_num -1 \
 --dc_coef_type fixed --partial_k_type fixed --tau_type fixed \
 --expname iclr_2025_formal \
 --key_info visda2017_UCon_sfda
 
echo $(date +%Y-%m-%d" "%H:%M:%S)