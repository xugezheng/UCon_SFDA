#!/bin/bash

##################### Shell arguments descriptions #####################
# $1: should be the path to config.yaml
########################################################################

##################### slurm Arguments ##################################
#SBATCH --array=0-2
#SBATCH --job-name=narval_domainnet_autoUCon_SFDA_cal
#SBATCH --account=def-ling
#SBATCH --time=00-02:55   # time (DD-HH:MM)
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=xugezheng@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --output=../slurm_output/2025_iclr_formal/domainnet-slurm-%A-%x-%a.out
#######################################################################


ENVDIR=~/envs/dl
PROJECT_DIR=~/scratch/sfda_lln_extend/UCon_SFDA
module load StdEnv/2020 cuda scipy-stack python/3.8
source $ENVDIR/bin/activate

############# DATA pre ###########
DATASOURCE_DIR=~/scratch/sfda_lln_extend/DATASOURCE

cd $SLURM_TMPDIR 
mkdir DATASOURCE
cd DATASOURCE

echo "copy begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
cp $DATASOURCE_DIR/domainnet.zip $SLURM_TMPDIR/DATASOURCE
echo "copy end ... $(date +%Y-%m-%d" "%H:%M:%S)"
echo "unzip begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
unzip domainnet.zip -d $SLURM_TMPDIR/DATASOURCE
echo "unzip end ... $(date +%Y-%m-%d" "%H:%M:%S)"
ls
cd domainnet 
unzip real.zip -d $SLURM_TMPDIR/DATASOURCE/domainnet > /dev/null
unzip sketch.zip -d $SLURM_TMPDIR/DATASOURCE/domainnet > /dev/null
unzip painting.zip -d $SLURM_TMPDIR/DATASOURCE/domainnet > /dev/null
unzip clipart.zip -d $SLURM_TMPDIR/DATASOURCE/domainnet > /dev/null
unzip image_list.zip -d $SLURM_TMPDIR/DATASOURCE/domainnet > /dev/null
# for domainnet 126
cp -r $DATASOURCE_DIR/domainnet/image_list_126/ $SLURM_TMPDIR/DATASOURCE/domainnet
echo Current dir is `pwd`
ls
############# DATA pre done ###########

SEED=`expr $SLURM_ARRAY_TASK_ID + 2019`
wandb offline

cd $PROJECT_DIR
echo Current dir is `pwd`
echo $(date +%Y-%m-%d" "%H:%M:%S)
python train_target_UCon_SFDA.py \
 --seed $SEED \
 --dset domainnet \
 --max_epoch 45 --interval 45 \
 --num_workers 6 \
 --source $1 --target $2 \
 --net resnet50 --net_mode fbc --list_name image_list \
 --root $SLURM_TMPDIR/DATASOURCE \
 --model_root ~/scratch/sfda_lln_extend/Models/partial_label_final \
 --output_dir ~/scratch/sfda_lln_extend/output/partial_label \
 --log_dir ~/scratch/sfda_lln_extend/logs/partial_label \
 --lr_decay True --lr_decay_type shot --lr 0.001 --lr_F_coef 0.1 --weight_decay 0.001 --lr_power 0.3 \
 --K 2 --alpha 1 --beta 0.75 --alpha_decay True \
 --dc_loss_type ce --dc_coef 0.5 --dc_temp 1.0 \
 --partial_coef 0.1 --partial_k 2 --sample_selection_R 1.1 \
 --warmup_eval_iter_num 3 \
 --dc_coef_type init_incons --partial_k_type cal --tau_type cal \
 --expname iclr_2025_formal \
 --key_info domainnet126_autoUCon_sfda_cal

echo $(date +%Y-%m-%d" "%H:%M:%S)