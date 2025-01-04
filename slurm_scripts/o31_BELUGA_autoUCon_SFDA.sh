#!/bin/bash

##################### Shell arguments descriptions #####################
# $1: should be the path to config.yaml
########################################################################

##################### slurm Arguments ##################################
#SBATCH --array=0-2
#SBATCH --job-name=beluga_o31_autoUCon_SFDA_cal
#SBATCH --account=def-ling
#SBATCH --time=00-02:55   # time (DD-HH:MM)
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=xugezheng@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --output=../slurm_output/2025_iclr_formal/o31-slurm-%A-%x-%a.out
#######################################################################


ENVDIR=~/envs/dl2023
PROJECT_DIR=~/scratch/sfda_lln_extend/UCon_SFDA
module load StdEnv/2020 cuda scipy-stack/2023b python/3.10.2
source $ENVDIR/bin/activate

############# DATA pre ###########
DATASOURCE_DIR=~/scratch/sfda_lln_extend/DATASOURCE

cd $SLURM_TMPDIR 
mkdir DATASOURCE
cd DATASOURCE

echo "copy begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
cp $DATASOURCE_DIR/office.zip $SLURM_TMPDIR/DATASOURCE
echo "copy end ... $(date +%Y-%m-%d" "%H:%M:%S)"

echo "unzip begin ... $(date +%Y-%m-%d" "%H:%M:%S)"
unzip $SLURM_TMPDIR/DATASOURCE/office.zip -d $SLURM_TMPDIR/DATASOURCE > /dev/null

cd office
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
 --dset office \
 --max_epoch 50 --interval 15 \
 --num_workers 1 \
 --source $1 --target $2 \
 --net resnet50 --net_mode fc --list_name image_list \
 --root $SLURM_TMPDIR/DATASOURCE \
 --model_root ~/scratch/sfda_lln_extend/Models/aad \
 --output_dir ~/scratch/sfda_lln_extend/output/partial_label \
 --log_dir ~/scratch/sfda_lln_extend/logs/partial_label \
 --lr_decay False --lr_decay_type shot --lr_F_coef 0.1 --weight_decay 0.001 \
 --K 3 --alpha 1 --beta 1 --alpha_decay True \
 --dc_loss_type ce --dc_coef 1.0 --dc_temp 1.0 \
 --partial_coef 0.05 --partial_k 2 --sample_selection_R 1.3 \
 --dc_coef_type init_incons --partial_k_type cal --tau_type cal \
 --warmup_eval_iter_num 10 \
 --expname iclr_2025_formal \
 --key_info o31_autoUCon_sfda_cal

echo $(date +%Y-%m-%d" "%H:%M:%S)


