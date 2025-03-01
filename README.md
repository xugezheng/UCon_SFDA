# Repo for UCon-SFDA

This will be the official git repo for our 2025 ICLR paper "[Revisiting Source-Free Domain Adaptation: a New Perspective via Uncertainty Control](https://openreview.net/forum?id=nx9Z5Kva96&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))".

---
## Get Started

### 1. Data Preperation

For office-31, please download the data from [here](https://drive.google.com/file/d/1dmEhhzoP-dnVOqsV_zDUNRKUy4dKN6Bl/view?usp=sharing). Unzip the `office.zip` into `./DATASOURCE/`. Make sure the `image_list` folder is in the `./DATASOURCE/office/`.

For office-home dataset, please download the data from [here](https://drive.google.com/file/d/1gRSeLmBTKWjSiqe6jRWaNsXTqzDdqKPr/view?usp=sharing). Unzip the `office-home.zip` into `./DATASOURCE/`. Make sure the `image_list_nrc` (for office-home) and `image_list_partial` (for partial set) are in the `./DATASOURCE/VISDA-C/` folder.

For VisDA2017 dataset, the original data can be downloaded from [here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification). Extract `train.tar` and `validation.tar` to ./DATASOURCE/VISDA-C. Make sure the `image_list` (for VisDA2017) and `image_list_imb` (for VisDA-RUST provided by [I-SFDA](https://github.com/LeoXinhaoLee/Imbalanced-Source-free-Domain-Adaptation)) are in the `./DATASOURCE/VISDA-C/` folder.

For DomainNet126 dataset, the original dataset can be downloaded from [here](https://ai.bu.edu/M3SDA/). Please extract the image data of each `DOMAIN` into `./DATASOURCE/domainnet/DOMAIN`. Make sure the `image_list_126` folder is in the `./DATASOURCE/domainnet/`.

---
### 2. Model Preperation

We utilized the source models of Office-Home and VisDA provided by previous gitRepos (including [SHOT](https://github.com/tim-learn/SHOT) and [NRC](https://github.com/Albert0147/NRC_SFDA)), and trained the source models on Office-31 and Domainnet126 by ourself following [SHOT](https://github.com/tim-learn/SHOT). All model checkpoints are provided [here](https://drive.google.com/drive/folders/18h1YhquJInMCkqxXlSaHax0hCJWMFgpX?usp=sharing).

To use the source model and implement the target adaptation, please download the provided model zip files (`generalDA.zip` and `specialCase.zip`) and unzip them into `./Models/`.

Or you can train your own source model from scratch using the [source code](https://github.com/tim-learn/SHOT).

The model dir should be consistent to the `--model_root` argument.

---
### 3. Target Adaptation

To implement the target domain adaptation, please use the `yml` scripts provided in `./scripts/configs` or try the following command (for example, on DomainNet126 dataset): 

```shell

# for the basic UCon-SFDA with more hyperparams
python train_target_UCon_SFDA.py \
 --seed 2019 \
 --dset domainnet \
 --max_epoch 45 --interval 45 \
 --num_workers 6 \
 --source c --target s \
 --net resnet50 --net_mode fbc --list_name image_list \
 --root ./DATASOURCE \
 --model_root ./Models/generalDA \
 --output_dir ./output \
 --log_dir ./logs \
 --lr_decay True --lr_decay_type shot --lr_F_coef 0.1 --lr_power 0.3 \
 --K 2 --alpha 1 --beta 0.75 --alpha_decay True \
 --dc_loss_type ce --dc_coef 0.5 \
 --partial_coef 0.1 --partial_k 2 --sample_selection_R 1.1 \
 --warmup_eval_iter_num 3 \
 --dc_coef_type fixed --partial_k_type fixed --tau_type fixed \
 --expname iclr_2025_formal \
 --key_info domainnet126_UCon_sfda

```

or 

```shell

# for the autoUCon_SFDA with tau_type being cal or stat
python train_target_UCon_SFDA.py \
 --seed 2019 \
 --dset domainnet \
 --max_epoch 45 --interval 45 \
 --num_workers 6 \
 --source c --target s \
 --net resnet50 --net_mode fbc --list_name image_list \
 --root ./DATASOURCE \
 --model_root ./Models/generalDA \
 --output_dir ./output \
 --log_dir ./logs \
 --lr_decay True --lr_decay_type shot --lr_F_coef 0.1 --lr_power 0.3 \
 --K 2 --alpha 1 --beta 0.75 --alpha_decay True \
 --dc_loss_type ce \
 --partial_coef 0.1 \
 --warmup_eval_iter_num 3 \
 --dc_coef_type init_incons --partial_k_type cal --tau_type cal \
 --expname iclr_2025_formal \
 --key_info domainnet126_autoUCon_sfda_cal
```


> Notes: The priority of the YAML configuration file is higher than command-line arguments.


