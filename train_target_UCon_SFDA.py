import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import ast
import yaml

from vis_sourcefree import *
from loggers import TxtLogger, set_logger
from utils_evaluation import cal_acc, partial_Y_evaluation, evaluate_unlearning_bank, cal_acc_aug
from utils_loss import dc_loss_calculate
from utils_dataloader import data_load_UCon_SFDA
from utils_partial_label import calculate_k_values, partial_label_loss, partial_label_bank_update, selection_mask_bank_update, logits_ratio_calculation, obtain_sample_R_ratio


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, args, gamma=10):
    decay = (1 + gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer, decay


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target_all(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set base network
    model = All_Model(args)

    # ============== output file init ==============
    seed_dir = "seed" + str(args.seed)
    args.name = args.source.upper()[0] + "2" + args.target.upper()[0]
    out_file_dir = osp.join(args.output_dir, args.expname, args.dset, args.name)
    if not osp.exists(out_file_dir):
        os.makedirs(out_file_dir)
    out_file_name = (
        out_file_dir + "/" + "log_" + seed_dir + "_" + args.key_info + ".txt"
    )
    args.out_file = open(out_file_name, "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    # ==========================================

    # ========== log ===========
    log_dir, log_file = set_logger(args)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger, fh, sh = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))
    logger.info(args)
    # ==========================

    # data load
    dset_loaders = data_load_UCon_SFDA(args)
        

    # optimizer
    if args.net_mode == "fbc":
        param_group = [
            {
                "params": model.netF.parameters(),
                "lr": args.lr * args.lr_F_coef,
            },
            {"params": model.netB.parameters(), "lr": args.lr * 1},
            {"params": model.netC.parameters(), "lr": args.lr * 1},
        ]
    elif args.net_mode == "fc":
        param_group = [
            {
                "params": model.netF.feature_layers.parameters(),
                "lr": args.lr * args.lr_F_coef,
            },
            {"params": model.netF.bottle.parameters(), "lr": args.lr * 1},  # 10
            {"params": model.netF.bn.parameters(), "lr": args.lr * 1},  # 10
            {"params": model.netC.parameters(), "lr": args.lr * 1},
        ]

    optimizer = optim.SGD(
        param_group, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )

    # lr_decay
    if args.lr_decay_type == "shot":
        optimizer = op_copy(optimizer)


    # training params
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best = 0
    best_log_str = " "
    test_num = 0
    sample_selection_R = args.sample_selection_R
    consistency = 0
    consistency_0 = 0
    k_values_list = []
    k_stars_list = []
    uncertain_ratio_list = []

    #  building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).to(args.device)
    pure_score_bank = torch.randn(num_sample, args.class_num).to(args.device)
    label_bank = torch.ones(num_sample).long() * -1
    partial_label_bank = torch.zeros(num_sample, args.class_num).long().to(args.device)
    sample_selection_mask_bank = torch.zeros(num_sample).long().to(args.device)

    model.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][1]
            indx = data[-1]
            labels = data[1]
            inputs = inputs.to(args.device)
            output, outputs = model(inputs)
            output_norm = F.normalize(output)
            outputs = nn.Softmax(-1)(outputs)
            

            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone() 
            pure_score_bank[indx] = outputs.detach().clone()
            label_bank[indx] = labels.detach().clone().cpu()
            
         
            if args.partial_k_type == "fixed":
                partial_label_bank = partial_label_bank_update(partial_label_bank, indx, outputs, k_values=args.partial_k)

            elif args.partial_k_type == "cal":
                k_stars, k_values = calculate_k_values(outputs)
                k_stars_list.append(k_stars.detach().clone())
                k_values_list.append(k_values.detach().clone())
                partial_label_bank = partial_label_bank_update(partial_label_bank, indx, outputs, k_values=k_values)
            sample_selection_mask_bank, uncertain_ratio = selection_mask_bank_update(sample_selection_mask_bank, indx, outputs, args, ratio=sample_selection_R)
            uncertain_ratio_list.append(uncertain_ratio.detach().clone())
    init_logits_ratio = logits_ratio_calculation(pure_score_bank.detach().clone())
    sample_selection_R = obtain_sample_R_ratio(args, init_logits_ratio)


    sample_selection_R = evaluate_unlearning_bank(pure_score_bank, label_bank, partial_label_bank, sample_selection_mask_bank, sample_selection_R, uncertain_ratio_list, k_stars_list, k_values_list, args, logger)
    k_stars_list = []
    k_values_list = []
    uncertain_ratio_list = []
    model.train()
        
    # =====================================================
    # ================= PRE TEST ====================
    # =====================================================
    model.eval()
    acc_s_te, acc_list, f1 = cal_acc(dset_loaders["test"], model, args, True)
    log_str = f"===== [Test Evaluation] =====\nTask: {args.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%, F1-Score = {f1:.2f}%\nAcc_list (cls, cls_acc):\n {acc_list}\n"
    acc_s_tr, acc_list_tr, f1_tr, acc_s_tr_aug, acc_list_tr_aug, f1_tr_aug, consistency_0 = cal_acc_aug(dset_loaders["pl"], model, args, True)
    log_str += f'===== [Training Augmentation Evaluation] =====\n Train Org DATA: Accuracy = {acc_s_tr:.2f}%, F1-Score = {f1_tr:.2f}% | Train AUG DATA: Accuracy = {acc_s_tr_aug:.2f}%, F1-Score = {f1_tr_aug:.2f}% \n CONSISTENCY: {consistency_0:.2f}%\n'
    log_str += f'Org Training Acc_list (cls, cls_acc):\n {acc_list_tr}\nAug Training Acc_list (cls, cls_acc):\n {acc_list_tr_aug}\n'
    logger.info(f"==================[Pre Test Info]==================\n{log_str}")
    consistency = consistency_0
    model.train()
    # =====================================================
    # =====================================================
    
    
    loader = dset_loaders["target"]
    logger.info("================== Adaptation Begin ==================")
    while iter_num < max_iter:
        batch_loss_str = f"[Loss Info] Iter [{iter_num}/{max_iter}]: \n"
        try:
            inputs_test_all, tar_real, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test_all, tar_real, tar_idx = next(iter_test)

       
        if inputs_test_all[0].size(0) == 1:
                continue


        inputs1, inputs_test = (
            inputs_test_all[0].to(args.device),
            inputs_test_all[1].to(args.device),
        )


        iter_num += 1

        # ========================================================
        # ============== Lr or other Param Decay ===============
        # ========================================================
        if args.lr_decay:
            if args.lr_decay_type == "shot":
                optimizer, lr_decay = lr_scheduler(
                    optimizer, iter_num=iter_num, max_iter=max_iter, args=args
                )
                batch_loss_str += f"Current lr is netF: {lr_decay * args.lr * args.lr_F_coef:.6f}, netB/C: {lr_decay * args.lr:.6f}\n"
        else:
            lr_decay = 1.0
           

        if args.alpha_decay:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha
        alpha = max(alpha, 5e-6)
        

       
        # ========================================================
        # =================== Model Forward ======================
        # ========================================================
        # main model forward
        features_test, outputs_test = model(inputs_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        # data aug for dispersion control
        features_test1, outputs_test1 = model(inputs1)
        features = []
        features.append(features_test1)
        outputs = []
        outputs.append(outputs_test1)

        # ========================================================
        # ======================== loss ==========================
        # ========================================================
        loss = torch.tensor(0.0).to(args.device)

        # generate pseudo-pred
        with torch.no_grad():
            pred = softmax_out.max(1)[1].long()  # .detach().clone()
        # K_PL update
        if args.partial_k_type == "fixed":
            partial_label_bank = partial_label_bank_update(partial_label_bank, tar_idx, softmax_out, k_values=args.partial_k)
        elif args.partial_k_type == "cal":
            k_stars, k_values = calculate_k_values(softmax_out)
            k_stars_list.append(k_stars.detach().clone())
            k_values_list.append(k_values.detach().clone())
            partial_label_bank = partial_label_bank_update(partial_label_bank, tar_idx, softmax_out, k_values=k_values)
            
        # tau update
        sample_selection_mask_bank, uncertain_ratio = selection_mask_bank_update(sample_selection_mask_bank, tar_idx, softmax_out, args, ratio=sample_selection_R)
        uncertain_ratio_list.append(uncertain_ratio.detach().clone())
        batch_selection_mask = sample_selection_mask_bank[tar_idx]
        batch_selection_indx = torch.nonzero(batch_selection_mask, as_tuple=True)[0]
        batch_loss_str += (
                f"[Sample Selection] Ratio = {sample_selection_R}; Cur Batch size: {tar_idx.size()[0]}; Selected Number per Batch: {batch_selection_indx.size()[0]}; selection bank len: {sample_selection_mask_bank.sum()}\n"
            )
           


        # update neighbor info
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()
            batch_loss_str += f"[Contrastive - Pos]: No Smooth |"
            score_bank[tar_idx] = (
                nn.Softmax(dim=-1)(outputs_test).detach().clone()
            )
            pure_score_bank[tar_idx] = (nn.Softmax(dim=-1)(outputs_test).detach().clone())
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
        

            distance = output_f_ @ fea_bank.T
            dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C
            

        # CL - POS
        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C
        pos_loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        )  
        loss += pos_loss
        batch_loss_str += f" loss is: {pos_loss.item()}; \n"
    
        # CL - NEG
        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0])).to(
            args.device
        )
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#
        dot_neg_all = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg_all * mask).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        neg_loss = neg_pred * alpha
        loss += neg_loss
        batch_loss_str += f"[Contrastive - Neg] (*{alpha}): {neg_pred.item()} * {alpha} = {neg_loss.item()}; \n"
            

            
        
        # NEG - DC
        dc_loss1 = dc_loss_calculate(outputs_test, outputs[0], args) # smooth should be 0.1
        if args.dc_coef_type == "fixed":
            cur_dc_coef_ = args.dc_coef 
        elif args.dc_coef_type == "init_incons": 
            cur_dc_coef_ = (100 - consistency) / 100
        dc_loss = cur_dc_coef_ * dc_loss1
        batch_loss_str += (
        f"[Dispersion Control - {args.dc_loss_type} - {args.dc_coef_type}]: (*{cur_dc_coef_}=){dc_loss.item()}; \n"
        )
        loss += dc_loss
        
        # POS - PL
        if batch_selection_indx.size()[0] > 0:
            partial_Y = partial_label_bank[tar_idx]
            partial_loss = partial_label_loss(outputs_test, partial_Y, args, mask=batch_selection_mask, smooth=0.1) 
                
                
            
            batch_partial_label_Y_acc = partial_Y_evaluation(partial_Y, tar_real)
            batch_selected_partial_label_Y_acc = partial_Y_evaluation(partial_Y[batch_selection_indx.to(partial_Y.device)], tar_real[batch_selection_indx.to(tar_real.device)])
            batch_self_pred_acc = torch.mean((pred.detach().clone().cpu() == tar_real.cpu()).float())
            batch_selected_self_pred_acc = torch.sum((pred.detach().clone().cpu()[batch_selection_indx.cpu()] == tar_real.cpu()[batch_selection_indx.cpu()]).float()) / batch_selection_indx.size()[0]
            loss += args.partial_coef * partial_loss
            batch_loss_str += (
                    f"[Partial Label Loss]: {args.partial_coef} * {partial_loss.item():.4f}; \nSelf Pred acc: {batch_self_pred_acc:.4f}; Selected Self Pred acc: {batch_selected_self_pred_acc:.4f}; \nPartial Label set acc: {batch_partial_label_Y_acc:.4f}; Selected Partial Label set acc: {batch_selected_partial_label_Y_acc:.4f};Average Partial Label Num: {(partial_Y.sum()/partial_Y.shape[0]):.4f}\n"
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # ========================================================
        # ======================== Evaluation ====================
        # ========================================================
        test_flag = (iter_num < args.warmup_eval_iter_num) and (iter_num >= (args.warmup_eval_iter_num - 100))
        if (iter_num % interval_iter == 0) or (iter_num == max_iter) or test_flag:
            # log
            logger.info(batch_loss_str)
            model.eval()
            # General Evaluation
            acc_s_te, acc_list, f1 = cal_acc(dset_loaders["test"], model, args, True)
            log_str = f"Task: {args.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%, F1-Score = {f1:.2f}%\nAcc_list (cls, cls_acc):\n {acc_list}\n"

            args.out_file.write(
                f"\n============= Iter: {iter_num}, Evaluate Current Result ==============\n{log_str}\n==================================================================\n"
            )
            args.out_file.flush()
            if acc_s_te > best:
                best = acc_s_te
                best_log_str = log_str

            model.train()

            logger.info(f"[Test Info]\n{log_str}")
            test_num += 1
            sample_selection_R = evaluate_unlearning_bank(pure_score_bank, label_bank, partial_label_bank, sample_selection_mask_bank, sample_selection_R, uncertain_ratio_list, k_stars_list, k_values_list, args, logger)
            k_stars_list = []
            k_values_list = []
            uncertain_ratio_list = []



    best_acc_str = f"Adaptation Training End, src/tar:[{args.source}/{args.target}]\nbest acc is {str(best)}\n"
    print(f"best acc is {best}")
    print("*" * 30)

    logger.info(f"\n{best_acc_str}\nBest Log Info is: {best_log_str}\n")
    args.out_file.write(f"\n{best_acc_str}\nBest Log Info is: {best_log_str}\n")
    args.out_file.flush()



    # remove current file handler to avoid log file error
    logger.removeHandler(fh)
    # remove current stream handler to avoid IO info error
    logger.removeHandler(sh)

    return model, best, best_log_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCon-SFDA")
    parser.add_argument(
        "--config",
        type=str,
        help="dir od config file, None for not using, eg,EXPS/local/aad_lln_officehome.yml, prior to other args",
        default=None,
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="domainnet",
        choices=["VISDA-C", "office", "office-home", "domainnet"],
    )

    parser.add_argument(
        "--list_name",
        type=str,
        default="image_list", # choices=["image_list", "image_list_nrc" - for office home, "image_list_partial" - for partial set setting using shot source model],
    )
    parser.add_argument("--net", type=str, default="resnet50", help="resnet50, res101")
    parser.add_argument("--net_mode", type=str, choices=["fbc", "fc"], default="fbc")
    parser.add_argument("-s", "--source", default="c", help="source domain(s)")
    parser.add_argument("-t", "--target", default="s", help="target domain(s)")

    parser.add_argument("--max_epoch", type=int, default=40, help="max iterations")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers")

    parser.add_argument("--seed", type=int, default=2020, help="random seed")

    # DIR related
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="root path of DATASOURCE, eg, sfda_partial/DATASOURCE",
        default="./DATASOURCE",
    )
    parser.add_argument(
        "--model_root",
        help="model load and save root dir",
        default="./Models/shot_exp",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/expname"
    )
    parser.add_argument(
        "--s_model", help="source model suffix name, e.g., new", default=None
    )

    # arguments for logger
    parser.add_argument(
        "--log_dir", type=str, default="./logs/expname", help="log dir"
    )
    parser.add_argument("--expname", type=str, default="tarAdap_test")
    parser.add_argument("--key_info", type=str, default="autoHyperparam")

    # lr
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_F_coef", type=float, default=0.1, help="learning rate coef for backbone"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="optimiser weight decay"
    ) 
    parser.add_argument("--lr_decay", type=ast.literal_eval, default=True)
    parser.add_argument(
        "--lr_decay_type",
        type=str,
        choices=["shot"],
        default="shot",
    ) 
    parser.add_argument("--lr_power", type=float, default=0.75, help="learning rate power")


    parser.add_argument("--K", type=int, default=5, help="kappa")  
    parser.add_argument("--alpha", type=float, default=1.0, help="lambda_CL_neg") 
    parser.add_argument("--beta", type=float, default=5.0, help="beta") 
    parser.add_argument("--alpha_decay", type=ast.literal_eval, default=True)


    # data augmentation - dispersion control
    parser.add_argument(
        "--dc_coef_type", default='fixed', type=str, help="dispersion control coef type: fixed or init_incons (auto)"
    )
    parser.add_argument(
        "--dc_coef", default=1.0, type=float, help="dispersion control coef when dc_coef_type is fixed, lambda_dc"
    )
    parser.add_argument(
        "--dc_temp", default=1.0, type=float, help="dc loss term temperature"
    )
    parser.add_argument(
        "--dc_loss_type", default='ce', type=str, help="dc loss type, for ablation study"
    )
    
    
    # partial label hyper-params
    parser.add_argument(
        "--partial_coef", default=1.0, type=float, help="partial label coef, lambda_pl"
    )
    parser.add_argument(
        "--partial_k_type", default="fixed", type=str, help="fixed/cal"
    )
    parser.add_argument(
        "--partial_k", default=1, type=int, help="partial label number, k_pl"
    )
    
    parser.add_argument("--tau_type", default="fixed", type=str, help="fixed, stat, cal")
    parser.add_argument(
        "--sample_selection_R", default=10.0, type=float, help="sample selection ratio, tau"
    )

    # revision resource and training time, complexity analysis
    parser.add_argument(
        "--warmup_eval_iter_num",
        default=100,
        type=int,
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg_dir = osp.abspath(osp.join(osp.dirname(__file__), args.config))
        opt = vars(args)
        args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    args.log_dir = osp.abspath(osp.expanduser(args.log_dir))
    args.root = osp.abspath(osp.expanduser(args.root))
    args.model_root = osp.abspath(osp.expanduser(args.model_root))
    args.output_dir = osp.abspath(osp.expanduser(args.output_dir))

    if args.dset == "office-home":
        names = ["Ar", "Cl", "Pr", "Rw"]
        args.class_num = 65
    if args.dset == "office":
        names = ["amazon", "dslr", "webcam"]
        args.class_num = 31
    if args.dset == "VISDA-C":
        names = ["Tr", "Val"]
        args.class_num = 12
    if args.dset == "domainnet":
        names = ["r", "c", "p", "s"]
        args.class_num = 126

    
    # torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.key_info = (
        args.key_info + "_" + args.s_model
        if args.s_model is not None
        else args.key_info
    )
    
   
    train_target_all(args)
