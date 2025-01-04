import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def partial_label_loss(raw_outputs, partial_Y,  args, mask=None, smooth=-1, pos_coef=1.0, neg_coef=0.0):
    # refer to https://github.com/hongwei-wen/LW-loss-for-partial-label
    # LEVERAGED WEIGHTED LOSS FOR PARTIAL LABEL LEARNING
    sm_outputs = nn.Softmax(dim=1)(raw_outputs)
    onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1], device=args.device)
    onezero[partial_Y > 0] = 1 # selection of positive labels
    counter_onezero = 1 - onezero
   
    confidence = 1.0

    if smooth > 0:
        onezero = (1 - smooth) * onezero + smooth / args.class_num
    if mask is not None:
        selected_num = mask.sum()
        mask = mask.unsqueeze(1).expand_as(onezero).to(args.device)
    else:
        mask = torch.ones(sm_outputs.shape[0], sm_outputs.shape[1], device=args.device)
        selected_num = sm_outputs.shape[0]

    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = confidence * onezero * sig_loss1 * mask / onezero.sum(dim=-1, keepdim=True)
    average_loss1 = torch.sum(l1) / selected_num # l1.size(0)

    sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
    l2 = confidence * counter_onezero * sig_loss2 * mask / counter_onezero.sum(dim=-1, keepdim=True)
    average_loss2 = torch.sum(l2) / selected_num # l2.size(0)

    average_loss = pos_coef * average_loss1 + neg_coef * average_loss2
    return average_loss


# Rolling (cumulative) selection of data points involved in the partial label loss
def selection_mask_bank_update(sample_selection_mask_bank, tar_idx, outputs, args, ratio=10):
    if args.tau_type == 'fixed' or args.tau_type == 'stat':
        logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
        logits_ratio = logits[:,0] / logits[:, 1]
        selected_indx = torch.where(logits_ratio < ratio)[0].to(tar_idx.device)
        selected_indx_bank = tar_idx[selected_indx]
        sample_selection_mask_bank[selected_indx_bank] = 1
    
    elif args.tau_type == 'cal':
        logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
        logits_max = logits[:,0]
        logits_second = logits[:,1]
        logits_ratio = logits[:,0] / logits[:, 1]
        Num_classes= outputs.size()[1]
        eps2 = (1.0 / Num_classes)
        level_1 = 1.0 / Num_classes + eps2
        level_2 = logits_second + eps2
        selected_indx = torch.where(torch.logical_or(logits_max < level_1, logits_max < level_2))[0].to(tar_idx.device)
        selected_indx_bank = tar_idx[selected_indx]
        sample_selection_mask_bank[selected_indx_bank] = 1
    
    return sample_selection_mask_bank, logits_ratio[selected_indx]

#####################################################################
# Auxiliary functions for autoUCon-SFDA (hyperparam self-generation)
#####################################################################
def logits_ratio_calculation(outputs):
    logits, indx = torch.topk(outputs, k=2, dim=-1, largest=True, sorted=True)
    logits_ratio = logits[:,0] / logits[:, 1]
    return logits_ratio.detach().cpu().numpy()

# Cumulative record of partial label set for each data point
def partial_label_bank_update(partial_label_bank, tar_idx, outputs, k_values):
    if tar_idx.size() == outputs.size(): # given pred
        partial_label_bank[tar_idx, outputs] = 1
    elif isinstance(k_values, int):
        _, batch_label = torch.topk(outputs.clone().detach(), k=k_values, dim=-1, largest=True)
        tar_idx_reshaped = tar_idx.unsqueeze(1).expand_as(batch_label)
        partial_label_bank[tar_idx_reshaped, batch_label] = 1
    else:
        batch_size, num_classes = outputs.size()
        device = outputs.device

        # Get the indices within the k_values range for each sample in the batch
        sorted_indices = outputs.clone().detach().argsort(dim=-1, descending=True)

        # Build the mask tensor
        col_indices = torch.arange(num_classes, device=device).unsqueeze(0) # 1 * cls_num
        mask = (col_indices < k_values.view(-1, 1)).to(device) # bs * cls_num with top-K col true for each row
        tar_idx = tar_idx.to(device)
        selected_rows = tar_idx.unsqueeze(1).expand_as(mask)[mask] # only keep the True places of mask and return a 1d tensor, with the value of selected rows
        selected_cols = sorted_indices[mask] # only keep the True places of mask and return a 1d tensor, with the value of selected labels

        # Update partial_label_bank
        partial_label_bank[selected_rows, selected_cols] = 1
    
    return partial_label_bank

def calculate_k_values(outputs, k_max=2):
    batch_size, Num_classes = outputs.size()
    # eps1 = 2 / (Num_classes)
    eps1 = 0.1
    output_sorted, output_idx = torch.topk(outputs.detach().clone(), k=Num_classes, dim=1) # bs * cls_num
    sorted_accumu_1 = torch.cat([(torch.sum(output_sorted[:, :i+1], dim=1) - eps1) / (i+1) for i in range(Num_classes)]).reshape(Num_classes, batch_size).transpose(0, 1) # [(128, ), ..., (128, )] -> (10, 128) -> (128, 10)
    max_values, k_star = torch.max(sorted_accumu_1, dim=1) 
    k_star += 1 # skip 0
    k_values = torch.where(k_star < k_max, k_star, k_max)
    # print(k_star)
    # print(k_values)
    return k_star, k_values

def obtain_sample_R_ratio(args, init_logits_ratio):
    if args.tau_type == "fixed":
        return args.sample_selection_R
    elif args.tau_type == "stat":
        return np.percentile(init_logits_ratio, [10])[0]
    else:
        return args.sample_selection_R

