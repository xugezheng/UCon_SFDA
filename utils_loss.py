import torch
import torch.nn as nn
import torch.nn.functional as F


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cross_entropy(targets, output, args, reduction='mean', smooth=0.1):
    if targets.size() != output.size():
        ones = torch.eye(args.class_num, device=output.device)
        targets_2d = torch.index_select(ones, dim=0, index=targets)
    else:
        targets_2d = targets
    
    pred = nn.Softmax(dim=1)(output)
    
    if smooth > 0:
        targets_2d = (1 - smooth) * targets_2d + smooth / args.class_num
    
    if reduction == 'none':
        return torch.sum(- targets_2d * torch.log(pred), 1)
    elif reduction == 'mean':
        return torch.mean(torch.sum(- targets_2d * torch.log(pred), 1))
    
def calculate_neg_var_loss(dot_neg_all, mask):
    dot_neg_mean = dot_neg_all.clone().detach().sum(-1) / (mask.sum(-1)+1) * mask
    neg_var_all = ((dot_neg_all - dot_neg_mean)**2).sum(-1) / (mask.sum(-1)+1)
    return torch.mean(neg_var_all)


def dc_loss_calculate(targets_logits, output_logits, args, reduction='mean', smooth=0.1, mask=None):
     
    if args.dc_loss_type == 'l2':
        aug_loss = F.mse_loss(output_logits, targets_logits.detach().clone(), reduction='mean')
        return aug_loss
    
    elif args.dc_loss_type == 'kl':
        targets = nn.Softmax(dim=-1)(targets_logits.detach().clone() / args.dc_temp)
        if smooth > 0:
            targets = (1 - smooth) * targets + smooth / args.class_num
        pred = nn.Softmax(dim=1)(output_logits)
        aug_loss = torch.mean((F.kl_div(pred, targets, reduction="none").sum(-1)))
        return aug_loss
    
    else:
        # default CE
        targets = nn.Softmax(dim=-1)(targets_logits.detach().clone() / args.dc_temp)
        if smooth > 0:
            targets = (1 - smooth) * targets + smooth / args.class_num
        
        pred = nn.Softmax(dim=1)(output_logits)
        aug_loss = torch.sum(- targets * torch.log(pred), 1)
        if reduction == 'mean':
            if mask is not None:
                return torch.sum(aug_loss * mask) / mask.sum()
            else:
                return torch.mean(aug_loss)

        else:
            return aug_loss
    
    