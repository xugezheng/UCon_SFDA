import numpy as np
import torch
from utils_loss import Entropy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
import logging
from collections import Counter
import h5py
import os.path as osp

logger = logging.getLogger("sfda")


def cal_acc(loader, model, args, flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_test = True
    re = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = 100.0 * torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    f1_ac = f1_score(all_label, torch.squeeze(
        predict).float(), average='macro')

    all_output_prob = nn.Softmax(dim=1)(all_output)
    all_fea_norm = F.normalize(all_fea)
    re["output"] = all_output.float().cpu().numpy()
    re["prob"] = all_output_prob.float().cpu().numpy()
    _, predict = torch.max(all_output, 1)
    re["pred"] = predict.float().cpu().numpy()
    re["acc"] = torch.sum(torch.squeeze(predict).float()
                          == all_label).item() / float(all_label.size()[0])
    re["label"] = all_label.cpu().numpy()
    re["fea_norm"] = all_fea_norm.cpu().numpy()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float()) # averaged recall?
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean() # This is not Accuracy!!!
    accuracy_used = aacc if args.dset == "VISDA-C" else accuracy
    aa = [f"({idx}, {str(np.round(i, 2))})" for idx, i in enumerate(acc)]
    acc = ' '.join(aa)
    
    return accuracy_used, acc, f1_ac * 100

def cal_acc_aug(loader, model, args, flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    start_test = True
    re = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs1, inputs2, inputs = data[0] # pl - weak strong train
            labels = data[1]
            inputs = inputs.to(device)
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            feas, outputs = model(inputs)
            # feas1, outputs1 = model(inputs1)
            feas1, outputs1 = model(inputs2)
            if start_test:
                all_output = outputs.detach().clone().float().cpu()
                all_fea = feas.detach().clone().float().cpu()
                all_label = labels.float()
                all_output1 = outputs1.detach().clone().float().cpu()
                all_fea1 = feas1.detach().clone().float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.detach().clone().float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.detach().clone().float().cpu()), 0)
                all_output1 = torch.cat((all_output1, outputs1.detach().clone().float().cpu()), 0)
                all_fea1 = torch.cat((all_fea1, feas1.detach().clone().float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        _, predict = torch.max(all_output, 1)
        _, predict1 = torch.max(all_output1, 1)
        accuracy = 100.0 * torch.sum(torch.squeeze(predict).float() ==
                            all_label).item() / float(all_label.size()[0])
        accuracy1 = 100.0 * torch.sum(torch.squeeze(predict1).float() ==
                            all_label).item() / float(all_label.size()[0])
        f1_ac = f1_score(all_label, torch.squeeze(
            predict).float(), average='macro')
        f1_ac1 = f1_score(all_label, torch.squeeze(
            predict1).float(), average='macro')
        consistency = 100.0 * torch.sum(torch.squeeze(predict).float() == torch.squeeze(predict1).float()).item() / float(predict1.size()[0])

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float()) # averaged recall?
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean() # This is not Accuracy!!!
        accuracy_used = aacc if args.dset == "VISDA-C" else accuracy
        aa = [f"({idx}, {str(np.round(i, 2))})" for idx, i in enumerate(acc)]
        acc = ' '.join(aa)
        
        matrix1 = confusion_matrix(all_label, torch.squeeze(predict1).float()) # averaged recall?
        acc1 = matrix1.diagonal() / matrix1.sum(axis=1) * 100
        aacc1 = acc1.mean() # This is not Accuracy!!!
        accuracy_used1 = aacc1 if args.dset == "VISDA-C" else accuracy1
        aa1 = [f"({idx}, {str(np.round(i, 2))})" for idx, i in enumerate(acc1)]
        acc1 = ' '.join(aa1)
        
    return accuracy_used, acc, f1_ac * 100, accuracy_used1, acc1, f1_ac1 * 100, consistency




#######################################################################################################################
##############################################  uncertainty  ################################################
#######################################################################################################################
def topk_acc(re, k):
    prob = re["prob"]
    true_label = re["label"]
    topk_idx, topk_logits = topk_2d_np(prob, k)
    true_label_2d = np.expand_dims(true_label, axis=-1)
    is_in_topk = np.any(true_label_2d == topk_idx, axis=-1)
    return is_in_topk.mean()
    
    
def topk_2d_np(arr, k, dim=1):
    # Ensure k is a valid number
    if k <= 0:
        return np.array([])
    if k > arr.shape[dim]:
        k = arr.shape[dim]
    
    # Sort along the specified dimension and get the indices
    sorted_indices = np.argsort(arr, axis=dim)[:, ::-1]
    
    # Select the top k elements
    if dim == 1:
        # Row-wise top k elements
        topk_indices = sorted_indices[:, :k]
        rows = np.arange(arr.shape[0])[:, None]
        return topk_indices, arr[rows, topk_indices]
    else:
        raise ValueError("Currently, only dim=1 is supported.")


def topk_acc_torch(prob, true_label, k):
    topk_logits, topk_idx = torch.topk(prob, k = k, dim=-1, largest=True)
    true_label_2d = torch.unsqueeze(true_label,dim=1)
    is_in_topk = torch.any(true_label_2d == topk_idx, dim=-1)
    return is_in_topk.float().mean()

def partial_Y_evaluation(partial_Y, true_label):
    partial_Y = partial_Y.cpu().numpy()
    one_hot_true_label = np.zeros_like(partial_Y)
    true_label = true_label.cpu().numpy()
    rows = np.arange(partial_Y.shape[0])
    one_hot_true_label[rows,true_label] = 1000
    re = (partial_Y * one_hot_true_label).sum(-1)
    acc = len(np.where(re >= 1000)[0]) / len(re)
    return acc
    
    
def evaluate_unlearning_bank(score_bank_org, label_bank_org, partial_label_bank_org, selection_bank_org, select_r, uncertain_ratio_list, k_stars_list, k_values_list, args, logger, out=False):

    score_bank = score_bank_org.detach().clone().cpu()
    label_bank = label_bank_org.detach().clone().cpu()
    test_partial_label_bank = partial_label_bank_org.detach().clone().cpu()
    selection_bank = selection_bank_org.detach().clone().cpu()
    # test overall and selected (first prob / second prob > ratio) acc, per-class acc 
    pred_label = score_bank.max(-1)[1].float()
    true_label = label_bank.float()
    true_label_2d = torch.unsqueeze(true_label,dim=1)
    overall_acc = 100.0 * torch.sum(pred_label == true_label).item() / float(pred_label.size()[0])


    # tau
    logits, indx = torch.topk(score_bank, k=2, dim=-1, largest=True, sorted=True)
    ratio = logits[:,0] / logits[:, 1]
    if args.tau_type == "fixed":
        new_select_r = select_r
        return_r = select_r
    elif args.tau_type == "stat":
        return_r = select_r
        new_select_r = np.percentile(ratio, [10])[0]
    elif args.tau_type == "cal":
        if not uncertain_ratio_list:
            print("Empty uncertain_ratio_list")
            new_select_r = select_r
            return_r = select_r
        else:
            concatenated_tensor = torch.cat(uncertain_ratio_list)
            if concatenated_tensor.numel() == 0:
                print("The concatenated tensor is empty. Cannot compute max value.")
                new_select_r = select_r
                return_r = select_r
            else:
                new_select_r = torch.max(concatenated_tensor).item()
                return_r = new_select_r
    uncertain_all = torch.cat(uncertain_ratio_list)

        
    # partial label cur epoch
    if  k_values_list:
        all_k_values = torch.cat(k_values_list)
        all_k_stars = torch.cat(k_stars_list)
    else:
        all_k_values = torch.tensor([args.partial_k])
        all_k_stars = torch.tensor([args.partial_k])
    
    # partial label bank
    unused_indx = torch.where(selection_bank == 0)[0]
    used_indx = torch.where(selection_bank == 1)[0]
    if unused_indx.size()[0] == 0:
        unused_sample_set_acc = 0.0
        unused_partialY_acc = 0.0
        unused_partialY_labellen = 0.0
    else:
        unused_pred_label = pred_label[unused_indx]
        unused_true_label = true_label[unused_indx]
        unused_sample_set_acc = 100.0 * torch.sum(unused_pred_label == unused_true_label).item() / float(unused_indx.size()[0]) 
        unused_partialY_acc = partial_Y_evaluation(test_partial_label_bank[unused_indx], unused_true_label.long())
        unused_partialY_labellen = test_partial_label_bank[unused_indx].sum() / unused_indx.size()[0]
    
    if used_indx.size()[0] == 0:
        used_sample_set_acc = 0.0
        used_partialY_acc = 0.0
        used_partialY_labellen = 0.0
    else:
        used_pred_label = pred_label[used_indx]
        used_true_label = true_label[used_indx]
        used_sample_set_acc = 100.0 * torch.sum(used_pred_label == used_true_label).item() / float(used_indx.size()[0])
        used_partialY_acc = partial_Y_evaluation(test_partial_label_bank[used_indx], used_true_label.long())
        used_partialY_labellen = test_partial_label_bank[used_indx].sum() / used_indx.size()[0]
        overall_partialY_labellen = test_partial_label_bank.sum() / test_partial_label_bank.size()[0]
        overall_partial_label_bank_acc = partial_Y_evaluation(test_partial_label_bank, true_label.long())
    
    partial_bank_log_str = '\n[Sample Selection Bank Info]\n'
    partial_bank_log_str += f"[DATASET INFO] Dataset len: {float(pred_label.size()[0])}\n[tau INFO] Selection Method: {args.tau_type} | [ratio change]: {select_r:.4f} -> {new_select_r:.4f} | [return ratio]: {return_r:.4f}\n"
    partial_bank_log_str += f"[CurrentEpoch Selection] ratio > {select_r:.4f} len: {pred_label.size()[0] - uncertain_all.size()[0]} | ratio < {select_r:.4f} len: {uncertain_all.size()[0]}; \n"
    partial_bank_log_str += f"[Accumulation Selection] unselected len: {pred_label.size()[0] - selection_bank.sum()} | selected len: {selection_bank.sum()}; \n"
    
    partial_bank_log_str += '[Partial Label Performance Info]\n'
    partial_bank_log_str += f"[CUR] | k stars len: {all_k_stars.size()[0]}, k values len: {all_k_values.size()[0]}\n"
    partial_bank_log_str += f"[CUR] | k stars avg: {torch.mean(all_k_stars.float())}, k values avg: {torch.mean(all_k_values.float())}\n"
    partial_bank_log_str += f"[BANK] | [Overall] ALL - self acc: {overall_acc:.2f} | partial label bank acc: {overall_partial_label_bank_acc:.4f} | partial set avg len: {overall_partialY_labellen:.2f}\n"
    partial_bank_log_str += f"[BANK] | [Accumulation] NOT used - self pred acc: {unused_sample_set_acc:.2f} | partial set acc: {unused_partialY_acc:.4f} | partial set avg len: {unused_partialY_labellen:.2f}\n"
    partial_bank_log_str += f"[BANK] | [Accumulation] USED - self pred acc: {used_sample_set_acc:.2f} | partial set acc: {used_partialY_acc:.4f} | partial set avg len: {used_partialY_labellen:.2f}\n"
    logger.info(partial_bank_log_str)
    
    if out == True:
        logger.info(f"Return Ratio: {return_r:.4f}\nAveraged K stars: {torch.mean(all_k_stars.float())}.")
        return return_r, torch.mean(all_k_stars.float())
    return return_r
    
def build_neighbor_partial_bank(feature_bank_org, score_bank_org, num_sample, args):
    feature_bank = feature_bank_org.clone().detach()
    score_bank = score_bank_org.detach().clone()
    neighbor_partial_label_bank = torch.zeros(num_sample, args.class_num).long().to(args.device)
    
    distance = feature_bank @ feature_bank.T
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
    idx_near = idx_near[:, 1:]  # batch x K
    score_near = score_bank[idx_near] 
    _, label_near = score_near.max(-1)
    dataset_idx = torch.arange(num_sample).to(args.device)
    dataset_idx_ = dataset_idx.unsqueeze(1).expand_as(label_near)
    neighbor_partial_label_bank[dataset_idx_.to(neighbor_partial_label_bank.device), label_near.to(neighbor_partial_label_bank.device)] = 1
    return neighbor_partial_label_bank
    

def calculate_false_neg_sample(tar_id):
    unique_elements, counts = torch.unique(tar_id, return_counts=True)
    count_in_others = torch.zeros_like(tar_id)

    # Iterate through each element in the tensor
    for i in range(tar_id.size()[0]):
        # counts the occurrences of current element
        element_count = counts[unique_elements == tar_id[i]].item()
        # subtracts 1 from the count for each occurrence of the element at other positions
        count_in_others[i] = element_count - 1
    return count_in_others.cpu().numpy()


if __name__ == '__main__':
    partial_Y = torch.tensor([[1,0,0,0,1], [0,1,0,0,0], [0,1,0,0,0]])
    true_label = torch.tensor([1, 3, 0])
    acc = partial_Y_evaluation(partial_Y, true_label)
    print(acc)
    
    
