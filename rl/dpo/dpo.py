import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_log_prob(logits:torch.tensor,labels:torch.tensor,mask:torch.tensor):
    """
    logits: [batch_size, seq_len, vocab_size]
    labels: [batch_size, seq_len]
    mask: [batch_size, seq_len]
    计算序列的对数概率
    只计算response的对数概率
    
    """
    #获取每个token的对数概率
    log_probs=F.log_softmax(logits,dim=-1)

    #获取对应的label的log概率
    per_token_log_probs=log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)

    #计算response的对数概率(mask为1的位置)
    response_log_probs=per_token_log_probs*mask
    #求和得到整个序列的log概率
    return response_log_probs.sum(dim=-1)


def dpo_los(policy_chosen_logits:torch.tensor,policy_rejected_logits:torch.tensor,ref_chosen_logits:torch.tensor,ref_rejected_logits:torch.tensor,chosen_labels:torch.tensor,rejected_labels:torch.tensor,chosen_mask:torch.tensor,rejected_mask:torch.tensor,beta:float=0.1):
    """
    计算 DPO 损失
    """
    #计算策略模型的对数概率
    policy_chosen_log_probs=compute_log_prob(policy_chosen_logits,chosen_labels,chosen_mask)
    policy_rejected_log_probs=compute_log_prob(policy_rejected_logits,rejected_labels,rejected_mask)

    #计算参考模型的对数概率
    ref_chosen_log_probs=compute_log_prob(ref_chosen_logits,chosen_labels,chosen_mask)
    ref_rejected_log_probs=compute_log_prob(ref_rejected_logits,rejected_labels,rejected_mask)


    #计算概率的对数比值
    chosen_log_ratios=policy_chosen_log_probs-ref_chosen_log_probs
    rejected_log_ratios=policy_rejected_log_probs-ref_rejected_log_probs

    #计算差值信号
    delta_log_ratios=chosen_log_ratios-rejected_log_ratios

    losses=-F.logsigmoid(delta_log_ratios*beta)

    return losses.mean()
    
