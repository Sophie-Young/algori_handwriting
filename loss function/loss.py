
nce
info_nce
pairwise_loss
triplet_loss
contrastive_loss

import torch

import torch.nn.functional as F

#meansquareerror loss
def MSE_loss(preds:torch.tensor,labels:torch.tensor):
    error=preds-labels
    return torch.mean(error**2)

#Cross-Entropy Loss
def CE_loss(preds:torch.tensor,labels:torch.tensor):
    """
    pred: [batch_size, num_classes]
    labels: [batch_size]
    """
    log_probs=F.log_softmax(preds,dim=-1)
    target_log_probs=log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

def BCE_loss(logits:torch.tensor,labels:torch.tensor):
    """
    pred: [batch_size, 1]
    labels: [batch_size]
    """
    log_probs_pos=F.logsigmoid(logits,dim=-1)
    # 负样本的 log 概率：log(1 - sigmoid(x)) = log(sigmoid(-x))
    log_probs_neg=F.logsigmoid(-logits,dim=-1)
    loss=-torch.mean(log_probs_pos*labels+log_probs_neg*(1.0-labels))
    return loss


"""
KL 散度在信息论中，代表的是**“用错误的代码本（Q）去编码真实的信息（P），所浪费的额外字节数”**。

如果你用最完美的、完全贴合真实分布 P 的代码本去编码，浪费的字节数是 0 只要你用的代码本 Q 不是最完美的，你必然会浪费额外的空间。你不可能因为用了一个错误的代码本，反而把文件压缩得比理论极限还要小！
所以，浪费的额外空间（KL 散度）永远不可能是一个负数，它最少也是 0。

kl散度反映的是在真实分布的视角下 预测分布与真实分布的差异

"""
def kl_divergence(logits_p:torch.tensor,logits_q:torch.tensor):
    """
    logits_p: [batch_size, num_classes] 参考模型的原始输出
    logits_q: [batch_size, num_classes] 当前预测的原始输出
    """
    p_probs=F.softmax(logits_p,dim=-1)
    p_log_probs=F.log_softmax(logits_p,dim=-1)
    q_log_probs=F.log_softmax(logits_q,dim=-1)

    kl_elements=p_probs*(p_log_probs*p_probs-q_log_probs*p_probs)

    return torch.sum(kl_elements,dim=-1).mean()


    

