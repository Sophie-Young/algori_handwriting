pairwise_loss
triplet_loss
focal loss
batchnorm
layernorm
dropout
ffn


import torch
import torch.nn.functional as F

#meansquareerror loss
def MSE_loss(preds:torch.Tensor,labels:torch.Tensor):
    error=preds-labels
    return torch.mean(error**2)

#Cross-Entropy Loss
def CE_loss(preds:torch.Tensor,labels:torch.Tensor):
    """
    pred: [batch_size, num_classes]
    labels: [batch_size]
    """
    log_probs=F.log_softmax(preds,dim=-1)
    target_log_probs=log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

def BCE_loss(logits:torch.Tensor,labels:torch.Tensor):
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
def kl_divergence(logits_p:torch.Tensor,logits_q:torch.Tensor):
    """
    logits_p: [batch_size, num_classes] 参考模型的原始输出
    logits_q: [batch_size, num_classes] 当前预测的原始输出
    """
    p_probs=F.softmax(logits_p,dim=-1)
    p_log_probs=F.log_softmax(logits_p,dim=-1)
    q_log_probs=F.log_softmax(logits_q,dim=-1)

    kl_elements=p_probs*(p_log_probs*p_probs-q_log_probs*p_probs)

    return torch.sum(kl_elements,dim=-1).mean()


#nce loss 是-（正样本对数概率+负样本对数概率）的平均 nce损失是标准的二元交叉熵
#noise contrastive estimation loss    
def NCE_loss(pos_scores:torch.Tensor,neg_scores:torch.Tensor):
    """
    手写NCE loss
    pos_scores: [batch_size, 1] 正样本得分
    neg_scores: [batch_size, k] 负样本得分
    """
    log_pos_probs=F.logsigmoid(pos_scores);
    log_neg_probs=F.logsigmoid(-neg_scores)
    
    sum_log_probs_neg=torch.sum(log_neg_probs,dim=-1).squeeze(-1)

    loss=-(log_pos_probs.squeeze(-1)+sum_log_probs_neg).mean()
    
    return loss

#infonce抛弃了nce的二元分类的思想，把它变成一个多分类的问题 模型需要在k+1个选项中，精准选出正样本
#工业实现通常使用batch内负样本
def info_NCE_loss(queries:torch.Tensor,keys:torch.Tensor,temperatue:float=0.05):
    """
    手写infonceloss
    queries ： 【batch_size, hidden——dim】 搜索词的向量
    keys： 【batch_size, hidden_dim】 候选词的向量
    """

    #l2归一化
    queries=F.normalize(queries,p=2,dim=-1)
    keys=F.normalize(keys,p=2,dim=-1)

    #计算相似度矩阵
    #形状：[batch_size, batch_size]
    logits=torch.matmul(queries,keys.transpose(0,1))
    logits=logits/temperatue

    #提取正样本得分
    pos_scores=logits.diag()

    #infonceloss的思想就是多分类交叉熵

    #计算分母logsumexp
    #找到每行最大值
    max_logits,_=torch.max(logits,dim=-1,keepdim=True)

    #减去最大值
    safe_exp=torch.exp(logits-max_logits)

    #沿着列求和
    safe_sum_exp=torch.sum(safe_exp,dim=-1)

    #取对数
    log_sum_exp=torch.log(safe_sum_exp)+max_logits.squeeze(-1)

    #计算最后的loss
    loss_per_sample=lod_sum_exp-pos_scores

    return loss_per_sample.mean()