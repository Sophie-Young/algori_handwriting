import torch
import numpy as np
import torch.nn as nn

def get_rotary_frequencies(dim:int,seq_len:int,theta:float=10000.0):
    """
        生成旋转频率
        args:
            dim: 维度
            seq_len: 序列长度
            theta: 旋转基础频率
        returns:
            frequencies: 旋转频率参数
    """
    for i in range(dim//2):
        freqs[i]=theta**(-2*i/dim)
    
    positions=torch.arange(seq_len,dtype=torch.float32) #shape: [seq_len]

    angles=torch.outer(positions,freqs) # shape: [seq_len,dim//2]

    return angles


def get_rotary_embedding(dim:int,seq_len:int,theta:float=10000.0):
    """
        预计算RoPE的sin和cos的值
        return：
    """
    angles=get_rotary_frequencies(dim,seq_len,theta)
    cos_angles=torh.cos(angles)
    sin_angles=torh.sin(angles)
    
    cos_angles=torch.cat([cos_angles,cos_angles],dim=-1)
    sin_angles=torch.cat([sin_angles,sin_angles],dim=-1)

    return cos_angles,sin_angles


def rotate_half(x:torch.Tensor):
    """
        将x的后一半维度与前一半维度交换 还要把后一半维度变- 这是实现旋转的关键操作
        [x1,x2,x3,x4]->[-x3,-x4,x1,x2]
    """

    x1=x[...,x.shape[-1]//2:]
    x2=x[...,:x.shape[-1]//2]

    return troch.cat([-x1,x2],dim=-1)
    
def apply_rotatry_pos_emd(q:torch.Tensor,k:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor):
    """
        应用旋转位置编码
        args:
            q: query tensor [batch_size,seq_len,num_heads,head_dim]
            k: key tensor [batch_size,seq_len,num_heads,head_dim]
            cos: cos tensor [seq_len,head_dim]
            sin: sin tensor [seq_len,head_dim]
        returns:
            q: query tensor [batch_size,seq_len,num_heads,head_dim]
            k: key tensor [batch_size,seq_len,num_heads,head_dim]
    """
    #调整cos和sin的维度 方便广播
    cos=cos.unsqueeze(0).unsqueeze(2)
    sin=sin.unsqueeze(0).unsqueeze(2)

    #应用旋转编码
    q_embed=q*cos+rotate_half(q)*sin
    k_embed=k*cos+rotate_half(k)*sin

    return q_embed,k_embed

class RotaryPositionEmbedding(nn.Module):
    def __init__(self,max_seq_len:int,dim:int,theta:float=10000.0):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.dim=dim
        self.theta=theta

        cos,sin=get_rotary_embedding(dim,max_seq_len,theta)
        #register_buffer是PyTorch中的一个重要方法,用于注册不需要梯度更新的模型参数
        #缓存的cos/sin会被保存到模型的state_dict中，确保模型加载后可以直接使用，无需重新计算。
        self.register_buffer("cos_cached",cos)
        self.register_buffer("sin_cached",sin)

    def forward(self,q:torch.Tensor,k:torch.Tensor):
        seq_len=q.size(1)

        cos_cached=self.cos_cached[:seq_len]
        sin_cached=self.sin_cached[:seq_len]
        return apply_rotatry_pos_emd(q,k,cos_cached,sin_cached)





