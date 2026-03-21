import torch
import torch.nn as nn
from thop import profile

#虽然在自注意力（Self-Attention）中，倒数第一和倒数第二维度的长度都是 seq_len，但它们的物理含义完全不同。
"""
倒数第二个维度 (seq_len_Q)：代表 Query（查询） 的序列长度。也就是“当前正在思考的 Token”。
最后一个维度 (seq_len_K)：代表 Key（键） 的序列长度。也就是“上下文中所有可供参考的 Token”。
为什么要在 dim=-1 (也就是 seq_len_K) 上做 Softmax？
注意力机制的核心逻辑是：“对于当前的这 1 个 Query，我应该把多少注意力分配给前面的 N 个 Key？”
既然是分配注意力，那么这 N 个 Key 的注意力权重加起来必须等于 1。
因此，我们必须沿着 Key 的维度（最后一个维度）进行 Softmax，把分数转化为概率分布。
"""

"""
Attention Dropout 的作用（加在 attn_weights 上）：
attn_weights 是一个概率矩阵，表示 Token 之间的关注度。在这里加 Dropout，意味着随机强行切断某些 Token 之间的联系（把某些注意力权重变成 0）。
直观理解：强迫模型不要总是死死盯着某一个特定的词（比如不要每次看到“苹果”就只关注“手机”），而是去多看看句子里的其他词。这能极大地提升模型对上下文理解的泛化能力。
Hidden/Residual Dropout 的作用（通常加在 out_proj 之后）：
如果在 out_proj 之后加 Dropout，那是为了随机丢弃某些神经元（特征维度），防止神经元之间产生共适应（Co-adaptation），这是最传统的防止全连接层过拟合的手段。
"""


class MultiQueryAttention(nn.Module):
    de __init__(self,hidden_size,num_heads,dropout=0.0):
        super(MultiQueryAttention,self).__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        self.dropout=dropout

        self.q_proj=nn.Linear(hidden_size,hidden_size)
        self.k_proj=nn.Linear(hidden_size,head_dim)
        self.v_proj=nn.Linear(hidden_size,head_dim)

        self.dropout=nn.Dropout(dropout)
        self.out_proj=nn.Linear(hidden_size,hidden_size)

    def forward(self,x,attn_mask=None):
        batch_size,seq_len=x.size()[:2]
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        #只需要给query分head
        q=q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        #为了矩阵计算 让k v 扩展到相同维度（广播）
        k=k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v=k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn_weights=torch.matmul(q,k.transpose(2,-3))/self.head_dim**0.5

        if attn_mask is not None:
            attn_weights=attn_weights+attn_mask
        attn_weights=torch.softmax(attn_weights,dim=-1)

        attn_weights=self.dropout(attn_weights)

        x=torch.matmul(attn_weights,v)
        x=x.transpose(1,2).contiguous().view(batch_size,seq_len,self.hidden_size)
        
        x=self.out_proj(x)
        x=x.dropout(self.dropout)
        return x

if __name__=='__main__':
    batch_size=4
    seq_len=16
    hidden_dim=128
    num_heads=8

    x=torch.randn(batch_size,seq_len,hidden_dim)

    attn_mask=torch.ones(batch_size,seq_len)
    attn_mask[:,5:]=0

    mqa=MultiQueryAttention(hidden_dim,num_heads)

    x=mqa(x,attn_mask)




#deepseek-v1
class GroupedQueryAttention(nn.Module):
    def __init__(self,hidden_size,group_size,num_heads,dropout=0.0):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads

        #分组大小
        self.group_size=group_size
        self.num_groups=num_heads//group_size
        self.dropout=dropout

        #Q的投影依然是完整的hidden_size
        self.q_proj=nn.Linear(hidden_size,hidden_size)

        #K和V的投影是分组的 只有num_groups*head_dim
        self.k_proj=nn.Linear(hidden_size,self.num_groups*self.head_dim)
        self.v_proj=nn.Linear(hidden_size,self.num_groups*self.head_dim)

        self.dropout=nn.Dropout(dropout)
        self.out_proj=nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,attn_mask=None):
        batch_size,seq_len=x.size()[:2]

        #q投影并作矩阵乘法准备
        q=self.q_proj(x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        #kv投影并作矩阵乘法准备
        k=self.k_proj(x).view(batch_size,seq_len,self.num_groups,self.head_dim).transpose(1,2)
        k=k.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).contiguous().view(batch_size,-1,seq_len,self.head_dim)
        v=self.v_proj(x).view(batch_size,seq_len,self.num_groups,self.head_dim).transpose(1,2)
        v=v.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).contiguous().view(batch_size, -1, seq_len, self.head_dim) 


        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 计算上下文向量
        context = torch.matmul(attention_weights, value)

        # 5. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 6. 输出投影
        output = self.out_projection(context)
        return output

 
#deepseek-v2  MLA配合RoPE
"""
#记忆口诀
cos sin扩充一倍
q切开 后一半变负 前一半不变
套用公式 原q乘cos 负q乘sin
"""
class RotaryPositionEmbedding(nn.Module):
    def __init__(self,max_seq_len,hidden_size,num_heads,base=10000):
        """
        RoPE位置编码模块

        Args:
            hidden_size (int): 模型维度
            num_heads (int): 注意力头数量
            base (int): 频率基值
            max_seq_len (int): 最大序列长度
        """     
        super().__init__()
        self.max_seq_len=max_seq_len
        self.hidden_size=hidden_size
        self.base=base
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        cos_cached,sin_cached=self.compute_pos_emb()

        # 必须注册为 buffer，这样它们会自动跟随模型移动到 GPU/CPU，且不参与梯度更新
        self.register_buffer("cos_cached", cos_emb, persistent=False)
        self.register_buffer("sin_cached", sin_emb, persistent=False)        

    def compute_pos_emb(self):
        # theta_i 形状: [head_dim / 2]
        theta_i=1.0/(self.base**(torch.arange(0,self.head_dim,2).float()/self.head_dim))
        # positions 形状: [max_seq_len]
        positions=torch.arange(self.max_seq_len)
        # positions 形状: [max_seq_len]
        pos_emb=positions.unsqueeze(1)*theta_i.unsqueeze(0)
        cos_emb=torch.cos(pos_emb)
        sin_emb=torch.sin(pos_emb)
        return cos_emb,sin_emb
    
    def forward(self,q):
        """
         RoPE位置编码应用

        Args:
            q (torch.Tensor): 输入张量 [bs, num_heads, seq_len, head_dim]

        Returns:
            torch.Tensor: 应用位置编码后的张量       
        """
        #旋转位置编码是应用在做过投影以后的query和key上
        batch_size,num_heads,seq_len,head_dim=q.size()
        cos_cached=self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_cached=self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # 因为 cos 和 sin 的最后一维是 head_dim/2，我们需要把它们复制一份变成 head_dim
        # cos 形状变成: [1, 1, seq_len, head_dim]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
     
        # 把 q 劈成两半
        q1 = q[..., :head_dim // 2]
        q2 = q[..., head_dim // 2:]

        q_half_rotated=torch.cat([-q2,q1],dim=-1)

        #应用rope公式 
        return q*cos+q_half_rotated*sin



class MultiLevelAttention(nn.Module):
    def __init__(self,hidden_size=256,down_dim=64,up_dim=128,num_heads=8,rope_head_dim=64,dropout=0.0):
        """
        Multi-Head Latent Attention 实现

        Args:
            hidden_size (int): 输入特征维度
            down_dim (int): 降维后的维度
            up_dim (int): 升维后的维度
            num_heads (int): 注意力头数量
            rope_head_dim (int): RoPE编码的头维度
            dropout (float): Dropout概率
        """
        super().__init__()
        self.d_model=hidden_size
        self.down_dim=down_dim 
        self.up_dim=up_dim  #内容总维度
        self.num_heads=num_heads

        self.c_head_dim=up_dim//num_heads #每个头的内容维度
        self.rope_head_dim=rope_head_dim #每个头的位置编码维度
        self.v_head_dim = self.c_head_dim      # V 的维度通常和 K 的内容维度一致


        #kv降维投影（核心：压缩）
        self.down_kv_proj=nn.Linear(hidden_size,down_dim)
        #q降维投影
        self.down_q_proj=nn.Linear(hidden_size,down_dim)

        #升维投影（从隐变量中恢复出内容 qkv）
        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)

        # 解耦的rope投影
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        # 注意：K 的 RoPE 向量直接从原 hidden_size 投影，且所有头共享！(类似 MQA)
        # 既然 Ckv这么宝贵，那就让它100% 纯粹地只负责语义内容。
        self.proj_kr = nn.Linear(hidden_size, rope_head_dim) 

        self.rope_q=RotaryPositionEmbedding(max_seq_len,num_heads,rope_head_dim*num_heads)
        self.rope_k=RotaryPositionEmbedding(max_seq_len,rope_head_dim,1)

        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(num_heads*self.v_head_dim,hidden_size)
        self.residual_dropout=nn.Dropout(dropout)

    def forward(self,h,attn_mask=None):
        """
        前向传播

        Args:
            h (torch.Tensor): 输入张量 [bs, seq_len, d_model]
            mask (torch.Tensor): 注意力掩码 [bs, seq_len]

        Returns:
            torch.Tensor: 输出张量 [bs, seq_len, d_model]
        """
        bs, seq_len, _ = h.size()
        # ================= Step 1: 低秩压缩与内容恢复 =================    
        c2kv=self.down_kv_proj(h)
        c2q=self.down_q_proj(h)
        q2c=self.up_proj_q(c2q)
        k2c=self.up_proj_k(c2kv)
        v2c=self.up_proj_v(c2kv)


        qr=self.proj_qr(c2q)
        qr=rope_q.view(bs,seq_len,self.num_heads,self.rope_head_dim).transpose(1,2)#[bs,num_heads,seq_len,rope_head_dim]
        qr=self.rope_q(qr)

        kr=self.proj_kr(h) #[bs,seq_len,rope_head_dim]
        kr=kr.unsqueeze(1) #[bs,1,seq_len,rope_head_dim]
        kr=self.rope_k(kr)

        q2c=q2c.view(bs,seq_len,self.num_heads,self.c_head_dim).transpose(1,2)#[bs,num_heads,seq_len,c_head_dim]
        q2c=torch.cat([q2c,qr],dim=-1)
        k2c=k2c.view(bs,seq_len,self.num_heads,self.c_head_dim).transpose(1,2)#[bs,num_heads,seq_len,c_head_dim]
        kr=kr.expand(-1, self.num_heads, -1, -1)
        k2c=torch.cat([k2c,kr],dim=-1)

        attn_weights= torch.matmul(q, k.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
        attn_weights = attn_weights / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim))        
        
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))  # [bs, num_heads, seq_len, seq_len]

        attn_weights = torch.softmax(scores, dim=-1)  # [bs, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        
        # V维度调整
        v2c = v2c.view(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)  # [bs, num_heads, seq_len, v_head_dim]
        context = torch.matmul(attn_weights, v2c)
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.hidden_size)
        output = self.fc(context)  
        output  = self.residual_dropout(output )
        return output 