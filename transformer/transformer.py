import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict,Optional
import numpy as np
from position_encoding.RoPE import RotaryPositionEmbedding

def get_sinusoid_encoding_table(max_length,d_model):
    """
    generate position encoding
    args：  
        max_length：max sequence length
        d_model：model dimension
    returns：
        position_enc：[max_length，d_model] position encoding
    """

    def cal_angle(pos,hid_idx):
        return pos/np.power(10000,2*(hid_idx//2)/d_model)

    def get_pos_angle_vec(pos):
        return [cal_angle(pos,hid_j) for hid_j in range(d_model)]
    
    sinusoid_table=np.array([get_pos_angle_vec(pos_i) for pos_i in range(max_length)])
    sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])

    return torch.FloatTensor(sinusoid_table)
    

class MultiHeadAttention(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()

        # set numheads and dimension of each head
        self.n_heads=args.n_heads
        self.head_dim=args.head_dim

        self.dropout=args.dropout

        # define q k v converting matrix
        self.query_proj=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)
        self.key_proj=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)
        self.value_proj=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)
        self.output_proj=nn.Linear(self.head_dim*self.n_heads,args.dim,bias=False)

        #define dropout layer
        #attention_dropout
        self.attn_dropout=nn.Dropout(self.dropout)

        #residual dropout
        self.residual_dropout=nn.Dropout(args.dropout)


        #kv cache
        self.key_cache,self.value_cache=None,None

        max_len=args.max_len
        #d 1 batch_size
        #d 2 head
        #d 3 q_len
        #d 4 k_len
        attn_mask=torch.full((1,1,max_len,max_len),float("-inf"))

        # torch.triu(xxx, diagonal=1) 上三角矩阵, diagonal=1表示从对角线开始,屏蔽对角线及其右边的元素
        # register_buffer("name", tensor, persistent=False) - 注册一个不需要梯度的张量作为模块的缓冲区
        # - 与普通的Parameter不同,buffer不会被优化器更新
        # - 但会被保存到模型的state_dict中,在加载模型时可以恢复
        # - 使用persistent=False表示这个buffer在保存模型时不会被保存
        # - 因为掩码可以在加载模型时重新生成
        # 原始论文中的编码器不需要使用掩码矩阵来屏蔽未来的信息，因为编码器处理的是整个输入序列，每个位置的token可以自由地访问序列中的其他位置。
        # 这里为了方便，就统一使用掩码矩阵来屏蔽未来的信息
        self.register_buffer("attn_mask",torch.triu(attn_mask,diagonal=1),persistent=False)

    def forward(self,x:torch.tensor,encoder_output: Optional[torch.Tensor] = None):
        """前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, dim]
            encoder_output: 编码器输出，用于交叉注意力。
                          如果为None则为自注意力模式
        """
        batch_size,seq_len,_=x.shape
        #query 始终来自输入x
        query=self.query_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)

        if encoder_output is None:
            key=self.key_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
            value=self.value_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
        else : #交叉注意力
            key=self.key_proj(encoder_output).view(batch_size,-1,self.n_heads,self.head_dim)
            value=self.value_proj(encoder_output).view(batch_size,-1,self.n_heads,self.head_dim)

        # dimension transpose
        #原本的维度是 [batch_size,seq_len,self.n_heads,self.head_dim]
        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)

        # attention calculation
        scale=1.0/math.sqrt(self.head_dim)
        attn=torch.matmul(query,key.transpose(2,3))*scale

        # 自注意力下要使用注意力掩码
        if encoder_output is None:
            attn=attn+self.attn_mask[:,:,:seq_len,:seq_len]
            
        attn_probs=F.softmax(attn.float(),dim=-1).type_as(query)
        attn_probs=self.attn_dropout(attn_probs)

        output=torch.matmul(attn_probs,value)
        output=output.transpose(1,2).contiguous().view(batch_size,seq_len,-1)

        return self.residual_dropout(self.output_proj(output))


#交叉注意力 的 x 和 encoder_output不在同一个坐标系下 因此不应该使用位置编码
class MultiHeadAttentionwithRoPE(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()

        self.n_heads=args.n_heads
        self.head_dim=args.head_dim

        self.dropout=args.dropout
        self.q_proj=nn.Linear(args.dim,args.head_dim*args.n_heads,bias=False)
        self.k_proj=nn.Linear(args.dim,args.head_dim*args.n_heads,bias=False)
        self.v_proj=nn.Linear(args.dim,args.head_dim*args.n_heads,bias=False)
        self.output_proj=nn.Linear(args.head_dim*args.n_heads,args.dim,bias=False)
        self.register_buffer("rope_cache",RotaryPositionEmbedding(args.max_len,args.dim))
        
        attn_mask=torch.full((1,1,args.max_len,args.max_len),float("-inf"))
        self.register_buffer("attn_mask",torch.triu(attn_mask,diagonal=1),persistent=False)

        self.attn_dropout=nn.Dropout(self.dropout)
        self.residual_dropout=nn.Dropout(args.dropout)

    def forward(self,x:torch.Tensor,encoder_output:Optional[torch.Tensor]=None):
        batch_size,seq_len,_=x.shape
        if encoder_output is None:
            q=self.q_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
            k=self.k_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
            v=self.v_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
            q,k=self.rope_cache(q,k)

        else:
            q=self.q_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim)
            k=self.k_proj(encoder_output).view(batch_size,-1,self.n_heads,self.head_dim)
            v=self.v_proj(encoder_output).view(batch_size,-1,self.n_heads,self.head_dim)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        attn=torch.matmul(q,k.transpose(2,3))/math.sqrt(self.head_dim)

        if encoder_output is None:
            attn=attn+self.attn_mask[:,:,:seq_len,:seq_len]

        attn_probs=F.softmax(attn.float(),dim=-1).type_as(q)
        attn_probs=self.attn_dropout(attn_probs)
        output=torch.matmul(attn_probs,v)
        output=output.transpose(1,2).contiguous().view(batch_size,seq_len,-1)
        return self.residual_dropout(self.output_proj(output))


class FeedForward(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,dropout:float):
        super().__init__()

        self.fc1=nn.Linear(dim,hidden_dim,bias=True)
        self.fc2=nn.Linear(hidden_dim,dim,bias=True)

        self.dropout=nn.Dropout(dropout)

    def FeedForward(self,x):
        x=F.relu(self.fc1(x))
        x=self.dropout(self.fc2(x))

class TransformerEncoderBlock(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()

        self.self_attenion=MultiHeadAttention(args)
        self.feed_forward=FeedForward(dim=args.dim,hidden_dim=4*args.dim,dropout=args.dropout)
        self.attention_norm=nn.LayerNorm(args.dim)
        self.ffn_norm=nn.LayerNorm(args.dim)

    def forward(self,x:torch.Tensor):
        #原文认为应该是pre_norm
        h=x+self.self_attenion(self.attention_norm(x))
        out=h+self.feed_forward(self.ffn_norm(h))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()

        self.token_embedding=nn.Embedding(args.vocab_size,args.dim)
        self.register_buffer('pos_embedding',get_sinusoid_encoding_table(args.max_seq_length,args.dim))
        self.layers=nn.ModuleList(TransformerEncoderBlock(args) for _ in range(args.n_layers))

        self.final_norm=nn.layer_norm(args.dim)

        self.dropout=nn.Dropout(args.dropout)

    def forward(self,tokens:torch.Tensor):
        token_embedding=self.token_embedding(tokens)
        h=self.dropout(token_embedding)+self.pos_embedding[:tokens.size(1),:] #tokens dimension:[seq_len,args.dim]

        for layer in self.layers:
            h=layer(h)

        return self.final_norm(h)

class TransformerDecoderBlock(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()
        self.self_attention=MultiHeadAttention(args)
        self.cross_attention=MultiHeadAttention(args)
        self.feed_forward=FeedForward(dim=args.dim,hidden_dim=4*args.dim,dropout=args.dropout)
        self.self_attention_norm=nn.LayerNorm(args.dim)
        self.cross_attention_norm=nn.LayerNorm(args.dim)
        self.ffn_norm=nn.LayerNorm(args.dim)
    
    def forward(self,x:torch.Tensor,encoder_output:torch.Tensor):
        h = x + self.self_attention(self.self_attention_norm(x))
        h = h + self.cross_attention(self.cross_attention_norm(h),encoder_output)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class TransformerDecoder(nn.Module):
    def __init__(self,args:Dict):
        super().__init__()
        self.token_embedding=nn.Embedding(args.vocab_size,args.dim)
        self.resgister_buffer('pos_embedding',get_sinusoid_encoding_table(args.max_seq_length,args.dim))
        self.layers=nn.ModuleList(TransformerDecoderBlock(args) for _ in range(args.n_layers))
        self.final_norm=nn.LayerNorm(args.dim)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,encoder_output:torch.Tensor,tgt_tokens:torch.Tensor):
        token_embedding=self.token_embedding(tgt_tokens)
        h=self.drouput(token_embedding)+self.pos_embedding[:tgt_tokens.size(1),:]
        for layer in self,layers:
            h=layer(h,encoder_output)
        return self.final_norm(h)


class Transformer(nn.module):
    """完整的transformer模型"""
    def __init__(self,args:Dict):
        super().__init__()
        #编码器
        self.encoder=TransformerEncoder(args)
        #解码器
        self.decoder=TransformerDecoder(args)
        #输出层
        self.outproj=nn.Linear(args.dim,args.vocab_size,bias=False)

        #初始化权重
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        

    def forward(self,src_tokens:torch.tensor,tgt_tokens:torch.tensor):
        """
        Args:
            src_tokens: [batch_size,src_seq_len]
            tgt_tokens: [batch_size,tgt_seq_len]
        Returns:
            output: [batch_size,tgt_seq_len,vocab_size]
        """
        #编码器前向传播
        src_tokens=self.encoder(src_tokens)
        #解码器前向传播
        tgt_tokens=self.decoder(src_tokens,tgt_tokens)
        #输出层
        output=self.outproj(tgt_tokens)
        #返回输出   
        return output

