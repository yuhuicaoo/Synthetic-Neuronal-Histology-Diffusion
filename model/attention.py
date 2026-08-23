import torch
import torch.nn as nn
import torch.nn.functional as F
from config import AttentionConfig

class Head(nn.Module):
    """
    Singular head for self-attention
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()

        head_size = config.num_embd // config.num_heads

        self.key = nn.Linear(config.num_embd, head_size, bias=False)
        self.query = nn.Linear(config.num_embd, head_size, bias=False)
        self.value = nn.Linear(config.num_embd, head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        B, C, H, W = x.shape

        # flatten H, W channels together: (B, C, H, W) -> (B, H*W, C)
        x_flatten = x.reshape(B , C, H*W).transpose(1,2)

        k = self.key(x_flatten)         # (B, H*W, head_size) 
        q = self.query(x_flatten)       # (B, H*W, head_size)
        v = self.value(x_flatten)      # (B, H*W, head_size)

        # compute attention scores
        weights = q @ torch.transpose(k, dim0=1, dim1=2) * k.shape[-1]**-0.5        # (B , H*W, head_size) @ (B, head_size, H*W) --> (B, H*W, H*W)
        weights = F.softmax(weights, dim=-1)                                        # (B, H*W , H*W)
        weights = self.dropout(weights)

        # weighted aggregation on values, v
        out = weights @ v      # (B, H*W, H*W) @ (B, H*W, head_size) --> (B, H*W, head_size) 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        head_size = config.num_embd // config.num_heads

        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.projection = nn.Linear(config.num_embd, config.num_embd)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=config.num_embd)

    def forward(self, x):
        B, C, H, W = x.shape

        # each head returns (B, H*W, head_size)
        out_self_attention = torch.cat([head(x) for head in self.heads], dim=-1)        # (B, H*W, num_heads * head_size)
        out = self.dropout(self.projection(out_self_attention))                         # (B, H*W, C)

        # reshape back to (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # residual + norm  (B, C, H, W)
        return self.norm(x + out)
    
class SelfAttention2D(nn.Module):
    """Self attention for 2D feature maps"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm  = nn.GroupNorm(8, channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # flatten spatial dims for attention
        h = self.norm(x)
        h = h.reshape(B, C, H * W).transpose(1, 2)   # (B, H*W, C)

        # every spatial position attends to every other
        h, _ = self.attn(h, h, h)                     # (B, H*W, C)

        # projection
        h = self.proj_out(h)

        # reshape back
        h = h.transpose(1, 2).reshape(B, C, H, W)    # (B, C, H, W)

        return x + h    # residual connection
