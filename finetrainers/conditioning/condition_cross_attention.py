import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block:
      - Query comes from 'x' (the main hidden states)
      - Key/Value come from 'context' (the conditioning or encoder output)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        Args:
            x: [batch_size, seq_len, embed_dim] - queries
            context: [batch_size, context_len, embed_dim] - keys & values (conditioning)
        Returns:
            out: The cross-attended hidden states (same shape as x)
        """
        # Apply LayerNorm before attention (pre-norm variant)
        x_norm = self.layernorm(x)
        
        # In cross attention:
        #   Q = x_norm
        #   K,V = context
        attn_out, _ = self.attn(
            query=x_norm, 
            key=context, 
            value=context
        )

        # Residual connection
        out = x + self.dropout_layer(attn_out)
        return out