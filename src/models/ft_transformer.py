"""
FT-Transformer: Feature Tokenizer + Transformer for Tabular Data
Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous numerical tabular features into d-dimensional tokens:
    e_j = x_j * w_j + b_j  where w_j, b_j in R^d.
    Also prepends a learnable [CLS] token.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Weight and bias per feature: shape (n_features, d_token)
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token))
        
        # Learnable [CLS] token: shape (1, 1, d_token)
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_token))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = 1
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, n_features)
        Returns:
            tokens: Tensor of shape (batch_size, n_features + 1, d_token)
        """
        batch_size = x.shape[0]
        
        # Linear projection per feature
        x_tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Prepend [CLS] token at index 0
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, x_tokens], dim=1)  # (batch_size, n_features + 1, d_token)
        return tokens


class TransformerBlock(nn.Module):
    """
    Single Transformer Layer with Multi-Head Self-Attention (Pre-LayerNorm) and FFN.
    """
    def __init__(self, d_token: int, n_heads: int, ffn_factor: float = 2.0,
                 attention_dropout: float = 0.1, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.dropout_res1 = nn.Dropout(residual_dropout)

        d_ffn = int(d_token * ffn_factor)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(ffn_dropout)
        )
        self.dropout_res2 = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + self.dropout_res1(attn_out)

        # Pre-LN Feed-Forward
        x = x + self.dropout_res2(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """
    Complete FT-Transformer Architecture:
    Tokenizer -> L Transformer Blocks -> [CLS] Extraction -> Classification Head
    """
    def __init__(
        self,
        n_features: int = 69,
        n_classes: int = 9,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
        ffn_factor: float = 2.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        head_dropout: float = 0.2,
        head_hidden_dim: int = 64
    ):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"

        # 1. Feature Tokenizer
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                ffn_factor=ffn_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout
            )
            for _ in range(n_blocks)
        ])

        # 3. Final Normalization
        self.final_norm = nn.LayerNorm(d_token)

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_token, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Tokenize features + [CLS] token
        tokens = self.tokenizer(x)

        # 2. Transformer encoder layers
        for block in self.blocks:
            tokens = block(tokens)

        # 3. Extract [CLS] token (index 0)
        cls_token = tokens[:, 0, :]
        cls_token = self.final_norm(cls_token)

        # 4. Classification logits
        logits = self.head(cls_token)
        return logits