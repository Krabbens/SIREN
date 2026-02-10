
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultra_low_bitrate_codec.models.bitlinear import BitLinear, BitConv1d, RMSNorm, activation_quant_8bit
from ultra_low_bitrate_codec.models.quantizers import RVQ

class BitConvFeatureExtractor(nn.Module):
    """
    BitNet version of the convolutional frontend.
    Uses BitConv1d with Ternary weights {-1, 0, +1}.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        
        def bit_block(in_ch, out_ch, k, s, p):
            return nn.Sequential(
                BitConv1d(in_ch, out_ch, k, stride=s, padding=p),
                nn.GELU(),
                # Activation quant is handled inside BitConv1d for the NEXT layer, 
                # but we add it here explicitly for clarity if needed.
                # Actually BitConv1d has RMSNorm + Quant built-in.
            )

        self.convs = nn.Sequential(
            # Layer 1: 5x downsampling
            bit_block(1, 128, 10, 5, 4),
            
            # Layers 2-5: 2x each (Total 16x -> 80x total)
            bit_block(128, 192, 3, 2, 1),
            bit_block(192, 192, 3, 2, 1),
            bit_block(192, 192, 3, 2, 1),
            bit_block(192, 256, 3, 2, 1),
            
            # Layers 6-7: 2x each (Total 4x -> 320x total)
            bit_block(256, 256, 4, 2, 1), # Adjusted kernel for exact 2x
            bit_block(256, out_dim, 4, 2, 1),
        )

    def forward(self, x):
        # x: (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.convs(x)

class BitAttention(nn.Module):
    """
    Multi-Head Attention using BitLinear.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        B, T, C = x.shape
        x = self.norm(x)
        
        qkv = self.qkv(x) # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Softmax Attention (standard, as weights are bits but attention map needs precision)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class BitMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(dim, hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, dim)
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.net(self.norm(x))

class BitTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4):
        super().__init__()
        self.attn = BitAttention(dim, num_heads)
        self.mlp = BitMLP(dim, dim * ff_mult)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class MicroTransformer(nn.Module):
    """
    BitNet-based lightweight Transformer for distillation.
    Upgraded for better convergence:
    - Higher hidden_dim (384)
    - More layers (8)
    - High-precision output projection (nn.Linear)
    - Optional RVQ Bottleneck for compression
    """
    def __init__(
        self, 
        hidden_dim=384, 
        output_dim=768, 
        num_layers=8, 
        num_heads=8,
        use_rvq=False,
        rvq_num_quantizers=8,
        rvq_codebook_size=1024,
        rvq_dropout_p=0.0
    ):
        super().__init__()
        
        self.encoder = BitConvFeatureExtractor(out_dim=hidden_dim)
        self.use_rvq = use_rvq
        
        if self.use_rvq:
            self.rvq = RVQ(
                num_quantizers=rvq_num_quantizers,
                codebook_size=rvq_codebook_size,
                dim=hidden_dim,
                dropout_p=rvq_dropout_p
            )
        
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        
        self.blocks = nn.ModuleList([
            BitTransformerBlock(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(hidden_dim)
        # Use Standard Linear for the final distillation target to preserve precision
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (B, T_audio)
        
        # 1. Conv
        x = self.encoder(x) # (B, 384, T_frames)
        x = x.transpose(1, 2) # (B, T_frames, 384)
        
        # 1.5 RVQ Bottleneck (Optional)
        rvq_loss = 0.0
        rvq_indices = None
        
        if self.use_rvq:
            x, rvq_loss, rvq_indices = self.rvq(x)
        
        # 2. PE
        T = x.size(1)
        if T > self.pos_emb.size(1):
             pe = F.interpolate(self.pos_emb.transpose(1, 2), size=T).transpose(1, 2)
        else:
             pe = self.pos_emb[:, :T, :]
        x = x + pe
        
        # 3. Transformer
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        
        # 4. Project
        x = self.out_proj(x)
        
        if self.use_rvq:
            return x, rvq_loss, rvq_indices
            
        return x

if __name__ == "__main__":
    # Test standard
    model = MicroTransformer()
    print(f"Standard MicroTransformer Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    x = torch.randn(1, 16000)
    y = model(x)
    print(f"In: {x.shape} -> Out: {y.shape}")
    
    # Test RVQ
    print("\ntesting RVQ integration:")
    model_rvq = MicroTransformer(use_rvq=True, rvq_num_quantizers=4)
    y_rvq, loss, indices = model_rvq(x)
    print(f"In: {x.shape} -> Out: {y_rvq.shape}, Indices: {indices.shape}, Loss: {loss.item()}")
