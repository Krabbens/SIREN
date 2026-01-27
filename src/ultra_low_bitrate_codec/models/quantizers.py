"""
Improved Quantizers with Residual Vector Quantization (RVQ)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FSQ(nn.Module):
    """Finite Scalar Quantization"""
    def __init__(self, levels, dim=None):
        super().__init__()
        if dim is not None:
            assert dim == len(levels), f"Dim {dim} must match len(levels) {len(levels)}"
        
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("_basis", torch.cumprod(
            torch.cat([torch.tensor([1], dtype=torch.long), 
                      torch.tensor(levels, dtype=torch.long)[:-1]]), dim=0))
        
        self.dim = len(levels)
        self.vocab_size = self._levels.prod().item()
    
    def bound(self, z):
        return torch.tanh(z)
    
    def quantize(self, z):
        levels = self._levels.to(z.device)
        z_norm = (z + 1) / 2
        z_scaled = z_norm * (levels - 1)
        indices = torch.round(z_scaled)
        z_q = indices / (levels - 1) * 2 - 1
        return z_q, indices.long()
    
    def codes_to_indices(self, z_q_indices):
        basis = self._basis.to(z_q_indices.device)
        indices = (z_q_indices * basis).sum(dim=-1)
        return indices
    
    def indices_to_codes(self, indices):
        basis = self._basis.to(indices.device)
        levels = self._levels.to(indices.device)
        indices = indices.unsqueeze(-1)
        z_q_indices = (indices // basis) % levels
        z_q = z_q_indices.float() / (levels - 1) * 2 - 1
        return z_q

    def forward(self, z):
        z = self.bound(z)
        z_q, indices_vec = self.quantize(z)
        z_q = z + (z_q - z).detach()  # STE
        indices = self.codes_to_indices(indices_vec)
        loss = torch.mean((z_q.detach() - z) ** 2)
        return z_q, loss, indices

    def from_indices(self, indices):
        """Reconstruct z_q from indices"""
        return self.indices_to_codes(indices)


class ResidualFSQ(nn.Module):
    """
    Residual Finite Scalar Quantization (RFSQ)
    Applies multiple levels of FSQ to progressively refine the quantization.
    
    Each level quantizes the residual from the previous level.
    Total bits = num_levels * bits_per_level
    """
    def __init__(self, levels, num_levels=4, input_dim=None, shared_codebook=False):
        super().__init__()
        self.num_levels = num_levels
        self.dim = len(levels)
        
        # Input projection if dimensions don't match
        if input_dim is not None and input_dim != self.dim:
            self.input_proj = nn.Linear(input_dim, self.dim)
            self.output_proj = nn.Linear(self.dim, input_dim)
            self.has_proj = True
            self.output_dim = input_dim
        else:
            self.input_proj = None
            self.output_proj = None
            self.has_proj = False
            self.output_dim = self.dim
        
        # Create FSQ layers (shared or separate codebooks)
        if shared_codebook:
            base_fsq = FSQ(levels)
            self.quantizers = nn.ModuleList([base_fsq for _ in range(num_levels)])
        else:
            self.quantizers = nn.ModuleList([FSQ(levels) for _ in range(num_levels)])
        
        # Residual scaling (fixed to 1.0 for stability with LayerNorm)
        self.register_buffer("residual_scales", torch.ones(num_levels))
        
        # Single vocab size from base FSQ
        self.vocab_size = self.quantizers[0].vocab_size
        self.total_vocab_size = self.vocab_size ** num_levels
    
    def forward(self, z, num_levels=None):
        """
        Args:
            z: (B, T, D) input features
            num_levels: (int) optional, limit number of RFSQ levels
        Returns:
            z_q: (B, T, D) quantized output
            loss: scalar quantization loss
            indices: (B, T, num_levels) indices per level
        """
        # Project if needed
        if self.has_proj:
            z_proj = self.input_proj(z)
        else:
            z_proj = z
        
        residual = z_proj
        quantized = torch.zeros_like(z_proj)
        total_loss = 0.0
        all_indices = []
        
        max_levels = num_levels if num_levels is not None else len(self.quantizers)
        
        for i, (quantizer, scale) in enumerate(zip(self.quantizers, self.residual_scales)):
            if i >= max_levels:
                break
                
            # Quantize current residual
            z_q_level, loss_level, indices_level = quantizer(residual)
            
            # Scale and accumulate
            z_q_scaled = z_q_level * scale
            quantized = quantized + z_q_scaled
            
            # Update residual
            residual = residual - z_q_scaled
            
            # Accumulate loss (weight later levels less)
            level_weight = 1.0 / (i + 1)
            total_loss = total_loss + loss_level * level_weight
            
            all_indices.append(indices_level)
        
        # Stack indices
        indices = torch.stack(all_indices, dim=-1)  # (B, T, num_levels)
        
        # Project back if needed
        if self.has_proj:
            quantized = self.output_proj(quantized)
        
        # Straight-through estimator for full path
        z_q = z + (quantized - z).detach()
        
        return z_q, total_loss, indices
    
    def get_total_bits_per_frame(self):
        """Calculate bits per frame for bitrate estimation"""
        bits_per_level = math.log2(self.vocab_size)
        return bits_per_level * self.num_levels

    def from_indices(self, indices):
        """
        Reconstruct z_q from indices
        Args:
            indices: (B, T, num_levels)
        """
        z_q = 0
        
        for i, (quantizer, scale) in enumerate(zip(self.quantizers, self.residual_scales)):
            if i >= indices.shape[-1]:
                break
            indices_level = indices[:, :, i]
            z_q_level = quantizer.from_indices(indices_level)
            z_q = z_q + z_q_level * scale
            
        # Project back if needed
        if self.has_proj:
            z_q = self.output_proj(z_q)
            
        return z_q


class ProductQuantizer(nn.Module):
    """Product Quantization for speaker embeddings"""
    def __init__(self, input_dim, num_groups, codes_per_group=256):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.codes_per_group = codes_per_group
        
        assert input_dim % num_groups == 0
        self.group_dim = input_dim // num_groups
        
        self.codebooks = nn.ModuleList([
            nn.Embedding(codes_per_group, self.group_dim)
            for _ in range(num_groups)
        ])
        
        for cb in self.codebooks:
            cb.weight.data.uniform_(-1.0 / codes_per_group, 1.0 / codes_per_group)

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        B, T, D = x.shape
        x_groups = x.view(B, T, self.num_groups, self.group_dim)
        
        quantized_groups = []
        indices_list = []
        losses = 0
        
        for i, codebook in enumerate(self.codebooks):
            x_g = x_groups[:, :, i, :]
            x_flat = x_g.reshape(-1, self.group_dim)
            
            d = torch.sum(x_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook.weight ** 2, dim=1) - \
                2 * torch.matmul(x_flat, codebook.weight.t())
                
            idx = torch.argmin(d, dim=1)
            x_q_g = codebook(idx).view(B, T, self.group_dim)
            
            loss_g = F.mse_loss(x_q_g.detach(), x_g) + F.mse_loss(x_q_g, x_g.detach())
            x_q_g = x_g + (x_q_g - x_g).detach()
            
            quantized_groups.append(x_q_g)
            indices_list.append(idx.view(B, T))
            losses += loss_g
            
        x_q = torch.cat(quantized_groups, dim=-1).view(original_shape)
        indices = torch.stack(indices_list, dim=-1)
        
        if len(original_shape) == 2:
            indices = indices.squeeze(1)
            
        return x_q, losses, indices

    def from_indices(self, indices):
        """
        Reconstruct z_q from indices
        Args:
            indices: (B, T, num_groups) or (B, num_groups)
        """
        if indices.dim() == 2: # (B, num_groups) for speaker
             indices = indices.unsqueeze(1) # (B, 1, num_groups)
             
        B, T, G = indices.shape
        quantized_groups = []
        
        for i, codebook in enumerate(self.codebooks):
            idx = indices[:, :, i]
            x_q_g = codebook(idx) # (B, T, D_g)
            quantized_groups.append(x_q_g)
            
        x_q = torch.cat(quantized_groups, dim=-1)
        
        if T == 1 and self.input_dim == x_q.shape[-1]: # Handle squeezing if needed
             pass 
             
        return x_q
