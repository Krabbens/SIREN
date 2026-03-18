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
        # Shift from tanh (which saturates and kills gradients for large inputs)
        # to a standardized L2 normalization approach scaled by a learned parameter
        # or just a simple bounded sigmoid that preserves gradient flow.
        
        # Simple scalable L2 norm (Project onto L2 sphere)
        # z_norm = F.normalize(z, p=2, dim=-1)
        # return z_norm * math.sqrt(self.dim) # Keep variance ~1
        
        # Alternative: use sigmoid-based bounding [-1, 1] which is softer than tanh
        return 2 * torch.sigmoid(z) - 1
    
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


class BinaryTreeCodebook(nn.Module):
    """
    Binary Tree Codebook — O(log N) memory representation of N codes.
    
    Instead of storing N vectors explicitly [O(N × D) memory],
    we store `depth` levels with 2 vectors each [O(depth × D) memory].
    
    Each codeword is the SUM of one vector chosen per level:
        code[i] = sum(level[k][bit_k(i)] for k in 0..depth-1)
    
    This gives 2^depth distinct codes using only 2 × depth × D parameters.
    
    Example: depth=8 → 256 codes using 16 vectors instead of 256.
    
    Encoding: greedy residual search (top-down tree traversal)
    Decoding: bit decomposition → additive reconstruction
    """
    def __init__(self, depth, dim):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.codebook_size = 2 ** depth
        
        # Each level: 2 learned vectors (binary choice at each tree level)
        # Shape per level: (2, dim)
        self.levels = nn.ParameterList([
            nn.Parameter(torch.randn(2, dim) * (0.02 / math.sqrt(depth)))
            for _ in range(depth)
        ])
    
    @property
    def memory_params(self):
        """Number of parameters in codebook."""
        return 2 * self.depth * self.dim
    
    @property
    def flat_equivalent_params(self):
        """Equivalent flat codebook parameters."""
        return self.codebook_size * self.dim
    
    @property
    def compression_ratio(self):
        """Memory compression vs flat codebook."""
        return self.flat_equivalent_params / self.memory_params
    
    def decode(self, indices):
        """
        Reconstruct vectors from integer indices.
        
        Args:
            indices: (...,) integer tensor, values in [0, 2^depth)
        Returns:
            vectors: (..., dim) reconstructed codebook entries
        """
        result = torch.zeros(*indices.shape, self.dim, 
                            device=indices.device, dtype=self.levels[0].dtype)
        
        for k in range(self.depth):
            bit = (indices >> k) & 1  # (...,) 0 or 1
            # Index into level's 2 vectors using bit
            vec = self.levels[k][bit]  # (..., dim)
            result = result + vec
        
        return result
    
    def encode(self, x):
        """
        Find nearest code via greedy residual tree traversal.
        
        At each level, pick the vector (0 or 1) that minimizes
        the residual, then subtract it and continue to next level.
        
        Args:
            x: (..., dim) input vectors
        Returns:
            indices: (...,) integer indices in [0, 2^depth)
        """
        residual = x
        indices = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        
        for k in range(self.depth):
            v0 = self.levels[k][0]  # (dim,)
            v1 = self.levels[k][1]  # (dim,)
            
            # Distance from residual to each option
            d0 = torch.sum((residual - v0) ** 2, dim=-1)  # (...,)
            d1 = torch.sum((residual - v1) ** 2, dim=-1)  # (...,)
            
            bit = (d1 < d0).long()  # 1 if v1 is closer
            
            # Gather chosen vector
            chosen = self.levels[k][bit]  # (..., dim)
            
            # Update residual
            residual = residual - chosen
            
            # Accumulate bit into index
            indices = indices + bit * (2 ** k)
        
        return indices
    
    def forward(self, x):
        """
        Quantize input vectors.
        
        Args:
            x: (..., dim) input vectors
        Returns:
            x_q: (..., dim) quantized (with STE gradient)
            loss: scalar commitment loss
            indices: (...,) codebook indices
        """
        indices = self.encode(x)
        x_q = self.decode(indices)
        
        # Commitment loss (EMA-style, both directions)
        loss = F.mse_loss(x_q.detach(), x) + F.mse_loss(x_q, x.detach())
        
        # Straight-through estimator
        x_q = x + (x_q - x).detach()
        
        return x_q, loss, indices


class TreeProductQuantizer(nn.Module):
    """
    Product Quantizer with Binary Tree Codebooks — O(G × log N × D/G) memory.
    
    Drop-in replacement for ProductQuantizer, but each group uses a
    BinaryTreeCodebook instead of a flat nn.Embedding.
    
    Memory comparison (for input_dim=256, 8 groups, 256 codes):
        Flat PQ:  8 × 256 × 32 = 65,536 params
        Tree PQ:  8 × 8 × 2 × 32 = 4,096 params  (16× smaller)
    """
    def __init__(self, input_dim, num_groups, codes_per_group=256):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.codes_per_group = codes_per_group
        
        assert input_dim % num_groups == 0
        self.group_dim = input_dim // num_groups
        
        # Depth = log2(codes_per_group)
        self.depth = int(math.log2(codes_per_group))
        assert 2 ** self.depth == codes_per_group, \
            f"codes_per_group must be power of 2, got {codes_per_group}"
        
        self.codebooks = nn.ModuleList([
            BinaryTreeCodebook(self.depth, self.group_dim)
            for _ in range(num_groups)
        ])
        
        # Report memory savings
        tree_params = sum(cb.memory_params for cb in self.codebooks)
        flat_params = sum(cb.flat_equivalent_params for cb in self.codebooks)
        self._tree_params = tree_params
        self._flat_params = flat_params
    
    def memory_report(self):
        """Return human-readable memory comparison."""
        return (f"TreePQ: {self._tree_params:,} params vs "
                f"FlatPQ: {self._flat_params:,} params "
                f"({self._flat_params / self._tree_params:.1f}× compression)")
    
    def forward(self, x):
        """
        Args:
            x: (B, D) or (B, T, D) input
        Returns:
            x_q: same shape, quantized (STE gradient)
            loss: scalar
            indices: (B, [T,] num_groups) integer indices
        """
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        B, T, D = x.shape
        x_groups = x.view(B, T, self.num_groups, self.group_dim)
        
        quantized_groups = []
        indices_list = []
        total_loss = 0
        
        for i, codebook in enumerate(self.codebooks):
            x_g = x_groups[:, :, i, :]  # (B, T, group_dim)
            
            x_q_g, loss_g, idx_g = codebook(x_g)
            
            quantized_groups.append(x_q_g)
            indices_list.append(idx_g)  # (B, T)
            total_loss = total_loss + loss_g
        
        x_q = torch.cat(quantized_groups, dim=-1).view(original_shape)
        indices = torch.stack(indices_list, dim=-1)
        
        if len(original_shape) == 2:
            indices = indices.squeeze(1)
        
        return x_q, total_loss, indices
    
    def from_indices(self, indices):
        """
        Reconstruct from indices.
        
        Args:
            indices: (B, num_groups) or (B, T, num_groups)
        Returns:
            x_q: (B, [T,] input_dim)
        """
        if indices.dim() == 2:
            indices = indices.unsqueeze(1)
        
        B, T, G = indices.shape
        quantized_groups = []
        
        for i, codebook in enumerate(self.codebooks):
            idx = indices[:, :, i]  # (B, T)
            x_q_g = codebook.decode(idx)  # (B, T, group_dim)
            quantized_groups.append(x_q_g)
        
        x_q = torch.cat(quantized_groups, dim=-1)
        
        if T == 1:
            x_q = x_q.squeeze(1)
        
        return x_q


class VectorQuantizer(nn.Module):
    """
    Improved Vector Quantizer with:
    1. Lazy initialization (kmeans style)
    2. Dead code revival (restart unused codes)
    3. Orthogonal regularization support
    """
    def __init__(self, codebook_size, dim, beta=0.25, threshold_ema_dead_code=2):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        # Stats for dead code revival
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embed', self.embedding.weight.data.clone())
        self.register_buffer('inited', torch.Tensor([0]))
        
    def init_codebook(self, z):
        """Lazy initialization using first batch of data"""
        if self.inited.item() == 1:
            return
            
        z_flattened = z.reshape(-1, self.dim)
        # Reservoir sampling or just random choice if batch >> codebook_size
        if z_flattened.size(0) < self.codebook_size:
            # Not enough data to init cleanly, wait for next batch or duplicates
            # For now, just duplicate random elements
            indices = torch.randint(0, z_flattened.size(0), (self.codebook_size,))
        else:
            # Random selection without replacement
            indices = torch.randperm(z_flattened.size(0))[:self.codebook_size]
            
        self.embedding.weight.data.copy_(z_flattened[indices])
        self.inited.fill_(1)
    
    def expire_codes_(self, z):
        """
        Restart dead codes by replacing them with random input vectors from current batch.
        """
        if self.threshold_ema_dead_code == 0:
            return

        # Calculate usage
        z_flattened = z.reshape(-1, self.dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Count usage
        bins = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        
        # Identify dead codes
        dead_codes = bins < self.threshold_ema_dead_code
        num_dead = dead_codes.sum()
        
        if num_dead > 0:
            # Pick random inputs to replace dead codes
            # We want to pick inputs that are NOT well represented, ideally.
            # But random sampling from input batch is a good heuristic.
            rand_indices = torch.randint(0, z_flattened.size(0), (num_dead,))
            self.embedding.weight.data[dead_codes] = z_flattened[rand_indices].to(self.embedding.weight.dtype)
            
            # Reset optimizer usage? (Not handled here as we don't control optim)

    def forward(self, z):
        """
        Args:
            z: (B, T, D) or (B, D) input
        Returns:
            z_q: quantized output
            loss: commitment loss
            indices: encoding indices
        """
        # Lazy init
        if self.training and self.inited.item() == 0:
            self.init_codebook(z)
            
        z_flattened = z.reshape(-1, self.dim)
        
        # Calculate distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices).view(z.shape)
        
        # Losses
        commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                          torch.mean((z_q - z.detach()) ** 2)
        
        # Dead code revival during training
        # if self.training:
        #     self.expire_codes_(z) # Can be expensive every step, maybe only every N steps?
        #     Implementation note: For simplicity/speed we skip automatic expire per step 
        #     unless explicitly requested or refactored to be strided.
        #     Ideally done via EMA updates, but here we use simple Embedding.
               
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, commitment_loss, indices.view(z.shape[:-1])


class RVQ(nn.Module):
    """
    Residual Vector Quantizer (RVQ) layer.
    """
    def __init__(self, num_quantizers, codebook_size, dim, dropout_p=0.0):
        super().__init__()
        self.num_quantizers = num_quantizers
        # Projections to isolate codebook dimension if needed in future
        self.dim = dim 
        self.codebook_dim = dim 
        self.dropout_p = dropout_p
        
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, dim) 
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x, n_quantizers=None):
        if n_quantizers is None:
            if self.training and self.dropout_p > 0:
                # Randomly dropout quantizers for bitrate scalability
                n_quantizers = int(torch.randint(1, self.num_quantizers + 1, (1,)).item())
            else:
                n_quantizers = self.num_quantizers
                
        residual = x
        quantized_out = 0
        total_loss = 0
        indices = []
        
        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            
            z_q, loss, idx = quantizer(residual)
            
            residual = residual - z_q
            quantized_out = quantized_out + z_q
            total_loss = total_loss + loss
            indices.append(idx)
            
        return quantized_out, total_loss, torch.stack(indices, dim=-1)

    def from_indices(self, indices):
        """
        Reconstruct from indices.
        indices: (B, T, num_quantizers)
        """
        z_q = 0
        for i, quantizer in enumerate(self.quantizers):
            if i >= indices.shape[-1]:
                break
            
            idx = indices[..., i]
            z_q = z_q + quantizer.embedding(idx)
            
        return z_q
