"""
BitNet Layers for Ultra-Low Bitrate Audio Codec

Implements ternary quantization {-1, 0, +1} based on BitNet 1.58b.
Uses Straight-Through Estimator (STE) for gradient computation.

Reference: "The Era of 1-bit LLMs" (Microsoft, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os

try:
    try:
        from ultra_low_bitrate_codec.kernels.bitnet_triton_packed import triton_bit_linear, pack_ternary
    except ImportError:
        from .kernels.bitnet_triton_packed import triton_bit_linear, pack_ternary
    HAS_TRITON = not os.environ.get('DISABLE_TRITON', '')
    if not HAS_TRITON:
        print("[BitLinear] Triton kernel DISABLED via DISABLE_TRITON env var")
except ImportError:
    HAS_TRITON = False
    pack_ternary = None

class RMSNorm(nn.Module):
    # ... (Keep existing RMSNorm) ...
    """
    Root Mean Square Layer Normalization.
    Used in BitNet instead of LayerNorm (no mean subtraction).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Cast to fp32 for stable variance calculation (prevents bfloat16 overflow over time)
        x_fp32 = x.float()
        rms = torch.rsqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).type_as(x) * self.weight

@torch.jit.script
def ste_sign(x):
    return (x.sign() - x).detach() + x

@torch.jit.script
def weight_quant_ternary(w):
    scale = w.abs().mean().clamp(min=1e-4)
    w_normalized = w / scale
    w_quantized = torch.round(w_normalized.clamp(-1, 1))
    return (w_quantized - w_normalized).detach() + w_normalized, scale

@torch.jit.script
def activation_quant_8bit(x, num_bits: int = 8):
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    # Guard against NaN/Inf propagation from upstream
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    scale = x.abs().max().clamp(min=1e-5)
    x_scaled = x / scale * Qp
    x_quant = torch.round(x_scaled.clamp(Qn, Qp))
    x_quant = (x_quant - x_scaled).detach() + x_scaled
    return x_quant / Qp * scale

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, norm_input: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm_input = norm_input
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        if norm_input:
            self.norm = RMSNorm(in_features)
        else:
            self.register_parameter('norm', None)
        self._init_weights()
        
        # Packed weight cache (invalidated when weight changes)
        self._cached_w_packed = None
        self._cached_K_aligned = None
        self._cached_weight_version = None
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def _get_packed_weights(self, w_quant):
        """Return cached packed weights, re-packing only when needed."""
        # Use data_ptr as a cheap version check — changes after optimizer step
        current_version = self.weight.data_ptr()
        if (self._cached_w_packed is not None 
            and self._cached_weight_version == current_version
            and not self.training):
            return self._cached_w_packed, self._cached_K_aligned
        
        w_packed, K_aligned = pack_ternary(w_quant)
        
        # Only cache in eval mode (weights don't change)
        if not self.training:
            self._cached_w_packed = w_packed
            self._cached_K_aligned = K_aligned
            self._cached_weight_version = current_version
        
        return w_packed, K_aligned
    
    def forward(self, x):
        if self.norm_input and self.norm is not None:
            x = self.norm(x)
        
        if HAS_TRITON and x.is_cuda and self.weight.is_cuda:
            w_quant, w_scale = weight_quant_ternary(self.weight)
            
            # Get (cached) packed weights
            w_packed, K_aligned = self._get_packed_weights(w_quant)
            
            # v3 kernel: fused matmul + optional bias epilogue
            out = triton_bit_linear(x, w_quant, w_scale, self.bias, w_packed, K_aligned)
            return out
        else:
            # Fallback
            x_quant = activation_quant_8bit(x)
            w_quant, w_scale = weight_quant_ternary(self.weight)
            out = F.linear(x_quant, w_quant, self.bias)
            return out * w_scale


class BitConv1d(nn.Module):
    """
    Conv1d layer with ternary weights {-1, 0, +1}.
    
    Same quantization scheme as BitLinear but for 1D convolutions.
    Ideal for audio processing in vocoder.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_input: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.norm_input = norm_input
        
        # Full precision weights for training
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # RMSNorm applied channel-wise
        if norm_input:
            self.norm = RMSNorm(in_channels)
        else:
            self.register_parameter('norm', None)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor (B, C, T)
        Returns:
            Output tensor (B, C_out, T')
        """
        B, C, T = x.shape
        
        # Apply RMSNorm (need to permute for channel-last norm)
        if self.norm_input and self.norm is not None:
            x = x.transpose(1, 2)  # (B, T, C)
            x = self.norm(x)
            x = x.transpose(1, 2)  # (B, C, T)
        
        # Quantize activations
        x_quant = activation_quant_8bit(x)
        
        # Quantize weights
        w_quant, w_scale = weight_quant_ternary(self.weight)
        
        # Convolution
        out = F.conv1d(
            x_quant, w_quant, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        # Scale output
        out = out * w_scale
        
        return out


class BitConvTranspose1d(nn.Module):
    """
    Transposed Conv1d with ternary weights.
    Used for upsampling in vocoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.norm = RMSNorm(in_channels)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        B, C, T = x.shape
        
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        x_quant = activation_quant_8bit(x)
        w_quant, w_scale = weight_quant_ternary(self.weight)
        
        out = F.conv_transpose1d(
            x_quant, w_quant, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
        
        out = out * w_scale
        return out


class BitConv2d(nn.Module):
    """
    Conv2d layer with ternary weights {-1, 0, +1}.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Determine kernel dimensions
        if isinstance(kernel_size, int):
            self.kernel_size_tuple = (kernel_size, kernel_size)
        else:
            self.kernel_size_tuple = kernel_size
            
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size_tuple)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.norm = RMSNorm(in_channels)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # RMSNorm expects channel last
        # Move C to last: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # Back to (B, C, H, W)
        
        x_quant = activation_quant_8bit(x)
        w_quant, w_scale = weight_quant_ternary(self.weight)
        
        out = F.conv2d(
            x_quant, w_quant, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        out = out * w_scale
        return out

# Test the implementation
if __name__ == "__main__":
    print("Testing BitNet layers...")
    
    # Test BitLinear
    linear = BitLinear(256, 512)
    x = torch.randn(2, 100, 256)
    y = linear(x)
    print(f"BitLinear: {x.shape} → {y.shape}")
    
    # Test BitConv1d
    conv = BitConv1d(256, 512, kernel_size=7, padding=3)
    x = torch.randn(2, 256, 100)
    y = conv(x)
    print(f"BitConv1d: {x.shape} → {y.shape}")
    
    # Count parameters and theoretical size
    total_params = sum(p.numel() for p in linear.parameters())
    ternary_bits = total_params * 1.58  # 1.58 bits for ternary
    print(f"\nBitLinear params: {total_params:,}")
    print(f"Theoretical size: {ternary_bits / 8 / 1024:.2f} KB (ternary)")
    print(f"FP32 size: {total_params * 4 / 1024:.2f} KB")
    
    print("\n✓ All tests passed!")

class BitSnakeBeta(nn.Module):
    """
    BitNet-optimized SnakeBeta activation.
    Combines:
    1. RMSNorm (for stability before activation)
    2. SnakeBeta (periodic activation)
    3. Activation Quantization (8-bit, for BitNet compatibility)
    """
    def __init__(self, channels):
        super().__init__()
        # RMSNorm for input stability
        self.norm = RMSNorm(channels)
        
        # Log-uniform initialization for alpha (frequencies)
        zeros = torch.zeros(1, channels, 1)
        self.alpha = nn.Parameter(zeros.normal_(0, 1).exp().abs() + 1e-9)
        
        # Beta controls amplitude, initialize to 1
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        # x: (B, C, T) or (B, T, C)
        # RMSNorm expects channel last
        orig_shape = x.shape
        if x.dim() == 3 and x.shape[1] == self.alpha.shape[1]: 
             # (B, C, T) case -> Permute to (B, T, C) for Norm
             x = x.transpose(1, 2)
             x = self.norm(x)
             x = x.transpose(1, 2)
        else:
             # (B, T, C) or (B, C)
             x = self.norm(x)

        # SnakeBeta logic
        alpha = self.alpha
        beta = self.beta
        
        if x.dim() == 2:
             # (B, C) case
            alpha = alpha.squeeze(-1)
            beta = beta.squeeze(-1)
        elif x.shape[-1] == self.alpha.shape[1] and x.shape[1] != self.alpha.shape[1]:
             # (B, T, C) case -> Permute parameters to (1, 1, C) to broadcast correctly check
             # Current alpha is (1, C, 1). If x is (B, T, C), we need alpha as (1, 1, C)
             alpha = alpha.permute(0, 2, 1)
             beta = beta.permute(0, 2, 1)
        
        x = x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x) ** 2
        
        # Quantize Output (8-bit) for next BitLinear layer
        x = activation_quant_8bit(x)
        
        return x
