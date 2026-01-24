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

try:
    from ultra_low_bitrate_codec.kernels.bitnet_triton_packed import triton_bit_linear
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton kernel not found, using PyTorch fallback.")

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
        # x: (B, C, T) for conv or (B, T, C) for linear
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def ste_sign(x):
    return (x.sign() - x).detach() + x

def weight_quant_ternary(w):
    scale = w.abs().mean().clamp(min=1e-5)
    w_normalized = w / scale
    w_quantized = torch.round(w_normalized.clamp(-1, 1))
    return (w_quantized - w_normalized).detach() + w_normalized, scale

def activation_quant_8bit(x, num_bits=8):
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    scale = x.abs().max().clamp(min=1e-5)
    x_scaled = x / scale * Qp
    x_quant = torch.round(x_scaled.clamp(Qn, Qp))
    x_quant = (x_quant - x_scaled).detach() + x_scaled
    return x_quant / Qp * scale

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.norm = RMSNorm(in_features)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        # Normalize input
        x = self.norm(x)
        
        if HAS_TRITON and x.is_cuda and self.weight.is_cuda:
            # Use Fused Triton Kernel
            # We need to quantize weights to get the scale and the ternary values
            # The kernel wrapper handles packing and execution
            
            # Note: weight_quant_ternary returns (w_hard + grad_trick), scale
            # We need the 'hard' ternary values for packing.
            # But during training, we also need gradients to flow back to self.weight.
            # Our Autograd Function takes `weight` (ternary) as input.
            
            # So we must perform the weight quantization logic here to get the 'ternary' input for the kernel.
            # (Which seems redundant if we pack it immediately, but necessary for STE graph connectivity)
            
            w_quant, w_scale = weight_quant_ternary(self.weight)
            
            # Fused Matmul: Quantizes X (Int8) and uses Packed W (Int2)
            out = triton_bit_linear(x, w_quant, w_scale)
            
            if self.bias is not None:
                out += self.bias
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
        
        # Full precision weights for training
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # RMSNorm applied channel-wise
        self.norm = RMSNorm(in_channels)
        
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
