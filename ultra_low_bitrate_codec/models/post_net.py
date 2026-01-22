import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Snake(nn.Module):
    """
    Snake activation function as described in:
    "Neural Networks Fail to Learn Periodic Functions and How to Fix It" (NeurIPS 2020).
    Feature: Periodic inductive bias.
    Formula: x + (1/alpha) * sin^2(alpha * x)
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x) ** 2


class SnakeBeta(nn.Module):
    """
    Enhanced Snake activation function (SnakeBeta).
    Adds a beta parameter to control magnitude of periodic components.
    Formula: x + (1/beta) * sin^2(alpha * x)
    """
    def __init__(self, channels):
        super().__init__()
        # Improve initialization: Log-scale distribution for alpha (frequencies)
        # This helps cover the full audio spectrum from low to high freqs
        # Similar to BigVGAN's specific initialization
        
        # Log-uniform distribution from 0.1 to 100
        # This corresponds roughly to low-freq to high-freq features
        zeros = torch.zeros(1, channels, 1)
        self.alpha = nn.Parameter(zeros.normal_(0, 1).exp().abs() + 1e-9)
        
        # Beta controls amplitude, initialize to 1
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / (self.beta + 1e-9)) * torch.sin(self.alpha * x) ** 2


class SnakePhase(nn.Module):
    """
    SnakePhase activation (Centered).
    Adds learnable phase shift (phi) to standard Snake, with centering correction.
    Formula: x + (1/beta) * (sin^2(alpha * x + phi) - sin^2(phi))
    Centering ensures f(0) = 0, improving stability and convergence.
    """
    def __init__(self, channels):
        super().__init__()
        # Log-scale init for frequencies (alpha)
        zeros = torch.zeros(1, channels, 1)
        self.alpha = nn.Parameter(zeros.normal_(0, 1).exp().abs() + 1e-9)
        
        # Learnable phase shift (phi)
        self.phi = nn.Parameter(torch.empty(1, channels, 1).uniform_(-math.pi, math.pi))
        
        # Beta controls amplitude
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        # Subtracting sin^2(phi) ensures f(0) = 0 (bias removal)
        return x + (1.0 / (self.beta + 1e-9)) * (torch.sin(self.alpha * x + self.phi) ** 2 - torch.sin(self.phi) ** 2)

class AudioEnhancer(nn.Module):
    """
    Post-processing model to enhance audio after decompression.
    Uses standard Conv1d layers and Snake activations.
    """
    def __init__(self, 
                 in_channels=1, 
                 hidden_channels=64, 
                 out_channels=1, 
                 num_layers=8,
                 kernel_size=7,
                 use_snake_beta=False):
        super().__init__()
        
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
                    SnakeBeta(hidden_channels) if use_snake_beta else Snake(hidden_channels)
                )
            )
            
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        """
        Args:
            x: Input audio waveform of shape (B, 1, T) or (B, T)
        Returns:
            Enhanced audio waveform (B, 1, T)
        """
        # Ensure input is (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        residual = x
        
        x = self.input_conv(x)
        
        for layer in self.layers:
            # Add simple residual connection for gradient flow
            res_layer = x
            x = layer(x)
            x = x + res_layer
            
        x = self.output_conv(x)
        
        # Global residual connection
        return residual + x
