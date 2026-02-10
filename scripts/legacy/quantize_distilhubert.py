#!/usr/bin/env python3
"""
DistilHuBERT INT8 Quantization for Edge Deployment

This script:
1. Loads pretrained DistilHuBERT from HuggingFace
2. Applies INT8 dynamic quantization
3. Saves quantized model for edge deployment
4. Optionally exports to ONNX

Usage:
    python quantize_distilhubert.py --output_dir checkpoints/distilhubert_int8
"""

import os
import argparse
import torch
import torch.quantization as quant
from pathlib import Path


def quantize_distilhubert(output_dir: str, export_onnx: bool = True):
    """Quantize DistilHuBERT to INT8."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')  # Quantization requires CPU
    
    print("=" * 60)
    print("DistilHuBERT INT8 Quantization")
    print("=" * 60)
    
    # =========================================================================
    # 1. Load DistilHuBERT
    # =========================================================================
    print("\n1. Loading DistilHuBERT from HuggingFace...")
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    
    model = AutoModel.from_pretrained("ntu-spml/distilhubert")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
    
    model.eval()
    model.to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    fp32_size = total_params * 4 / (1024**2)
    print(f"   Params: {total_params/1e6:.2f}M")
    print(f"   FP32 Size: {fp32_size:.1f} MB")
    
    # =========================================================================
    # 2. Apply Dynamic INT8 Quantization
    # =========================================================================
    print("\n2. Applying INT8 Dynamic Quantization...")
    
    # Dynamic quantization for Linear layers
    quantized_model = quant.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    # Estimate quantized size
    int8_size = total_params * 1 / (1024**2)  # 1 byte per param (approx)
    print(f"   Estimated INT8 Size: {int8_size:.1f} MB")
    print(f"   Compression: {fp32_size / int8_size:.1f}x")
    
    # =========================================================================
    # 3. Verify Quantized Model
    # =========================================================================
    print("\n3. Verifying quantized model...")
    
    # Test inference
    test_audio = torch.randn(1, 16000)  # 1 second of audio
    
    with torch.no_grad():
        # Original
        orig_out = model(test_audio, output_hidden_states=True)
        orig_features = orig_out.last_hidden_state
        
        # Quantized
        quant_out = quantized_model(test_audio, output_hidden_states=True)
        quant_features = quant_out.last_hidden_state
    
    # Compare
    diff = (orig_features - quant_features).abs().mean().item()
    print(f"   Output shape: {quant_features.shape}")
    print(f"   L1 difference (FP32 vs INT8): {diff:.6f}")
    
    # =========================================================================
    # 4. Save Quantized Model
    # =========================================================================
    print("\n4. Saving quantized model...")
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': model.config.to_dict(),
        'quantization': 'dynamic_int8'
    }, f"{output_dir}/distilhubert_int8.pt")
    
    # Save processor config
    processor.save_pretrained(output_dir)
    
    # Measure actual file size
    pt_size = os.path.getsize(f"{output_dir}/distilhubert_int8.pt") / (1024**2)
    print(f"   Saved: {output_dir}/distilhubert_int8.pt ({pt_size:.1f} MB)")
    
    # =========================================================================
    # 5. Export to ONNX (Optional)
    # =========================================================================
    if export_onnx:
        print("\n5. Exporting to ONNX...")
        try:
            import onnx
            
            # ONNX export requires the non-quantized model with tracing
            # We'll export FP32 and note that ONNX Runtime can quantize
            dummy_input = torch.randn(1, 16000)
            
            # Wrapper for clean export
            class DistilHuBERTWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    
                def forward(self, x):
                    return self.model(x).last_hidden_state
            
            wrapped = DistilHuBERTWrapper(model)
            wrapped.eval()
            
            onnx_path = f"{output_dir}/distilhubert.onnx"
            torch.onnx.export(
                wrapped,
                dummy_input,
                onnx_path,
                input_names=['audio'],
                output_names=['features'],
                dynamic_axes={
                    'audio': {0: 'batch', 1: 'samples'},
                    'features': {0: 'batch', 1: 'frames'}
                },
                opset_version=14
            )
            
            onnx_size = os.path.getsize(onnx_path) / (1024**2)
            print(f"   Saved: {onnx_path} ({onnx_size:.1f} MB)")
            print("   Note: Use ONNX Runtime quantization for INT8 ONNX")
            
        except Exception as e:
            print(f"   ONNX export failed: {e}")
            print("   Continuing without ONNX...")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   Original (FP32): {fp32_size:.1f} MB")
    print(f"   Quantized (INT8): {pt_size:.1f} MB")
    print(f"   Compression: {fp32_size / pt_size:.1f}x")
    print(f"   Quality loss (L1): {diff:.6f}")
    print("=" * 60)
    
    return quantized_model, processor


class QuantizedDistilHuBERT(torch.nn.Module):
    """
    Wrapper for loading and using quantized DistilHuBERT.
    
    Usage:
        model = QuantizedDistilHuBERT.load("checkpoints/distilhubert_int8")
        features = model(waveform)  # (B, T, 768)
    """
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.out_dim = 768
        
    @classmethod
    def load(cls, checkpoint_dir: str):
        """Load quantized model from checkpoint."""
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        
        # Load base model and apply quantization
        model = AutoModel.from_pretrained("ntu-spml/distilhubert")
        model.eval()
        
        # Apply quantization
        model = quant.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Load quantized weights
        ckpt = torch.load(f"{checkpoint_dir}/distilhubert_int8.pt", map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Load processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_dir)
        
        return cls(model, processor)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio.
        
        Args:
            waveform: (B, T) raw audio at 16kHz
            
        Returns:
            features: (B, T', 768)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        with torch.no_grad():
            outputs = self.model(waveform, output_hidden_states=True)
            
        return outputs.last_hidden_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="checkpoints/distilhubert_int8")
    parser.add_argument("--no_onnx", action="store_true")
    args = parser.parse_args()
    
    quantize_distilhubert(args.output_dir, export_onnx=not args.no_onnx)


if __name__ == "__main__":
    main()
