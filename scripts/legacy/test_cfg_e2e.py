
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import argparse
from src.ultra_low_bitrate_codec.models.micro_encoder import MicroEncoderTiny
from src.ultra_low_bitrate_codec.models.factorizer import InformationFactorizerV2
from src.ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from src.ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching

def load_models(args, device):
    # Load MicroEncoder
    encoder = MicroEncoderTiny(hidden_dim=256, n_layers=4)
    encoder_path = os.path.join(args.checkpoint_dir, "encoder_ep25.pt")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Loaded {encoder_path}")
    encoder.to(device).eval()

    # Load Factorizer
    factorizer = InformationFactorizerV2()
    factorizer_path = os.path.join(args.checkpoint_dir, "factorizer_ep25.pt")
    if os.path.exists(factorizer_path):
        factorizer.load_state_dict(torch.load(factorizer_path, map_location=device))
        print(f"Loaded {factorizer_path}")
    factorizer.to(device).eval()

    # Load Fuser
    # Using 128 hidden dim as per common config
    fuser = ConditionFuserV2(hidden_dim=256, output_dim=256) 
    # Check dimensions in train_e2e_encoder.py main() - it uses 256 for Flow but Fuser might be diff?
    # Checking train_e2e_encoder.py...
    # fuser = ConditionFuserV2(hidden_dim=args.encoder_hidden, output_dim=flow.hidden_dim)
    # encoder_hidden=256 default. Flow hidden=256 default.
    fuser_path = os.path.join(args.checkpoint_dir, "fuser_ep25.pt")
    if os.path.exists(fuser_path):
        fuser.load_state_dict(torch.load(fuser_path, map_location=device))
        print(f"Loaded {fuser_path}")
    fuser.to(device).eval()

    # Load Flow
    flow = ConditionalFlowMatching(hidden_dim=256, out_dim=80)
    flow_path = os.path.join(args.checkpoint_dir, "flow_ep25.pt")
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"Loaded {flow_path}")
    flow.to(device).eval()
    
    return encoder, factorizer, fuser, flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/microencoder_e2e")
    parser.add_argument("--wav_path", default="data/audio/1034_1034_121119_000010_000004.wav")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder, factorizer, fuser, flow = load_models(args, device)
    
    # Load Audio
    wav, sr = torchaudio.load(args.wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.to(device)
    if wav.shape[0] > 1:
        wav = wav[:1, :] # Mono
    
    # Trim to 3s for speed
    total_len = wav.shape[1]
    if total_len > 16000*3:
        wav = wav[:, :16000*3]
    
    with torch.no_grad():
        # GT Mels (for reference)
        # Using simple mel transform as in training but simplified
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=600, # train_e2e_encoder default
            hop_length=120, # train_e2e_encoder default
            n_mels=80
        ).to(device)
        
        gt_mel = mel_transform(wav)
        gt_mel = torch.log(torch.clamp(gt_mel, min=1e-5))
        # Norm
        MEAN = -6.0 # Approx
        STD = 2.5   # Approx
        gt_mel = (gt_mel - MEAN) / STD
        gt_mel = gt_mel.permute(0, 2, 1) # B, T, C
        
        # Forward pass to get conditioning
        features = encoder(wav.unsqueeze(1)) # B, 1, T -> features
        
        # Factorizer
        main_info, remaining, _, _ = factorizer(features)
        
        # Fuser
        cond = fuser(main_info, remaining)
        
        # Flow Inference with different CFG scales
        scales = [1.0, 1.5, 2.5]
        results = []
        
        print("Generating samples...")
        for s in scales:
            pred = flow.solve_ode(cond, steps=50, solver='rk4', cfg_scale=s)
            results.append(pred)
            print(f"Generated with CFG={s}")
            
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # GT
    plt.subplot(len(scales)+1, 1, 1)
    plt.imshow(gt_mel[0].T.cpu().numpy(), origin='lower', aspect='auto', interpolation='nearest', vmin=-2, vmax=2)
    plt.title("Ground Truth Mel")
    plt.colorbar()
    
    for i, (s, res) in enumerate(zip(scales, results)):
        plt.subplot(len(scales)+1, 1, i+2)
        plt.imshow(res[0].T.cpu().numpy(), origin='lower', aspect='auto', interpolation='nearest', vmin=-2, vmax=2)
        plt.title(f"Generated Mel (CFG={s})")
        plt.colorbar()
        
    plt.tight_layout()
    plt.savefig("cfg_test_comparison.png")
    print("Saved cfg_test_comparison.png")

if __name__ == "__main__":
    main()
