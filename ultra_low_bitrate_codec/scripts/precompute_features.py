"""
Precompute DistilHuBERT features for faster training.
Saves features as .pt files to disk.
"""
import os
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import json
import argparse

def precompute_features(manifest_path, output_dir, model_name="ntu-spml/distilhubert", target_layer=4, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading {model_name}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        files = json.load(f)
    paths = [d['path'] if isinstance(d, dict) else d for d in files]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(paths)} files...")
    
    for path in tqdm(paths):
        try:
            # Load audio
            audio, sr = torchaudio.load(path)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            
            audio = audio.squeeze(0)
            
            # Extract features
            with torch.no_grad():
                inputs = audio.unsqueeze(0).to(device)
                outputs = model(inputs, output_hidden_states=True)
                # DistilHuBERT has 6 transformer layers (7 hidden states total including embedding)
                # Use the last layer for best features
                features = outputs.hidden_states[-1]  # (1, T, 768)
            
            # Save
            basename = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{basename}.pt")
            torch.save(features.squeeze(0).cpu(), out_path)  # (T, 768)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"Done! Features saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='/home/sperm/diff/data/train.json')
    parser.add_argument('--output', type=str, default='/home/sperm/diff/data/features')
    parser.add_argument('--model', type=str, default='ntu-spml/distilhubert')
    parser.add_argument('--layer', type=int, default=4)
    args = parser.parse_args()
    
    precompute_features(args.manifest, args.output, args.model, args.layer)
