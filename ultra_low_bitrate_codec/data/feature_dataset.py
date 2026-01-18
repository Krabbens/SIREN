"""Feature dataset for precomputed DistilHuBERT features."""

import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
import numpy as np


class PrecomputedFeatureDataset(Dataset):
    """
    Optimized Feature Dataset using static manifest.
    Eliminates all runtime file system checks.
    """
    
    def __init__(
        self,
        features_dir: str = None, # For backward compatibility (ignored if manifest used correctly)
        max_frames: int = 400,
        sample_rate: int = 16000,
        hop_length: int = 320,
        manifest_path: Optional[str] = None,
    ):
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Load optimized manifest
        self.entries = []
        if manifest_path and os.path.exists(manifest_path):
            print(f"Loading optimized manifest: {manifest_path}")
            with open(manifest_path, 'r') as f:
                self.entries = json.load(f)
        else:
            # Fallback (legacy mode - should ideally not be reached in optimized flow)
            print("WARNING: No optimized manifest provided. Falling back to slow legacy mode.")
            features_dir = Path(features_dir)
            files = sorted(list(features_dir.glob("*.pt")))
            for f in files:
                self.entries.append({"feature_path": str(f), "audio_path": None, "legacy": True})
        
        print(f"Dataset loaded with {len(self.entries)} samples")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        feature_path = entry['feature_path']
        audio_path = entry.get('audio_path')
        
        # 1. Load Features
        try:
            data = torch.load(feature_path, map_location='cpu', weights_only=True)
            if isinstance(data, dict):
                features = data.get('features', data.get('hidden_states'))
            else:
                features = data
                
            if features.dim() == 3:
                features = features.squeeze(0)
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            features = torch.zeros(100, 768) # Dummy return
            
        # 2. Duration/Padding logic
        num_frames = features.shape[0]
        start_frame = 0
        
        if num_frames > self.max_frames:
            start_frame = random.randint(0, num_frames - self.max_frames)
            features = features[start_frame:start_frame + self.max_frames]
        elif num_frames < self.max_frames:
            pad_frames = self.max_frames - num_frames
            features = F.pad(features, (0, 0, 0, pad_frames))
            
        # 3. Load Audio
        audio = None
        if audio_path and os.path.exists(audio_path):
            try:
                # Calculate sample range
                start_sample = start_frame * self.hop_length
                num_samples = self.max_frames * self.hop_length
                
                # seeking with soundfile is fast
                info = sf.info(audio_path)
                
                if start_sample < info.frames:
                    # Avoid reading past end (though soundfile handles this, explicit is safer)
                    read_frames = min(num_samples, info.frames - start_sample)
                    
                    with sf.SoundFile(audio_path) as f:
                        f.seek(start_sample)
                        audio_np = f.read(frames=read_frames, dtype='float32') # Direct float32 read
                        
                    audio = torch.from_numpy(audio_np)
                    
                    # Mono
                    if audio.dim() > 1:
                        audio = audio.mean(dim=1)
                        
                    # Resample if needed (should be rare if pre-verified)
                    if info.samplerate != self.sample_rate:
                        audio = torchaudio.functional.resample(audio, info.samplerate, self.sample_rate)
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
        
        # 4. Final Padding/Constraint
        expected_len = self.max_frames * self.hop_length
        if audio is None:
            audio = torch.zeros(expected_len)
        else:
            if audio.shape[0] < expected_len:
                audio = F.pad(audio, (0, expected_len - audio.shape[0]))
            elif audio.shape[0] > expected_len:
                audio = audio[:expected_len]
                
        return {
            'features': features,
            'audio': audio
        }


class MultiSpeakerFeatureDataset(Dataset):
    """Dataset combining multiple speaker datasets with language balancing."""
    
    def __init__(
        self,
        features_dirs: List[str],
        manifests: Optional[List[str]] = None,
        max_frames: int = 400,
        sample_rate: int = 16000,
        hop_length: int = 320,
        language_balance: bool = True,
    ):
        """
        Args:
            features_dirs: List of directories containing precomputed features
            manifests: Optional list of manifest paths
            max_frames: Maximum number of frames
            sample_rate: Audio sample rate  
            hop_length: Hop length
            language_balance: Whether to balance sampling across languages
        """
        self.datasets = []
        self.language_indices = {}  # language -> list of (dataset_idx, sample_idx)
        
        manifests = manifests or [None] * len(features_dirs)
        
        for i, (feat_dir, manifest) in enumerate(zip(features_dirs, manifests)):
            if not os.path.exists(feat_dir) and not manifest:
                print(f"Warning: {feat_dir} does not exist and no manifest provided, skipping")
                continue
                
            ds = PrecomputedFeatureDataset(
                features_dir=feat_dir,
                manifest_path=manifest,
                max_frames=max_frames,
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            self.datasets.append(ds)
            
            # Determine language from path or manifest
            # Simple heuristic
            lang = 'en'
            if 'pl' in str(feat_dir).lower() or (manifest and 'pl' in manifest):
                lang = 'pl'
                
            if lang not in self.language_indices:
                self.language_indices[lang] = []
            
            for j in range(len(ds)):
                self.language_indices[lang].append((len(self.datasets) - 1, j))
        
        self.language_balance = language_balance
        self.languages = list(self.language_indices.keys())
        
        # Compute total length
        self._total_length = sum(len(ds) for ds in self.datasets)
        
        # Build flat index
        self._flat_index = []
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                self._flat_index.append((ds_idx, sample_idx))
        
        print(f"MultiSpeakerDataset: {self._total_length} samples, languages: {self.languages}")
    
    def __len__(self) -> int:
        return self._total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.language_balance and len(self.languages) > 1:
            # Randomly select language then sample
            lang = random.choice(self.languages)
            ds_idx, sample_idx = random.choice(self.language_indices[lang])
        else:
            ds_idx, sample_idx = self._flat_index[idx]
        
        return self.datasets[ds_idx][sample_idx]
