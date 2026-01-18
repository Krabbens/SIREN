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
    """Dataset for precomputed DistilHuBERT features with on-the-fly audio loading."""
    
    def __init__(
        self,
        features_dir: str,
        max_frames: int = 400,
        sample_rate: int = 16000,
        hop_length: int = 320,
        manifest_path: Optional[str] = None,
    ):
        """
        Args:
            features_dir: Directory containing precomputed .pt feature files
            max_frames: Maximum number of frames to use
            sample_rate: Audio sample rate
            hop_length: Hop length used for feature extraction
            manifest_path: Optional path to manifest JSON for multi-speaker datasets
        """
        self.features_dir = Path(features_dir)
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Find all feature files
        self.feature_files = sorted(list(self.features_dir.glob("*.pt")))
        
        if len(self.feature_files) == 0:
            raise ValueError(f"No .pt files found in {features_dir}")
        
        # Load manifest if provided (for multi-speaker)
        self.manifest = None
        self.audio_paths = {}
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            # Build audio path lookup
            for entry in self.manifest:
                feature_name = Path(entry.get('feature_path', '')).stem
                if feature_name:
                    self.audio_paths[feature_name] = entry.get('audio_path', '')
        
        print(f"Loaded {len(self.feature_files)} feature files from {features_dir}")
    
    def __len__(self) -> int:
        return len(self.feature_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feature_path = self.feature_files[idx]
        
        # Load precomputed features
        data = torch.load(feature_path, map_location='cpu', weights_only=True)
        
        # Handle different saved formats
        if isinstance(data, dict):
            features = data.get('features', data.get('hidden_states', None))
            audio = data.get('audio', None)
            audio_path = data.get('audio_path', None)
        else:
            features = data
            audio = None
            audio_path = None
        
        if features is None:
            raise ValueError(f"Could not find features in {feature_path}")
        
        # Ensure features are 2D [time, features]
        if features.dim() == 3:
            features = features.squeeze(0)
        
        # Load audio from audio_path if audio tensor not present
        if audio is None and audio_path is not None:
            audio = self._load_audio_from_path(audio_path)
        
        # Truncate or pad to max_frames
        num_frames = features.shape[0]
        start = 0
        if num_frames > self.max_frames:
            # Random crop features
            start = random.randint(0, num_frames - self.max_frames)
            features = features[start:start + self.max_frames]
        elif num_frames < self.max_frames:
            # Pad
            pad_frames = self.max_frames - num_frames
            features = F.pad(features, (0, 0, 0, pad_frames))
        
        # Audio loading optimization: Load only what we need
        if audio is None:
            # Calculate audio samples range
            audio_start = start * self.hop_length
            audio_len = self.max_frames * self.hop_length
            
            # Use audio_path if available or fallback to lookup
            target_path = audio_path
            if target_path is None:
                # Find path logic...
                feature_name = feature_path.stem
                if feature_name in self.audio_paths:
                    target_path = self.audio_paths[feature_name]
                else:
                    audio_dir = self.features_dir.parent
                    for ext in ['.wav', '.flac', '.mp3', '.m4a']:
                        p = audio_dir / f"{feature_name}{ext}"
                        if p.exists():
                            target_path = str(p)
                            break
            
            if target_path:
                audio = self._load_audio_from_path(target_path, offset=audio_start, num_frames=audio_len)

        # Fallback if partial load failed or wasn't tried
        if audio is None:
             # Try seeking logic again inside _load_audio_for_feature (if we updated it, but we didn't)
             # Fallback to full load (old behavior) if optimize fails
             audio = self._load_audio_for_feature(feature_path)
        
        # Ensure audio length matches features
        expected_audio_len = self.max_frames * self.hop_length
        if audio is not None:
            if audio.shape[-1] > expected_audio_len:
                audio = audio[..., :expected_audio_len]
            elif audio.shape[-1] < expected_audio_len:
                audio = F.pad(audio, (0, expected_audio_len - audio.shape[-1]))
            
            # Ensure audio is 1D
            if audio.dim() > 1:
                audio = audio.squeeze(0)
        else:
            # Create dummy audio if loading failed
            audio = torch.zeros(expected_audio_len)
        
        return {
            'features': features,  # [max_frames, feature_dim]
            'audio': audio,        # [max_frames * hop_length]
        }
    
    def _load_audio_from_path(self, audio_path: str, offset: int = 0, num_frames: int = None) -> Optional[torch.Tensor]:
        """Load audio from a specific path using soundfile with seeking."""
        if not os.path.exists(audio_path):
            return None
        try:
            # Use soundfile for reliable seeking
            # Calculate sample offset based on hop_length context usually
            # Here inputs are in 'samples' domain mostly if called correctly
            # But let's assume offset/num_frames are in samples
            
            with sf.SoundFile(audio_path) as f:
                if offset > f.frames:
                    offset = 0
                f.seek(offset)
                
                frames_to_read = -1
                if num_frames is not None:
                    frames_to_read = num_frames
                    
                data = f.read(frames_to_read)
                sr = f.samplerate
                
            waveform = torch.from_numpy(data).float()
            
            if sr != self.sample_rate:
                # Note: Resampling a small chunk might have boundary artifacts
                # but for training randomness it's usually acceptable or we fetch a bit more context
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Ensure 1D
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=-1)  # Convert stereo to mono
                
            return waveform
        except Exception as e:
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return None
    
    def _load_audio_for_feature(self, feature_path: Path) -> Optional[torch.Tensor]:
        """Try to load corresponding audio file."""
        feature_name = feature_path.stem
        
        # Check manifest first
        if feature_name in self.audio_paths:
            audio_path = self.audio_paths[feature_name]
            if os.path.exists(audio_path):
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                    return waveform.squeeze(0)
                except Exception:
                    pass
        
        # Try common audio extensions
        audio_dir = self.features_dir.parent
        for ext in ['.wav', '.flac', '.mp3', '.m4a']:
            audio_path = audio_dir / f"{feature_name}{ext}"
            if audio_path.exists():
                try:
                    waveform, sr = torchaudio.load(str(audio_path))
                    if sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                    return waveform.squeeze(0)
                except Exception:
                    continue
        
        return None


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
            if not os.path.exists(feat_dir):
                print(f"Warning: {feat_dir} does not exist, skipping")
                continue
                
            ds = PrecomputedFeatureDataset(
                features_dir=feat_dir,
                manifest_path=manifest,
                max_frames=max_frames,
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            self.datasets.append(ds)
            
            # Determine language from path
            lang = 'en' if 'english' in feat_dir.lower() or 'libri' in feat_dir.lower() else 'pl'
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
