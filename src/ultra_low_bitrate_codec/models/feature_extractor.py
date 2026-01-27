import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torchaudio

class HubertFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/hubert-base-ls960", target_layer=9, freeze=True):
        """
        Extracts features from a pretrained HuBERT model.
        
        Args:
            model_name (str): HuggingFace model identifier.
            target_layer (int): Which transformer layer output to use.
            freeze (bool): Whether to freeze the model weights.
        """
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.target_layer = target_layer
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, audio, sample_rate=16000):
        """
        Args:
            audio (torch.Tensor): Audio waveform of shape (B, T)
            sample_rate (int): Sample rate of input audio (must be 16000)
            
        Returns:
            torch.Tensor: Features of shape (B, Frames, HiddenDim)
        """
        # Ensure input is on sound device
        device = audio.device
        
        # Audio normalization (optional but recommended for wav2vec/hubert)
        # Assuming audio is float32 usually in [-1, 1]
        
        # We process manually to return hidden states
        # The transformers processor usually does padding/norm, but we handle tensors directly
        
        # Forward pass
        # output_hidden_states=True allows us to grab specific layers
        with torch.no_grad():
            outputs = self.model(
                audio, 
                output_hidden_states=True,
                return_dict=True
            )
            
        # Extact specific layer
        # outputs.hidden_states is a tuple of (input_ops, layer_0, ..., layer_11)
        # So index 0 is embeddings, index 1 is layer 1, etc.
        # We usually want the layer representations.
        # Note: transformers hubert base has 12 layers.
        # range is 0-12 (13 elements total).
        feat = outputs.hidden_states[self.target_layer]
        
        return feat

if __name__ == "__main__":
    # Simple test
    extractor = HubertFeatureExtractor()
    dummy_audio = torch.randn(1, 16000) # 1 sec
    feats = extractor(dummy_audio)
    print(f"Input: {dummy_audio.shape}, Output: {feats.shape}")
