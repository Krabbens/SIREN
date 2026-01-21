import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticLM(nn.Module):
    def __init__(self, num_tokens, dim, depth=2, heads=4, max_seq_len=4096):
        super().__init__()
        self.num_tokens = num_tokens
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        
        self.head = nn.Linear(dim, num_tokens)
        
    def forward(self, x):
        """
        x: (B, T) indices
        Returns: logits (B, T, V)
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).expand(B, T)
        
        h = self.token_emb(x) + self.pos_emb(positions)
        
        # Causal Mask
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        h = self.transformer(h, mask=mask)
        logits = self.head(h)
        return logits

# ... (forward remains same)

class EntropyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # We process inputs as bytes (0-255), so vocab is fixed
        vocab_size = 256
        
        context_len = config['model']['entropy'].get('context_length', 4096)
        # Force large context if not specified sufficient
        if context_len < 4096: context_len = 4096
        
        self.sem_lm = ProbabilisticLM(vocab_size, 
                                     config['model']['entropy']['lm_dim'],
                                     config['model']['entropy']['lm_layers'],
                                     config['model']['entropy']['lm_heads'],
                                     max_seq_len=context_len)
                                     
        self.pro_lm = ProbabilisticLM(vocab_size, 
                                     config['model']['entropy']['lm_dim'],
                                     config['model']['entropy']['lm_layers'],
                                     config['model']['entropy']['lm_heads'],
                                     max_seq_len=context_len)
                                     
    def indices_to_bytes(self, indices):
        """
        Convert large indices (up to 24-bit) into sequence of 3 bytes.
        indices: (B, T)
        Returns: (B, T*3)
        """
        # 24-bit support (16M vocab)
        b1 = (indices >> 16) & 0xFF
        b2 = (indices >> 8) & 0xFF
        b3 = indices & 0xFF
        
        # Stack and flatten: (B, T, 3) -> (B, T*3)
        bytes_seq = torch.stack([b1, b2, b3], dim=-1).view(indices.shape[0], -1)
        return bytes_seq.long()

    def forward(self, sem_idx, pro_idx):
        # Flatten input to (B, N) first
        sem_flat = sem_idx.view(sem_idx.shape[0], -1)
        pro_flat = pro_idx.view(pro_idx.shape[0], -1)
        
        # Convert to bytes
        sem_bytes = self.indices_to_bytes(sem_flat)
        pro_bytes = self.indices_to_bytes(pro_flat)
        
        # Run LM
        sem_logits = self.sem_lm(sem_bytes) 
        pro_logits = self.pro_lm(pro_bytes)
        
        return sem_logits, pro_logits
        
    def estimate_bits(self, sem_idx, pro_idx):
        """
        Calculates theoretical bitrate (Cross Entropy) using byte-level modeling.
        """
        # Helper to calc bits for a stream
        def calc_stream_bits(model, indices):
            # Flatten to (B, Seq)
            flat = indices.view(indices.shape[0], -1)
            # To bytes (B, Seq*3)
            bytes_seq = self.indices_to_bytes(flat)
            
            # Forward (predict next byte)
            # causal: predict [1:] from [:-1]
            logits = model(bytes_seq[:, :-1])
            targets = bytes_seq[:, 1:]
            
            # Loss per byte
            nll = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
            
            # Sum nll over sequence -> total nats
            total_nats = nll.sum(dim=1)
            
            # Convert to bits
            total_bits = total_nats / 0.693147
            return total_bits

        sem_bits = calc_stream_bits(self.sem_lm, sem_idx)
        pro_bits = calc_stream_bits(self.pro_lm, pro_idx)
        
        return sem_bits, pro_bits
