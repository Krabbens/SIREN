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
        
        sem_tokens = config['model']['semantic']['vocab_size']
        pro_tokens = config['model']['prosody']['vocab_size']
        context_len = config['model']['entropy'].get('context_length', 4096)
        # Force large context if not specified sufficient
        if context_len < 4096: context_len = 4096
        
        self.sem_lm = ProbabilisticLM(sem_tokens, 
                                     config['model']['entropy']['lm_dim'],
                                     config['model']['entropy']['lm_layers'],
                                     config['model']['entropy']['lm_heads'],
                                     max_seq_len=context_len)
                                     
        self.pro_lm = ProbabilisticLM(pro_tokens, 
                                     config['model']['entropy']['lm_dim'],
                                     config['model']['entropy']['lm_layers'],
                                     config['model']['entropy']['lm_heads'],
                                     max_seq_len=context_len)
                                     
    def forward(self, sem_idx, pro_idx):
        # sem_idx: (B, T_s)
        # pro_idx: (B, T_p)
        
        sem_logits = self.sem_lm(sem_idx) # P(t_i | t_<i)
        pro_logits = self.pro_lm(pro_idx)
        
        return sem_logits, pro_logits
        
    def estimate_bits(self, sem_idx, pro_idx):
        """
        Calculates theoretical bitrate (Cross Entropy)
        """
        sem_logits, pro_logits = self(sem_idx[:, :-1], pro_idx[:, :-1])
        
        # Target is next token
        sem_target = sem_idx[:, 1:]
        pro_target = pro_idx[:, 1:]
        
        sem_nll = F.cross_entropy(sem_logits.transpose(1, 2), sem_target, reduction='none')
        pro_nll = F.cross_entropy(pro_logits.transpose(1, 2), pro_target, reduction='none')
        
        # Sum bits per sequence
        sem_bits = sem_nll.sum(dim=1) / 0.693147 # nats to bits
        pro_bits = pro_nll.sum(dim=1) / 0.693147
        
        return sem_bits, pro_bits
