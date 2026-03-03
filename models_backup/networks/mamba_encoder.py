import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaEncoder(nn.Module):
    """
    Mamba-based sequence encoder for multimodal emotion recognition
    Can replace both LSTMEncoder and TextCNN
    """
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False, num_layers=1):
        super(MambaEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        assert embd_method in ['maxpool', 'attention', 'last', 'mean']
        self.embd_method = embd_method
        
        # Input projection to match Mamba's expected dimension
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Mamba layers
        if bidirectional:
            # Bidirectional: forward + backward
            self.mamba_forward = nn.ModuleList([
                Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            self.mamba_backward = nn.ModuleList([
                Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            final_dim = hidden_size * 2
        else:
            # Unidirectional
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            final_dim = hidden_size
        
        # Attention mechanism for embd_method='attention'
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(final_dim, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
            nn.init.xavier_uniform_(self.attention_vector_weight)
        
        # Output projection to ensure output size = hidden_size
        if bidirectional:
            self.output_proj = nn.Linear(final_dim, hidden_size)
        else:
            self.output_proj = None

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            embd: [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_size]
        
        if self.bidirectional:
            # Forward pass
            h_forward = x
            for mamba_layer in self.mamba_forward:
                h_forward = mamba_layer(h_forward)
            
            # Backward pass (reverse sequence)
            h_backward = torch.flip(x, dims=[1])
            for mamba_layer in self.mamba_backward:
                h_backward = mamba_layer(h_backward)
            h_backward = torch.flip(h_backward, dims=[1])
            
            # Concatenate forward and backward
            h_out = torch.cat([h_forward, h_backward], dim=-1)  # [batch_size, seq_len, hidden_size*2]
        else:
            # Unidirectional pass
            h_out = x
            for mamba_layer in self.mamba_layers:
                h_out = mamba_layer(h_out)
        
        # Apply embedding method
        if self.embd_method == 'last':
            embd = h_out[:, -1, :]  # [batch_size, hidden_size or hidden_size*2]
        elif self.embd_method == 'maxpool':
            embd = torch.max(h_out, dim=1)[0]  # [batch_size, hidden_size or hidden_size*2]
        elif self.embd_method == 'mean':
            embd = torch.mean(h_out, dim=1)  # [batch_size, hidden_size or hidden_size*2]
        elif self.embd_method == 'attention':
            embd = self.embd_attention(h_out)  # [batch_size, hidden_size or hidden_size*2]
        
        # Project back to hidden_size if bidirectional
        if self.bidirectional and self.output_proj is not None:
            embd = self.output_proj(embd)
        
        return embd
    
    def embd_attention(self, h_out):
        """Attention-based embedding"""
        hidden_reps = self.attention_layer(h_out)  # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)  # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)
        sentence_vector = torch.sum(h_out * atten_weight, dim=1)  # [batch_size, hidden_size]
        return sentence_vector