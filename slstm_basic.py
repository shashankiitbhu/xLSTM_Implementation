# Save as: lstm_basics.py

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        #input_size=10, hidden_size=128, output_size=10
        # All 4 gates (forget, input, cell, output)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # Cell
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output
      
        
        self.W_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size)
            c = torch.zeros(batch_size, self.hidden_size)
            n = torch.zeros(batch_size, self.hidden_size)  
        else:
            h, c, n = state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden
            combined = torch.cat([x_t, h], dim=1)
            
            # Gate activations
            f_t = torch.exp(self.W_f(combined))  # Forget gate
            i_t = torch.exp(self.W_i(combined))  # Input gate
            c_hat_t = torch.tanh(self.W_c(combined))  # Candidate cell
            o_t = torch. sigmoid(self.W_o(combined))  # Output gate
            
            # Update cell state (LONG-TERM MEMORY)
            c = f_t * c + i_t * c_hat_t
            n = f_t * n + i_t 
            
            # Update hidden state (SHORT-TERM MEMORY)
            h = o_t * (c/n)
            
            # Output
            output = self.W_out(h)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c, n)


if __name__ == "__main__": 
    lstm = SimpleLSTM(input_size=10, hidden_size=128, output_size=10)
    
    # EXPERIMENT 1: Long sequence memory
    x_long = torch.randn(1, 100, 10)
    outputs, (h, c, n) = lstm(x_long)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden shape:  {h.shape}")
    print(f"Cell shape: {c.shape}")
    
    # EXPERIMENT 2: Gradient flow (vs RNN)
    loss = c.sum()
    loss.backward()
    grad_norm = lstm.W_f.weight.grad.norm()
    print(f"\nLSTM gradient (100 steps): {grad_norm:.4f}")
    # Compare to RNN: Should be MUCH larger (better gradient flow)
    
    # EXPERIMENT 3: PyTorch built-in LSTM (use this in practice)
    lstm_pytorch = nn.LSTM(input_size=10, hidden_size=128, batch_first=True)
    outputs_pt, (h_pt, c_pt) = lstm_pytorch(x_long)
    print(f"\nPyTorch LSTM output:  {outputs_pt.shape}")