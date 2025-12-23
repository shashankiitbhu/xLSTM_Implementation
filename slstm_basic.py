# Save as: lstm_basics.py

import torch
import torch.nn as nn

class ScalarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
    
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size) 
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size) 
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size) 
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size) 
      
        
        self.W_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size)
            c = torch.zeros(batch_size, self.hidden_size)
            n = torch.zeros(batch_size, self.hidden_size)  
            m = torch.zeros(batch_size, self.hidden_size)
        else:
            h, c, n , m = state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden
            combined = torch.cat([x_t, h], dim=1)
            
            f_log = self.W_f(combined)
            i_log = self.W_i(combined)
            m_prev = m
            m = torch.max(f_log + m_prev, i_log)

            i_t = torch.exp(i_log - m)
            f_t = torch.exp(f_log + m_prev - m)
            c_hat_t = torch.tanh(self.W_c(combined)) 
            o_t = torch.sigmoid(self.W_o(combined)) 


            c = f_t * c + i_t * c_hat_t
            n = f_t * n + i_t 
            
            h = o_t * (c / (n + 1e-8))
            
            # Output
            output = self.W_out(h)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c, n, m)


if __name__ == "__main__": 
    lstm = ScalarLSTM(input_size=10, hidden_size=128, output_size=10)
    x_long = torch.randn(1, 100, 10)
    outputs, (h, c, n, m) = lstm(x_long)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden shape:  {h.shape}")
    print(f"Cell shape: {c.shape}")
    print(f"Normalizer shape: {n.shape}")
    print(f"Stabilizer shape: {m.shape}")