# Save as: lstm_basics.py

import torch
import torch.nn as nn

class ScalarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # q,k,v projections
        self.W_q = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_k = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_v = nn.Linear(input_size + hidden_size, hidden_size)
    
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size) 
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size) 
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size) 
      
        
        self.W_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size)
            C = torch.zeros(batch_size, self.hidden_size, self.hidden_size)
            n = torch.zeros(batch_size, self.hidden_size)  
            m = torch.zeros(batch_size, self.hidden_size)
        else:
            h, C, n , m = state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h], dim=1)


            # 1. Calculate Q, K, V and Log-Gates
            q = self.W_q(combined)
            k = self.W_k(combined) / (self.hidden_size ** 0.5) # Scale K is good practice
            v = self.W_v(combined)
            
            # Concatenate input and hidden
           
            
            f_log = self.W_f(combined)
            i_log = self.W_i(combined)
            m_prev = m
            m = torch.max(f_log + m_prev, i_log)

            i_t = torch.exp(i_log - m).unsqueeze(2)
            f_t = torch.exp(f_log + m_prev - m).unsqueeze(2)
            o_t = torch.sigmoid(self.W_o(combined)) 
            kv_product = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))


            C = f_t * C + i_t * kv_product
            n = f_t.squeeze(2) * n + i_t.squeeze(2) * k
            
            num = torch.bmm(C, q.unsqueeze(2)).squeeze(2)
            den = torch.abs((n * q).sum(dim=1, keepdim=True)) + 1e-8
            h = o_t * (num / den)

            # Output
            output = self.W_out(h)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, C, n, m)


if __name__ == "__main__": 
    lstm = ScalarLSTM(input_size=10, hidden_size=128, output_size=10)
    x_long = torch.randn(1, 100, 10)
    outputs, (h, C, n, m) = lstm(x_long)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden shape:  {h.shape}")
    print(f"Cell shape: {C.shape}")
    print(f"Normalizer shape: {n.shape}")
    print(f"Stabilizer shape: {m.shape}")