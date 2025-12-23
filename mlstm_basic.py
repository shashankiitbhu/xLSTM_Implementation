import torch
import torch.nn as nn

class MatrixLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        self.W_f = nn.Linear(input_size, hidden_size) 
        self.W_i = nn.Linear(input_size, hidden_size) 
        self.W_o = nn.Linear(input_size, hidden_size) 
        
        self.W_out = nn.Linear(hidden_size, output_size)
        

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.size()
        
        # --- PRE-COMPUTE STEP ---
        # Since gates/values don't depend on 'h', we can compute them 
        # for all timesteps at once! This is much faster.
        
        q = self.W_q(x)
        k = self.W_k(x) / (self.hidden_size ** 0.5)
        v = self.W_v(x)
        
        f_log = self.W_f(x)
        i_log = self.W_i(x)
        o_log = self.W_o(x)
        o = torch.sigmoid(o_log) 
        
        if state is None:
            C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=x.device)
            n = torch.zeros(batch_size, self.hidden_size, 1, device=x.device)  
            m = torch.zeros(batch_size, self.hidden_size, 1, device=x.device)
        else:
            C, n, m = state
            
        outputs = []
        
        for t in range(seq_len):
            q_t = q[:, t, :].unsqueeze(2)
            k_t = k[:, t, :].unsqueeze(2)
            v_t = v[:, t, :].unsqueeze(2)
            
            f_log_t = f_log[:, t, :].unsqueeze(2)
            i_log_t = i_log[:, t, :].unsqueeze(2)
            o_t = o[:, t, :].unsqueeze(2)

            m_prev = m
            m = torch.max(f_log_t + m_prev, i_log_t)

            i_t = torch.exp(i_log_t - m)
            f_t = torch.exp(f_log_t + m_prev - m)
            
            kv_product = torch.bmm(v_t, k_t.transpose(1, 2))
            C = f_t * C + i_t * kv_product
            
            n = f_t * n + i_t * k_t
            
            # Retrieve Hidden State (h)
            c_q = torch.bmm(C, q_t)
            
            n_q = torch.abs(torch.bmm(n.transpose(1, 2), q_t)) + 1e-8
            
            # Final Hidden State
            h = o_t * (c_q / n_q)
            h = h.squeeze(2)
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        final_output = self.W_out(outputs)
        
        return final_output, (C, n, m)

if __name__ == "__main__": 
    lstm = MatrixLSTM(input_size=10, hidden_size=32, output_size=10)
    x_long = torch.randn(1, 50, 10)
    outputs, (C, n, m) = lstm(x_long)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Matrix State shape: {C.shape}")