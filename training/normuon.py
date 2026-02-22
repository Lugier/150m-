"""
NorMuon Optimizer (Neuron-wise Normalized Muon) + Hybrid wrapper
Plan §6: "NorMuon-AdamW-Hybrid (NorMuon für 2D-Matrizen, AdamW für 1D-Parameter)"
"""

import torch
from torch.optim import Optimizer, AdamW

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # We execute Newton-Schulz optimally in bfloat16 to avoid precision issues
    # and to exploit tensor cores.
    # Fallback to float32 on devices that don't support bfloat16.
    dtype = torch.bfloat16 if G.device.type != "mps" and torch.cuda.is_bf16_supported() else torch.float32
    X = G.to(dtype)
    
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(G.dtype)

class NorMuon(Optimizer):
    """
    NorMuon: Neuron-wise Normalized Muon optimizer.
    For 2D matrices only. We use Newton-Schulz for orthogonalization.
    Neuron-wise normalized: The orthogonalized update is normalized per neuron (row).
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.ndim != 2:
                    raise RuntimeError("NorMuon only supports 2D parameter tensors.")
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Momentum step
                buf.mul_(momentum).add_(grad)
                
                # Orthogonalize
                update = zeropower_via_newtonschulz5(buf, steps=5)
                
                # Neuron-wise normalization
                row_norms = update.norm(dim=1, keepdim=True) + 1e-7
                update = update / row_norms
                
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                
                p.add_(update, alpha=-lr)
                
        return loss

class HybridNorMuonAdamW(Optimizer):
    """
    Wrapper to use NorMuon for 2D params and AdamW for 1D params.
    """
    def __init__(self, model: torch.nn.Module, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), normuon_momentum=0.95):
        decay_2d = []
        decay_1d = []
        no_decay_1d = []
        
        no_decay_names = {"bias", "LayerNorm.weight", "norm.weight", "embed.weight"}
        
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_2d.append(p)
            else:
                if any(nd in n for nd in no_decay_names):
                    no_decay_1d.append(p)
                else:
                    decay_1d.append(p)
                    
        self.normuon = NorMuon([{'params': decay_2d, 'weight_decay': weight_decay}], lr=lr, momentum=normuon_momentum)
        
        adam_groups = []
        if decay_1d:
            adam_groups.append({'params': decay_1d, 'weight_decay': weight_decay})
        if no_decay_1d:
            adam_groups.append({'params': no_decay_1d, 'weight_decay': 0.0})
            
        self.adamw = AdamW(adam_groups, lr=lr, betas=betas)
        
        # Combine param_groups so LR schedulers can iterate over them uniformly
        self.param_groups = self.normuon.param_groups + self.adamw.param_groups
        
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("Closure not supported for HybridNorMuonAdamW")
        self.normuon.step()
        self.adamw.step()
        
    def zero_grad(self, set_to_none=True):
        self.normuon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
        
    def state_dict(self):
        return {
            'normuon': self.normuon.state_dict(),
            'adamw': self.adamw.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        self.normuon.load_state_dict(state_dict['normuon'])
        self.adamw.load_state_dict(state_dict['adamw'])
