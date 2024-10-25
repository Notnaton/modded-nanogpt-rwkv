"""
RWKV7 implementation in tinygrad with training code.
Includes model architecture, Muon optimizer, and training/inference utilities.
"""

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm, GroupNorm, Embedding
from tinygrad import dtypes
from tinygrad import TinyJit
from tinygrad.nn.optim import Adam
import numpy as np
import os
import glob
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import time
import math

@dataclass
class TrainingConfig:
    # Model config
    vocab_size: int = 50304  # Extended GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    head_size: int = 64
    
    # Data config
    batch_size: int = 512  # Global batch size
    sequence_length: int = 1024
    
    # Training config
    num_iterations: int = 5100
    muon_lr: float = 0.00036
    adam_lr: float = 0.0022
    emb_scale: float = 2.0
    warmup_iters: int = 0
    warmdown_iters: int = 1450
    
    # Paths
    train_data_path: str = "data/fineweb10B/fineweb_train_*.bin"
    val_data_path: str = "data/fineweb10B/fineweb_val_*.bin"
    
    # Evaluation config
    val_interval: int = 125
    val_tokens: int = 10485760

class RWKV7:
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int = 64):
        self.n_embd = n_embd
        self.n_head = n_embd // head_size 
        assert n_embd % self.n_head == 0
        self.head_size = head_size
        
        # Calculate position-specific initialization values
        ratio_0_to_1 = layer_id / (n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
        
        # Initialize time mixing attention
        ddd = Tensor.ones(1, 1, n_embd)
        for i in range(n_embd):
            ddd.realize()[0,0,i] = i / n_embd
            
        self.time_maa_x = Tensor.ones(1, 1, n_embd) - ddd.pow(0.6 * ratio_1_to_almost0 ** 0.9)
        self.time_maa_rg = Tensor.ones(1, 1, n_embd) - ddd.pow(0.2 * ratio_1_to_almost0)
        self.time_maa_wa = Tensor.ones(1, 1, n_embd) - ddd.pow(0.9 * ratio_1_to_almost0)
        self.time_maa_k = Tensor.ones(1, 1, n_embd) - (ddd.pow(0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
        self.time_maa_v = Tensor.ones(1, 1, n_embd) - (ddd.pow(0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)

        # Initialize decay rate
        decay_speed = Tensor.ones(n_embd)
        for n in range(n_embd):
            decay_speed.realize()[n] = -7 + 5 * (n / (n_embd - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            
        self.time_decay = decay_speed.reshape(1,1,n_embd) + 0.5
        
        self.time_faaaa = Tensor.zeros(1, 1, self.n_head, head_size)
        self.time_aaaaa = Tensor.zeros(1, 1, n_embd)

        # Initialize all the LoRA weights
        D_MIX_LORA = 32
        D_DECAY_LORA = 64
        D_AAA_LORA = 16
        D_KKK_LORA = 16
        D_GATE_LORA = 128
        D_MA_LORA = 16
        D_MK_LORA = 16
        
        scale_factor = 0.1
        
        self.time_maa_w1 = Tensor.zeros(n_embd, D_MIX_LORA*4)
        self.time_maa_w2 = self._init_orthogonal([4, D_MIX_LORA, n_embd], scale_factor)
        
        self.time_decay_w1 = Tensor.zeros(n_embd, D_DECAY_LORA)
        self.time_decay_w2 = self._init_orthogonal([D_DECAY_LORA, n_embd], scale_factor)
        
        self.time_aaa_w1 = Tensor.zeros(n_embd, D_AAA_LORA)
        self.time_aaa_w2 = self._init_orthogonal([D_AAA_LORA, n_embd], scale_factor)
        
        self.time_kkk_w1 = Tensor.zeros(n_embd, D_KKK_LORA)
        self.time_kkk_w2 = self._init_orthogonal([D_KKK_LORA, n_embd], scale_factor)
        
        self.gate_w1 = self._init_orthogonal([n_embd, D_GATE_LORA], scale_factor)
        self.gate_w2 = self._init_orthogonal([D_GATE_LORA, n_embd], scale_factor)
        
        self.ma_w1 = Tensor.zeros(n_embd, D_MA_LORA)
        self.ma_w2 = self._init_orthogonal([D_MA_LORA, n_embd], scale_factor)
        self.time_misc_a = Tensor.zeros(1, 1, n_embd)
        
        self.mk_w1 = Tensor.zeros(n_embd, D_MK_LORA)
        self.mk_w2 = self._init_orthogonal([D_MK_LORA, n_embd], scale_factor)
        self.time_misc_k = Tensor.zeros(1, 1, n_embd)

        # Initialize attention layers
        self.receptance = Linear(n_embd, n_embd, bias=False)
        self.key = Linear(n_embd, n_embd, bias=False)
        self.value = Linear(n_embd, n_embd, bias=False)
        self.output = Linear(n_embd, n_embd, bias=False)

        # Initialize weights
        limit_r = 0.5/(n_embd**0.5)
        limit_k = 0.05/(n_embd**0.5)
        limit_v = 0.5/(n_embd**0.5)
        
        self.receptance.weight = Tensor.uniform(-limit_r, limit_r, self.receptance.weight.shape)
        self.key.weight = Tensor.uniform(-limit_k, limit_k, self.key.weight.shape)
        self.value.weight = Tensor.uniform(-limit_v, limit_v, self.value.weight.shape)
        self.output.weight = Tensor.zeros(self.output.weight.shape)

        self.ln_x = GroupNorm(self.n_head, n_embd, eps=64e-5)

    def _init_orthogonal(self, shape: list, scale: float=1.0) -> Tensor:
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            weight = Tensor.eye(max(shape[0], shape[1]))[:shape[0], :shape[1]] * gain * scale
            return weight
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            weights = []
            for i in range(shape[0]):
                w = Tensor.eye(max(shape[1], shape[2]))[:shape[1], :shape[2]] * gain * scale
                weights.append(w)
            return Tensor.stack(weights)
        else:
            raise ValueError("Orthogonal initialization only supports 2D or 3D tensors")

    def shift(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        return Tensor.cat([Tensor.zeros(B, 1, C), x[:, :-1, :]], dim=1)
        
    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H = self.n_head 

        xx = self.shift(x) - x
        xxx = x + xx * self.time_maa_x 
        xxx = (xxx @ self.time_maa_w1).reshape(B*T, 4, -1).transpose(1,0,2)
        xxx = (xxx @ self.time_maa_w2).reshape(4, B, T, -1)
        mrg, mwa, mk, mv = [xxx[i] for i in range(4)]

        xrg = x + xx * (self.time_maa_rg + mrg)
        xwa = x + xx * (self.time_maa_wa + mwa)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)

        r = self.receptance(xrg)
        w = -(-self.time_decay + (xwa @ self.time_decay_w1) @ self.time_decay_w2).exp().log1p() - 0.5
        k = self.key(xk)
        v = self.value(xv)
        g = (xrg @ self.gate_w1).tanh() @ self.gate_w2

        kk = k + (xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = kk.reshape(B,T,H,-1)
        kk = kk / (kk.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()
        kk = kk.reshape(B,T,C)
        a = (self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2).sigmoid()

        ma = (self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2).sigmoid()
        k = k * ma + k * a * (1 - ma)
        mk = (self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2).sigmoid()
        k = k * (w * mk).clip(max=0).exp()

        # RWKV attention
        y = self._rwkv_attention(r, w, k, v, -kk, kk*a)
        y = self.ln_x(y.reshape(B * T, C)).reshape(B, T, C)

        y = y + ((r.reshape(B,T,H,-1) * k.reshape(B,T,H,-1) * self.time_faaaa).sum(dim=-1, keepdim=True) * v.reshape(B,T,H,-1)).reshape(B,T,C)
        return self.output(y * g)

    def _rwkv_attention(self, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> Tensor:
        B, T, C = r.shape
        H = self.n_head
        
        r = r.reshape(B,T,H,-1)
        w = w.reshape(B,T,H,-1)
        k = k.reshape(B,T,H,-1)
        v = v.reshape(B,T,H,-1)
        a = a.reshape(B,T,H,-1)
        b = b.reshape(B,T,H,-1)

        w_pref = w.exp().cumprod(axis=1)
        w_pref_shifted = Tensor.cat([Tensor.ones(B,1,H,-1), w_pref[:, :-1]], dim=1)
        
        wk = k * w_pref_shifted
        wkv = (wk @ v.transpose(2,3)).transpose(2,3)
        wkv = wkv + (a @ wkv.transpose(2,3)).transpose(2,3)
        out = r * wkv + b
        
        return out.reshape(B,T,C)

class MLP:
    def __init__(self, n_embd: int):
        hidden_dim = 7 * n_embd // 2
        self.c_fc = Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = Linear(hidden_dim, n_embd, bias=False)
        self.c_proj.weight = Tensor.zeros(self.c_proj.weight.shape)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).relu().square())

class Block:
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int = 64):
        self.attn = RWKV7(n_embd, n_layer, layer_id, head_size)
        self.mlp = MLP(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT:
    def __init__(self, vocab_size: int = 50304, n_layer: int = 12, 
                 n_head: int = 12, n_embd: int = 768, head_size: int = 64):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = head_size
        
        self.transformer = {
            'wte': Embedding(vocab_size, n_embd),
            'h': [Block(n_embd, n_layer, i, head_size) for i in range(n_layer)]
        }
        
        self.lm_head = Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer['wte'].weight
        self.ln_out = LayerNorm(n_embd)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None, 
                emb_scale: float = 1.0, return_logits: bool = True) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # Get embeddings and apply scaling
        x = self.transformer['wte'](idx) * emb_scale
        
        # Pass through transformer blocks
        for block in self.transformer['h']:
            x = block.forward(x)
            
        # Final layer norm
        x = self.ln_out(x)
        
        if targets is not None:
            # Training path
            logits = self.lm_head(x)
            
            # Compute cross entropy loss
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)
            
            loss = (logits.log_softmax(dim=-1) * targets.one_hot(self.vocab_size)).sum() / (targets != -1).sum()
            loss = -loss
            
        else:
            # Inference path - only compute last logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        if not return_logits:
            logits = None
            
        return logits, loss

class Muon:
    def __init__(self, params: List[Tensor], lr: float = 3e-4, 
                 momentum: float = 0.95, nesterov: bool = True,
                 backend: str = 'newtonschulz5', backend_steps: int = 5):
        self.params = list(params)
        self.state: Dict[int, Dict] = {}
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.backend = backend
        self.backend_steps = backend_steps
        
        for p in self.params:
            assert len(p.shape) == 2, "Muon only supports 2D parameters"

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def _zeropower_via_newtonschulz5(self, G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.cast("half")
        X = X / (X.norm() + eps)
        
        transpose = False
        if X.shape[0] > X.shape[1]:
            X = X.transpose()
            transpose = True
            
        for _ in range(steps):
            A = X @ X.transpose()
            B = A @ X
            X = a * X + b * B + c * A @ B
            
        if transpose:
            X = X.transpose()
            
        return X
    
    def _split_qkv_params(self, grad: Tensor) -> List[Tensor]:
        if grad.shape[0] == 3 * grad.shape[1]:
            splits = []
            for g in grad.split(grad.shape[1], dim=0):
                splits.append(self._zeropower_via_newtonschulz5(g, self.backend_steps))
            return splits
        return [self._zeropower_via_newtonschulz5(grad, self.backend_steps)]

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
                
            state = self.state.get(id(p), {})
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = Tensor.zeros_like(p.grad)
            self.state[id(p)] = state
            
            buf = state['momentum_buffer']
            buf = buf * self.momentum + p.grad
            state['momentum_buffer'] = buf
            
            grad = p.grad
            if self.nesterov:
                grad = grad + buf * self.momentum
                
            if self.backend == 'newtonschulz5':
                grads = self._split_qkv_params(grad)
                scales = [g.shape[1]**0.5 if g.shape[0] == 3 * g.shape[1]
                         else max(g.shape[0], g.shape[1])**0.5 
                         for g in grads]
                
                if len(grads) == 1:
                    p.assign(p - grads[0] * (self.lr * scales[0]))
                else:
                    p.assign(p - Tensor.cat(grads, dim=0) * (self.lr * scales[0]))
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

class DataLoader:
    def __init__(self, file_pattern: str, batch_size: int, seq_length: int):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.files = sorted(glob.glob(file_pattern))
        assert len(self.files) > 0, f"No files found matching {file_pattern}"
        
        self.total_tokens = 0
        for fname in self.files:
            ntok = self._peek_shard(fname)
            assert ntok >= batch_size * seq_length + 1
            self.total_tokens += ntok
            
        self.reset()
        
    def _peek_shard(self, filename: str) -> int:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520, "Invalid magic number"
            assert header[1] == 1, "Unsupported version"
            return header[2]
            
    def _load_shard(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            ntok = header[2]
            data = np.frombuffer(f.read(), dtype=np.uint16)
            assert len(data) == ntok
            return data
            
    def reset(self):
        self.current_shard = 0
        self.current_pos = 0
        self.tokens = self._load_shard(self.files[0])
        
    def next_batch(self) -> Tuple[Tensor, Tensor]:
        B, T = self.batch_size, self.seq_length
        
        if self.current_pos + B*T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.current_pos = 0
            self.tokens = self._load_shard(self.files[self.current_shard])
            
        data = self.tokens[self.current_pos:self.current_pos + B*T + 1]
        self.current_pos += B*T
        
        x = Tensor(data[:-1].reshape(B, T).astype(np.int32))
        y = Tensor(data[1:].reshape(B, T).astype(np.int32))
        
        return x, y

@TinyJit
def train_step(model: GPT, x: Tensor, y: Tensor, emb_scale: float) -> Tuple[Tensor, Tensor]:
    logits, loss = model(x, y, emb_scale=emb_scale)
    loss.backward()
    return logits, loss

@TinyJit  
def val_step(model: GPT, x: Tensor, y: Tensor, emb_scale: float) -> Tensor:
    _, loss = model(x, y, emb_scale=emb_scale, return_logits=False)
    return loss

def train(config: TrainingConfig):
    model = GPT(vocab_size=config.vocab_size,
                n_layer=config.n_layer, 
                n_head=config.n_head,
                n_embd=config.n_embd,
                head_size=config.head_size)
    
    # Split parameters between optimizers
    muon_params = []
    adam_params = []
    
    for name, p in model.named_parameters():
        if (".attn.receptance." in name or 
            'attn.key.' in name or 
            'attn.value.' in name or 
            'attn.gate.' in name or 
            'attn.output.' in name or 
            '.mlp.' in name):
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print(f"Adam params: {len(adam_params)}")
    print(f"Muon params: {len(muon_params)}")
    
    muon_opt = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adam_opt = Adam(adam_params, lr=config.adam_lr, b1=0.9, b2=0.95)
    
    train_loader = DataLoader(config.train_data_path, config.batch_size, config.sequence_length)
    val_loader = DataLoader(config.val_data_path, config.batch_size, config.sequence_length)
    
    training_start = time.time()
    val_batches = config.val_tokens // (config.batch_size * config.sequence_length)
    
    for step in range(config.num_iterations + 1):
        # Learning rate schedule
        if step < config.warmup_iters:
            lr_mult = step / config.warmup_iters
        elif step > config.num_iterations - config.warmdown_iters:
            lr_mult = (config.num_iterations - step) / config.warmdown_iters
        else:
            lr_mult = 1.0
            
        muon_opt.lr = config.muon_lr * lr_mult
        adam_opt.lr = config.adam_lr * lr_mult
        
        # Training step
        x, y = train_loader.next_batch()
        _, loss = train_step(model, x, y, config.emb_scale)
        
        # Optimizer steps and zero grads
        muon_opt.step()
        adam_opt.step()
        model.zero_grad()
        muon_opt.zero_grad()
        adam_opt.zero_grad()
        
        # Validation
        if step % config.val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_loader.reset()
            
            for _ in range(val_batches):
                x_val, y_val = val_loader.next_batch()
                val_loss += val_step(model, x_val, y_val, config.emb_scale).numpy()
                
            val_loss /= val_batches
            
            time_per_step = (time.time() - training_start) * 1000 / (step + 1)
            print(f"Step {step}: train_loss={loss.numpy():.4f}, val_loss={val_loss:.4f}, ms/step={time_per_step:.1f}")
            
            model.train()

def generate(model: GPT, prompt: List[int], max_tokens: int = 100,
            temperature: float = 0.7, top_p: float = 0.9) -> List[int]:
    model.eval()
    x = Tensor([prompt])
    generated = []
    
    for _ in range(max_tokens):
        logits, _ = model(x, return_logits=True)
        logits = logits[:, -1, :] / temperature
        
        probs = logits.softmax()
        sorted_probs = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum()
        nucleus = cumsum < top_p
        probs[~nucleus] = 0
        
        next_token = probs.multinomial(1)
        generated.append(next_token.item())
        
        x = Tensor([prompt + generated])
        
        if next_token == 0:  # EOT token
            break
            
    return generated

if __name__ == "__main__":
    config = TrainingConfig()
    print("Starting training...")
    train(config)