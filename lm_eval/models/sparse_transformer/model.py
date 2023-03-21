"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from .submodules.utils import CfgNode as CN
from .submodules.parallel_linear.src import moe, distributed_moe, singlemoe

# -----------------------------------------------------------------------------

@torch.jit.script
def NewGELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@torch.jit.script
def stickbreaking(logits: torch.Tensor, mask: torch.Tensor, cum_weight: torch.Tensor) -> torch.Tensor:
    """
    Stick-breaking attention weights.
    """
    log_z = F.logsigmoid(logits)
    log_beta = (log_z - logits).masked_fill(mask[None, :, :, None, None] == 0, 0)
    re_cum_log_beta = torch.einsum('bijnh,jk->biknh', log_beta, cum_weight)
    log_p = log_z + re_cum_log_beta
    return log_p.exp().masked_fill(mask[None, :, :, None, None] == 0, 0)

class SparseCausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.att_hidden % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        if config.n_att_experts == 1:
            self.q_proj = singlemoe.MoE(
                input_size=config.n_embd, 
                head_size=config.att_hidden, 
                num_experts=config.n_att_experts, 
                top_k=config.k_att,
                acc_aux_loss=config.universal, 
                bias=False,
            )
        elif config.moe_type == 'moe':
            self.q_proj = moe.MoE(
                input_size=config.n_embd, 
                head_size=config.att_hidden, 
                num_experts=config.n_att_experts, 
                top_k=config.k_att,
                acc_aux_loss=config.universal, 
                bias=False,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
            )
        elif config.moe_type == 'distmoe':
            self.q_proj = distributed_moe.MoE(
                input_size=config.n_embd, 
                head_size=config.att_hidden, 
                num_experts=config.n_att_experts, 
                top_k=config.k_att,
                acc_aux_loss=config.universal, 
                local_size=config.local_size,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
                local_group=config.local_group,
                param_group=config.param_group,
            )
        if config.att_hidden == config.n_embd and config.n_head == 1:
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
        else:
            self.k_proj = nn.Linear(config.n_embd, config.att_hidden)
            self.v_proj = nn.Linear(config.n_embd, config.att_hidden)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.history_length = config.history_length

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size + config.history_length, config.block_size  + config.history_length, dtype=torch.int8))
        )
        self.register_buffer(
            "cum_weight", 
            torch.tril(torch.ones(config.block_size + config.history_length, config.block_size  + config.history_length), -1)
        )
        self.n_head = config.n_head
        self.top_k = config.k_att
        self.n_embd = config.n_embd
        self.att_hidden = config.att_hidden
        self.head_size = config.att_hidden // config.n_head

        self.att_func = config.att_func

    def add_history(self, k, v, hidden):
        if hidden is None:
            new_k = k
            new_v = v
        else:
            k_history, v_history = hidden
            new_k = torch.cat([k_history, k], dim=1)
            new_v = torch.cat([v_history, v], dim=1)
        k_history = new_k.detach()
        v_history = new_v.detach()
        # k_history = new_k[:, -self.history_length:].detach()
        # v_history = new_v[:, -self.history_length:].detach()

        return new_k, new_v, (k_history, v_history)

    def forward(self, x, hidden):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        q, aux_loss = self.q_proj.map(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k, v, hidden = self.add_history(k, v, hidden)
        context_length = k.size(1)
        
        q = q.view(B, T, self.top_k, self.n_head, self.head_size) # (B, T, k, nh, hs)
        k = k.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)
        v = v.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)

        # mask = self.mask[context_length - T:context_length, :context_length]
        tril_ones = torch.tril(torch.ones(context_length, context_length, dtype=x.dtype, device=x.device))
        mask = tril_ones[context_length - T:]

        # causal self-attention; Self-attend: (B, T, k, nh, hs) x (B, T, nh, hs) -> (B, T, T, k, nh)
        att = torch.einsum('bikhd,bjhd->bijkh', q, k) * (1.0 / math.sqrt(k.size(-1)))
        if self.att_func == 'softmax':
            att = att.masked_fill(mask[None, :, :, None, None] == 0, float('-inf'))
            att = F.softmax(att, dim=2)
        else:
            cum_weight = tril_ones.tril(-1)
            # cum_weight = self.cum_weight[:context_length, :context_length]
            att = stickbreaking(att, mask=mask, cum_weight=cum_weight)
        att = self.attn_dropout(att)
        # y = att @ v 
        y = torch.einsum('bijkh,bjhd->bikhd', att, v) # (B, T, T, k, nh) x (B, T, nh, hs) -> (B, T, k, nh, hs)

        # output projection
        y = self.q_proj.reduce(y.reshape(B, T, self.top_k, self.att_hidden).type_as(x))

        y = y.view(B, T, C) # re-assemble all head outputs side by side
        return y, aux_loss, hidden

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # self.attn = CausalSelfAttention(config)
        self.attn = SparseCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        if config.n_mlp_experts == 1:
            self.mlpf = singlemoe.MoE(
                input_size=config.n_embd, 
                head_size=config.ffd_hidden, 
                num_experts=config.n_mlp_experts, 
                top_k=config.k_mlp, 
                bias=False, 
                activation=NewGELU,
                acc_aux_loss=config.universal,
            )
        elif config.moe_type == 'moe':
            self.mlpf = moe.MoE(
                input_size=config.n_embd, 
                head_size=config.ffd_hidden, 
                num_experts=config.n_mlp_experts, 
                top_k=config.k_mlp, 
                bias=False, 
                activation=NewGELU,
                acc_aux_loss=config.universal,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
            )
        elif config.moe_type == 'distmoe':
            self.mlpf = distributed_moe.MoE(
                input_size=config.n_embd, 
                head_size=config.ffd_hidden, 
                num_experts=config.n_mlp_experts, 
                top_k=config.k_mlp, 
                activation=NewGELU,
                local_size=config.local_size,
                gating_dropout=config.moe_pdrop,
                acc_aux_loss=config.universal,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
                local_group=config.local_group,
                param_group=config.param_group,
            )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def get_aux_loss_and_clear(self):
        return self.attn.q_proj.get_aux_loss_and_clear() + self.mlpf.get_aux_loss_and_clear()

    def forward(self, x, hidden=None):
        x_att, att_loss, hidden = self.attn(self.ln_1(x), hidden)
        x = x + self.resid_dropout(x_att)
        x_mlp, mlp_loss = self.mlpf(self.ln_2(x))
        x = x + self.resid_dropout(x_mlp)
        return x, att_loss + mlp_loss, hidden

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        C.att_hidden = None
        C.ffd_hidden = None
        C.universal = False
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0
        C.resid_pdrop = 0
        C.attn_pdrop = 0
        C.moe_pdrop = 0
        C.sample_topk = 0
        C.gating_size = 256,
        C.n_att_experts = 32
        C.k_att = 2
        C.n_mlp_experts = 32
        C.k_mlp = 2
        C.moe_type = 'distmoe'
        C.world_size = None
        C.local_size = None
        C.att_func = 'softmax'
        C.history_length = 0
        C.aux_loss_type = 'mi'
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
                # new model
                'vt':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ),  
                'vt-large':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=2048, ffd_hidden=8092, moe_type='moe'
                    ), 
                'vt-deep':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ), 
                'vt-350m':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ), 
                'vt-350m-moe':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=6, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='distmoe'
                    ), 
                'st-4e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=4, n_mlp_experts=4, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'st':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=12, n_mlp_experts=12, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'st-16e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=16, n_mlp_experts=16, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-deep-16e':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=16, n_mlp_experts=16, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-deep':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=6, n_mlp_experts=6, 
                    att_hidden=1024, ffd_hidden=2048
                    ), 
                'st-wide':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=False, 
                    n_att_experts=12, n_mlp_experts=12, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-18e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=18, n_mlp_experts=18,
                    att_hidden=1024, ffd_hidden=4096
                    ),   
                'st-24e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=24, n_mlp_experts=24, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'sut':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=30, n_mlp_experts=30, 
                    att_hidden=1024, ffd_hidden=4096),  
                'sut-12-36':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=12, n_mlp_experts=36, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-32':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=32, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-48':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=48, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-48-1head':         dict(
                    n_layer=24, n_head=1, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=48, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-wide':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=True, 
                    n_att_experts=24, n_mlp_experts=24, 
                    att_hidden=1024, ffd_hidden=2048),
            }[config.model_type])

        self.universal = config.universal
        if config.moe_type == 'distmoe':
            config.local_group, config.param_group = self.init_groups()
        
        if self.universal:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd) if config.att_func == 'softmax' else None,
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([Block(config)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd) if config.att_func == 'softmax' else None,
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.n_layer = config.n_layer

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def init_groups(self):
        local_size = self.config.local_size
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return distributed_moe.init_groups(world_size, local_size, rank)

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layer):
            hidden.append(None)
        return hidden

    def forward(self, idx, targets=None, hidden=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            tok_emb = tok_emb + pos_emb
        x = self.transformer.drop(tok_emb)
        
        if hidden is None:
            hidden = self.init_hidden()
        new_hidden = []
        if self.universal:
            for i in range(self.n_layer):
                x, _, hidden_i = self.transformer.h[0](x, hidden[i])
                new_hidden.append(hidden_i)
            aux_loss = self.transformer.h[0].get_aux_loss_and_clear()
        else:
            aux_loss = 0
            for block, hidden in zip(self.transformer.h, hidden):
                x, aux_loss_i, hidden_i = block(x, hidden)
                aux_loss = aux_loss + aux_loss_i
                new_hidden.append(hidden_i)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = 0
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss, aux_loss, new_hidden

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, hidden=None, eos_token_id=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        def sample_idx(logits):
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            return idx_next

        idx_next = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
        output = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # forward the model to get the logits for the index in the sequence
            logits, _, _, hidden = self(idx_next, hidden=hidden)
            idx_next = sample_idx(logits)
            output.append(idx_next)
            if (idx_next == eos_token_id).all():
                break

        return torch.cat(output, dim=1), hidden
