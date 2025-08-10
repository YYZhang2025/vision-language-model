import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm.config import VLMConfig


class RMSNorm(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(config.lm_hidden_dim))
        self.eps = config.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        div_term = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        return x * div_term * self.weight


# Rotary Position Embedding
class RotaryEmbedding(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        assert (
            config.lm_hidden_dim % 2 == 0
        ), "Hidden dimension must be even for rotary embeddings"
        assert (
            config.lm_hidden_dim % config.lm_n_heads == 0
        ), "Hidden dimension must be divisible by number of heads"

        self.dim = config.lm_hidden_dim // config.lm_n_heads
        self.base = config.lm_re_base
        self.max_seq_len = config.lm_max_position_embeddings

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self.original_max_seq_len = config.lm_max_position_embeddings
        self.attention_scaling = config.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = position_ids.shape

        # Dynamic Scaling for longer sequences
        max_seq = position_ids.max().item() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq

        # Compute theta
        flat_position_ids = position_ids.reshape(-1).float()
        
        freqs = flat_position_ids[:, None] * inv_freq[None, :]
        
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        
        emb = torch.cat([freqs,freqs], dim=-1)
        
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin 

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int= 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot



class LanguageModelGroupQueryAttention(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        
        
        self.n_heads = config.lm_n_heads
        self.n_kv_heads = config.lm_n_kv_heads
        self.embd_dim = config.lm_hidden_dim
        self.dropout = config.lm_dropout
        
        
        assert self.n_heads % self.n_kv_heads == 0, "Number of heads must be divisible by number of KV heads"
        assert self.embd_dim % self.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads
        
        self.q_proj = nn.Linear(self.embd_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.n_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        
        self.sdpa = hasattr(F, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Using custom scaled dot product attention implementation.")
            

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask = None, block_kv_cache = None) -> tuple[torch.Tensor, dict]:   
        
        is_prefill = block_kv_cache is None 
        
        B, T_curr, C = x.size()
        
        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) 
        
        
        q, k_rotated = apply_rotary_pos_emb(q_curr, k_curr, cos, sin)
        
        
        if not is_prefill and block_kv_cache['key'] is not None:
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=-1)
            v = torch.cat([v, v_curr], dim=-1)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        
        else:
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}
    
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # head dimension expansion
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) #

        T_kv = k_exp.size(2)
        
        
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
        
        if self.sdpa and 'mps' not in str(x.device):
            is_causal = (T_curr == T_kv  and T_curr > 1)
            y = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Custom implementation of scaled dot product attention
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None: # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask 

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp
            
        y = y.transpose(1, 2).contiguous().view(B, T_curr, self.embd_dim)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        
        return y, block_kv_cache



class LanguageModelMLP(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        
        
        self.embd_dim = config.lm_hidden_dim
        self.inter_dim = config.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation_fn(self.gate_proj(x))
        up = self.activation_fn(self.up_proj(x))
        
        y = gate * up
        y = self.down_proj(y)
        
        return y

class LanguageModelBlock(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        
        self.mlp = LanguageModelMLP(config)
        self.attn = LanguageModelGroupQueryAttention(config)
        self.rms_norm1 = RMSNorm(config)
        self.rms_norm2 = RMSNorm(config)
        
    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None
    ) -> tuple[torch.Tensor, dict]:

        
        residual = x
        
        x = self.rms_norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        
        x = residual + x
        
        residual = x
        x = self.rms_norm2(x)
        x = self.mlp(x)
        
        x = residual + x
        
        return x, block_kv_cache


class LanguageModel(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        
        self.config = config 
        self.lm_use_tokens = config.lm_use_tokens
        self.lm_tie_weights = config.lm_tie_weights
        
        self.token_embeddings = nn.Embedding(
            config.lm_vocab_size, config.lm_hidden_dim, 
        )
        
        self.rotary_embd = RotaryEmbedding(config)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(config) for _ in range(config.lm_n_blocks)
        ])
        self.norm = RMSNorm(config) # Final Norm
        self.head = nn.Linear(config.lm_hidden_dim, config.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0):
        if self.lm_use_tokens:
            x = self.token_embedding(x)
            
        
        B, T_curr, _ = x.size()
        
        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs

    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")
        
        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    
                    # Special handling for token embeddings if vocab sizes differ
                    if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                        has_extended_embeddings = True
                        print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                        
                        # Copy existing embeddings to the beginning of our larger embedding matrix
                        sd[our_key][:tensor.shape[0]].copy_(tensor)
                        
                        # Initialize the new embeddings using the same approach as the original model
                        std = 0.02  # Common value, but you might want to adjust based on model
                        init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                        
                        print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                        sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                    elif tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                if 'lm_head.weight' in f.keys():
                    lm_head = f.get_tensor('lm_head.weight')
                    if lm_head.shape[0] != sd['head.weight'].shape[0]:
                        print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                        # Copy existing weights
                        sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                        # Initialize new weights
                        std = 0.02
                        init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                        # Load updated weights
                        model.load_state_dict(sd)
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model