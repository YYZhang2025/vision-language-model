from vlm.config import VLMConfig
from vlm.model.utils import scale_dot_product_attention, count_parameters

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTPatchEmbedding(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.img_size = config.vit_img_size
        self.patch_size = config.vit_patch_size

        assert (
            self.img_size % self.patch_size == 0
        ), "Image size must be divisible by patch size."
        self.num_patches = (self.img_size // self.patch_size) ** 2

        self.cls_flag = config.vit_cls_flag
        self.embd_dim = config.vit_hidden_dim

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))  # (1, 1, D)
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.embd_dim)
            )  # (B, P+1, D)
        else:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches, self.embd_dim)
            )

    def forward(self, imgs: torch.Tensor):
        # (B, C, H, W) -> (B, D, H // P, W // P)
        x = self.conv(imgs)
        # (B, D, H // P, W // P) -> (B, H // P * W // P, D)
        x = x.flatten(2).transpose(1, 2)

        if self.cls_flag:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, P+1, D)

        assert (
            self.position_embeddings.shape[1] == x.shape[1]
        ), f"Position embeddings shape {self.position_embeddings.shape[1]} does not match input shape {x.shape[1]}"

        x = x + self.position_embeddings  # (B, P+1, D) or (B, P, D)

        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.n_heads = config.vit_n_heads
        self.embd_dim = config.vit_hidden_dim

        assert (
            self.embd_dim % self.n_heads == 0
        ), "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads

        self.dropout = config.vit_dropout

        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)

        # Dropout layer
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention
        self.sdpa = hasattr(F, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "Warning: Scaled Dot Product Attention not available. Using custom implementation."
            )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        q, k, v = map(
            lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2),
            self.qkv_proj(x).chunk(3, dim=-1),
        )

        if self.sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            y, _ = scale_dot_product_attention(
                q=q, k=k, v=v, dropout=self.dropout if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(config.vit_hidden_dim, config.vit_inter_dim)
        self.fc2 = nn.Linear(config.vit_inter_dim, config.vit_hidden_dim)
        self.dropout = nn.Dropout(config.vit_dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.attn = ViTMultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.vit_hidden_dim, eps=config.vit_ln_eps)
        self.ln2 = nn.LayerNorm(config.vit_hidden_dim, eps=config.vit_ln_eps)

    def forward(self, x: torch.Tensor):
        # Layer normalization and multi-head attention
        x = x + self.attn(self.ln1(x))
        # Layer normalization and MLP
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()

        self.config = config

        self.patch_embedding = ViTPatchEmbedding(config)

        self.cls_flag = config.vit_cls_flag
        self.dropout = nn.Dropout(config.vit_dropout)

        self.blocks = nn.ModuleList(
            [ViTBlock(config) for _ in range(config.vit_n_blocks)]
        )

        self.layer_norm = nn.LayerNorm(config.vit_hidden_dim, eps=config.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, imgs: torch.Tensor):
        x = self.patch_embedding(imgs)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = x[:, 0]
        else:
            x = self.layer_norm(x)

        return x

    @classmethod
    def from_pretrained(cls, config):
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(config.vit_model_type)
        config.vit_dropout = hf_config.attention_dropout
        config.vit_hidden_dim = hf_config.hidden_size
        config.vit_img_size = hf_config.image_size
        config.vit_inter_dim = hf_config.intermediate_size
        config.vit_ln_eps = hf_config.layer_norm_eps
        config.vit_n_heads = hf_config.num_attention_heads
        config.vit_n_blocks = hf_config.num_hidden_layers
        config.vit_patch_size = hf_config.patch_size
        model = cls(config)
        safetensors_file = hf_hub_download(
            repo_id=config.vit_model_type, filename="model.safetensors"
        )

        sd = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(config.vit_n_blocks):
            # Layer norms
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = (f"blocks.{i}.ln1.weight")
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = (f"blocks.{i}.ln1.bias")
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = (f"blocks.{i}.ln2.weight")
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = (f"blocks.{i}.ln2.bias")

            # MLP
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (f"blocks.{i}.mlp.fc1.weight")
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (f"blocks.{i}.mlp.fc1.bias")
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (f"blocks.{i}.mlp.fc2.weight")
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (f"blocks.{i}.mlp.fc2.bias")

            # Output projection
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (f"blocks.{i}.attn.out_proj.weight")
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (f"blocks.{i}.attn.out_proj.bias")

        with safetensors.safe_open(
            filename=safetensors_file, framework="pt", device="cpu"
        ) as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if "position_embedding" in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.config.vit_n_blocks):
                q_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight")
                k_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight")
                v_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight")

                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                q_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias")
                k_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias")
                v_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias")

                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)

        model.load_state_dict(sd)
        print(
            f"Successfully loaded {config.vit_model_type} weights from safetensors. Model has {count_parameters(model):,} trainable parameters."
        )
        return model
