import torch


def expand_tensor(x: torch.Tensor, size: int, front: bool = True) -> torch.Tensor:
    while x.dim() < size:
        x = x.unsqueeze(0) if front else x.unsqueeze(-1)
    return x


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal mask for self-attention.
    Mask shape is (seq_len, seq_len) where the upper triangle is masked.
    E.g., for seq_len=4, the mask will look like:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    return mask


def scale_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
):

    d_k = q.shape[-1]

    # Compute the dot product attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
        d_k**0.5
    )  # Scale by the square root of the dimension
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(attn_scores, dim=-1)
    if dropout > 0.0:
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=True
        )
    # Compute the attention output
    attn_output = torch.matmul(attn_weights, v)  # Shape: (B, S_q, D)

    return attn_output, attn_weights


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
