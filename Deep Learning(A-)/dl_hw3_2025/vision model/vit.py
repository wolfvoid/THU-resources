import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 2D Rotary Positional Embedding (RoPE) Implementation ---

class RotaryEmbedding2D(nn.Module):
    """
    1.2 2D RoPE Implementation: Generates 2D rotation matrices based on patch indices.

    This implementation adapts the standard 1D RoPE for 2D images by treating the
    (x, y) coordinates of each patch as independent indices.
    """
    def __init__(self, dim, patch_size=8, image_size=64):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # TODO
        # Grid dimensions (number of patches in height and width)
        self.h_grid = image_size // patch_size
        self.w_grid = image_size // patch_size

        # We split the head dimension into two halves: one for Height (y), one for Width (x)
        dim_t = dim // 2

        # 1. Generate frequencies for one dimension (standard RoPE logic)
        # theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_t, 2).float() / dim_t))

        # 2. Create grid for 2D positions
        # Standard ViT flattening is row-major: iterate width (x) then height (y)
        # t_h: [0, 1, ..., h_grid-1]  (行索引)
        # t_w: [0, 1, ..., w_grid-1]  (列索引)
        t_h = torch.arange(self.h_grid).float()
        t_w = torch.arange(self.w_grid).float()

        # meshgrid(..., indexing='ij') creates grids where y varies first, then x
        # grid_h, grid_w: shape (h_grid, w_grid) 记录每个图片/patch的行列索引
        grid_h, grid_w = torch.meshgrid(t_h, t_w, indexing='ij')

        # Flatten to match the sequence of patches: (N_patches, )
        grid_h = grid_h.flatten()
        grid_w = grid_w.flatten()

        # 3. Outer product to generate angles for each position
        # angle_ij = pos_i * inv_freq_j
        # shape: (N_patches, dim_t/2) = (N_patches, 1) * (1, dim_t/2)
        freqs_h = torch.einsum('i,j->ij', grid_h, inv_freq)
        freqs_w = torch.einsum('i,j->ij', grid_w, inv_freq)

        # 4. Construct the full embedding
        # We repeat the frequencies because cos/sin act on pairs of elements [x, y]
        # emb_h: (N_patches, dim_t)
        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)

        # Concatenate height and width embeddings: (N_patches, dim)
        emb = torch.cat((emb_h, emb_w), dim=-1)

        # Calculate cos and sin once and cache them
        # Reshape to (1, 1, N_patches, dim) for broadcasting with (B, H, N, D)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, t: torch.Tensor, seq_len: int):
        """
        Applies rotation to tensor t (Q or K).
        t: [B, H, N, D_head]
        """
        # TODO
        # Crop cached cos/sin to the current sequence length (in case of dynamic sizes, though fixed here)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # RoPE rotation logic:
        # standard: | x1 | * cos - | x2 | * sin
        #           | x2 |         | x1 |
        #
        # Efficient implementation:
        # split last dim into pairs (..., D/2, 2) is complex,
        # but standard way is: [-x2, x1, -x4, x3...]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # x1' = x1 * cos + (-x2) * sin
        # x2' = x2 * cos + x1 * sin
        # [x1 x2]' = [x1 x2] * cos + rotate_half([x1 x2]) * sin
        return (t * cos) + (rotate_half(t) * sin)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0, rope_module=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope_module = rope_module

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.rope_module is not None:

            # TODO
            # Note that q and k contain CLS token at index 0, so we need to separate it out
            # Then apply RoPE only to the patch tokens
            # Last concatenate the CLS token back

            # 1. Separate CLS token and Patch tokens
            # q shape: [B, Num_Heads, Seq_Len, Head_Dim]
            q_cls = q[:, :, 0:1, :]
            q_patches = q[:, :, 1:, :]

            k_cls = k[:, :, 0:1, :]
            k_patches = k[:, :, 1:, :]

            # 2. Apply RoPE to patches only
            # We pass the number of patches as seq_len
            num_patches = q_patches.shape[2]
            q_patches_rotated = self.rope_module(q_patches, num_patches)
            k_patches_rotated = self.rope_module(k_patches, num_patches)

            # 3. Concatenate back
            q = torch.cat((q_cls, q_patches_rotated), dim=2)
            k = torch.cat((k_cls, k_patches_rotated), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.0, rope_module=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout, rope_module=rope_module)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):

    def __init__(self, image_size=64, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=4, num_heads=8, mlp_ratio=4., dropout=0.0):
        super().__init__()


        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (image_size // patch_size) ** 2


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = None
        self.pos_drop = nn.Dropout(p=dropout)

        self.num_heads = num_heads

        self.rope_module = RotaryEmbedding2D(dim=embed_dim // num_heads, # RoPE operates on head dimension
                                             patch_size=patch_size,
                                             image_size=image_size)

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope_module=self.rope_module
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
