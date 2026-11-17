import torch
import torch.nn as nn
import torch.nn.functional as F
import vit

# --- MAE Core Implementation ---

class MaskedAutoencoder(nn.Module):

    """
    MAE Architecture for Self-Supervised Pre-training.
    Asymmetric: ViT Encoder (only visible patches) + Lightweight Decoder.
    """

    def __init__(self, encoder: vit.ViT, decoder_dim=128, decoder_depth=1,
                 masking_ratio=0.75, image_size=64, patch_size=4, in_channels=3):
        super().__init__()

        # --- Encoder Setup ---
        self.encoder = encoder
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder_embed_dim = encoder.blocks[0].attn.qkv.in_features

        # --- Decoder Setup ---
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim),
            requires_grad=True
        )

        decoder_num_heads = decoder_dim // (self.encoder_embed_dim // encoder.num_heads)
        if decoder_num_heads == 0: decoder_num_heads = 4 # Ensure at least 4 heads

        self.decoder_blocks = nn.Sequential(*[
            vit.Block(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.,
                dropout=0.0,
                rope_module=None # No RoPE in decoder
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)

        # Initialize mask token and decoder PE
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)


    # --- random_masking function ---
    def random_masking(self, x, mask_ratio):
        """
        Implementation of random_masking.
        x: [B, N_patch, D] (Patches, NO CLS Token)
        """

        B, N, D = x.shape # N = num_patches (e.g., 64)
        len_keep = int(N * (1 - mask_ratio)) # Number of patches to keep

        # 1. Generate random permutation indices
        noise = torch.rand(B, N, device=x.device)

        # 2. Sort to get indices for kept and masked patches
        # ids_shuffle: [B, N] indices to shuffle the sequence
        # ids_restore: [B, N] indices to restore the original order
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 3. Separate kept patches and their indices
        ids_keep = ids_shuffle[:, :len_keep] # [B, len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # [B, len_keep, D]

        # 4. Generate the mask for loss calculation
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0 # 0 for visible, 1 for masked
        mask = torch.gather(mask, dim=1, index=ids_restore) # [B, N] in original order

        return x_kept, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        x_patch_emb = self.encoder.patch_embed(x)
        x_patch_emb = x_patch_emb.flatten(2).transpose(1, 2)

        # Masking
        x_visible, mask, ids_restore = self.random_masking(x_patch_emb, mask_ratio)

        # Add CLS token and Positional Embedding to visible patches
        cls_token = self.encoder.cls_token
        x_visible_seq = torch.cat((cls_token.expand(x.shape[0], -1, -1), x_visible), dim=1) # [B, N_keep + 1, D_enc]

        encoded_features = self.encoder.blocks(x_visible_seq)

        encoded_features = self.encoder.norm(encoded_features)

        return encoded_features, mask, ids_restore

    def forward_decoder(self, encoded_features, ids_restore):

        # TODO: Implement the decoder forward pass
        # Steps:
        # 1. Project encoded features to decoder embedding dimension
        # 2. Separate CLS token and encoded patches
        # 3. Insert Mask Tokens to form the full sequence
        # 4. Re-order the sequence to the original Patch order
        # 5. Prepend CLS Token and Add Decoder Positional Embedding
        # 6. Pass through decoder blocks and normalization
        # 7. Predict pixel values for masked patches

        # 1. Separate CLS token and encoded patches
        x_cls = encoded_features[:, :1, :]
        x_visible = encoded_features[:, 1:, :]

        # 2. Create Mask Tokens
        B, N_visible, C = x_visible.shape
        N_total = ids_restore.shape[1]
        N_masked = N_total - N_visible
        # Expand mask token: [1, 1, enc_dim] -> [B, N_masked, enc_dim]
        mask_tokens = self.mask_token.expand(B, N_masked, -1)

        # 3. Concatenate visible patches and mask tokens
        # Current state: [visible_1, visible_2, ... | mask_1, mask_2, ...] (Shuffled)
        # [B, N_visible + N_masked, C]
        x_cat = torch.cat([x_visible, mask_tokens], dim=1)

        # 4. Re-order the sequence to the original Patch order using ids_restore
        # ids_restore shape: [B, N_total] -> expand to [B, N_total, enc_dim] for gather
        # Out[b, i, c] = Input[b, Index[b, i, c], c]
        x_restored = torch.gather(
            x_cat, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        # 5. Prepend CLS Token
        x_full = torch.cat([x_cls, x_restored], dim=1)

        # 6. Project to decoder embedding dimension
        x = self.decoder_embed(x_full)

        # 7. Add Decoder Positional Embedding
        x = x + self.decoder_pos_embed

        # 8. Pass through decoder blocks and normalization
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # 9. Predict pixel values
        # Output shape: [B, N+1, patch_size^2 * channels]
        x_pred = self.decoder_pred(x)

        # Remove CLS token from prediction (we only care about image patches)
        x_pred = x_pred[:, 1:, :]

        return x_pred

    def forward(self, imgs):

        # 1. Encode (Masking happens here)
        encoded_features, mask, ids_restore = self.forward_encoder(imgs, self.masking_ratio)

        # 2. Decode
        x_pred = self.forward_decoder(encoded_features, ids_restore)

        p = self.patch_size
        c = imgs.shape[1]
        target = imgs.unfold(2, p, p).unfold(3, p, p).reshape(imgs.shape[0], c, -1, p, p)
        target = target.permute(0, 2, 3, 4, 1).flatten(2)


        # TODO: Compute MAE Loss
        # 1. Calculate MSE per pixel
        loss = (x_pred - target) ** 2

        # 2. Mean over the patch dimension (pixels) to get loss per patch
        loss = loss.mean(dim=-1)  # [B, N]

        # 3. Apply mask: sum loss over masked patches only, then average
        loss = (loss * mask).sum() / mask.sum()


        return loss, x_pred, mask
