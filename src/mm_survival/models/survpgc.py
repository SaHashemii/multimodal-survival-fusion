from __future__ import annotations

import torch
from torch import einsum, nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 128, heads: int = 1, num_pathways: int = 6):
        super().__init__()
        self.num_pathways = num_pathways
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, n, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, n, h, -1).permute(0, 2, 1, 3)

        if mask is not None:
            keep = mask[:, None, :, None].to(dtype=q.dtype)
            q, k, v = q * keep, k * keep, v * keep

        q = q * self.scale
        q_pathways = q[:, :, : self.num_pathways, :]
        k_pathways = k[:, :, : self.num_pathways, :]
        q_histology = q[:, :, self.num_pathways :, :]
        k_histology = k[:, :, self.num_pathways :, :]

        eq = "... i d, ... j d -> ... i j"
        cross_attn_histology = einsum(eq, q_histology, k_pathways).softmax(dim=-1)
        attn_pathways = einsum(eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(eq, q_pathways, k_histology)
        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        out_pathways = attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, : self.num_pathways]
        out = torch.cat((out_pathways, out_histology), dim=2)
        return out.permute(0, 2, 1, 3).reshape(b, n, -1)


class MMAttentionLayer(nn.Module):
    def __init__(self, dim: int = 256, dim_head: int = 128, heads: int = 1, num_pathways: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MMAttention(dim=dim, dim_head=dim_head, heads=heads, num_pathways=num_pathways)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(self.norm(x), mask=mask)


class GroupedRNAEncoder(nn.Module):
    """Encode full RNA vectors into one token per category or pathway."""

    def __init__(
        self,
        group_gene_indices: list[list[int]],
        out_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        if not group_gene_indices:
            raise ValueError("At least one RNA group is required.")
        self.num_groups = len(group_gene_indices)
        self.index_buffer_names: list[str] = []
        encoders = []
        for idx, indices in enumerate(group_gene_indices):
            if not indices:
                raise ValueError(f"RNA group {idx} has no genes.")
            name = f"group_{idx}_indices"
            self.register_buffer(name, torch.tensor(indices, dtype=torch.long), persistent=False)
            self.index_buffer_names.append(name)
            encoders.append(
                nn.Sequential(
                    nn.LayerNorm(len(indices)),
                    nn.Linear(len(indices), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            )
        self.encoders = nn.ModuleList(encoders)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        tokens = []
        for name, encoder in zip(self.index_buffer_names, self.encoders):
            indices = getattr(self, name)
            group_expr = rna.index_select(dim=1, index=indices).float()
            tokens.append(encoder(group_expr))
        return torch.stack(tokens, dim=1)


class SurvPGCUnifiedCox(nn.Module):
    """SurvPGC-style co-attention for pathology, clinical tokens, and omics tokens."""

    def __init__(
        self,
        pathology_in_dim: int,
        clinical_in_dim: int,
        clinical_token_count: int,
        omics_in_dim: int,
        omics_token_count: int,
        projection_dim: int,
        attention_dim_head: int,
        fusion_hidden_dims: list[int],
        fusion_dropout: float,
        rna_gene_indices: list[list[int]] | None = None,
        rna_hidden_dim: int = 256,
        rna_token_dim: int = 256,
        rna_dropout: float = 0.25,
    ):
        super().__init__()
        self.uses_rna_encoder = rna_gene_indices is not None
        self.num_omics = len(rna_gene_indices) if rna_gene_indices is not None else omics_token_count
        self.num_clinic = clinical_token_count

        self.pathology_projection = nn.Linear(pathology_in_dim, projection_dim)
        self.rna_encoder = None
        if self.uses_rna_encoder:
            self.rna_encoder = GroupedRNAEncoder(
                group_gene_indices=rna_gene_indices or [],
                out_dim=rna_token_dim,
                hidden_dim=rna_hidden_dim,
                dropout=rna_dropout,
            )
            omics_in_dim = rna_token_dim
        self.omics_projection = nn.Linear(omics_in_dim, projection_dim)
        self.clinical_projection = nn.Linear(clinical_in_dim, projection_dim)

        self.omics_pathology_attn = MMAttentionLayer(
            dim=projection_dim,
            dim_head=attention_dim_head,
            heads=1,
            num_pathways=self.num_omics,
        )
        self.clinical_pathology_attn = MMAttentionLayer(
            dim=projection_dim,
            dim_head=attention_dim_head,
            heads=1,
            num_pathways=self.num_clinic,
        )
        self.feed_forward = FeedForward(attention_dim_head, dropout=0.1)
        self.layer_norm = nn.LayerNorm(attention_dim_head)

        fused_dim = attention_dim_head * 4
        layers: list[nn.Module] = [nn.LayerNorm(fused_dim), nn.Dropout(fusion_dropout)]
        prev = fused_dim
        for hidden_dim in fusion_hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), nn.SELU(), nn.AlphaDropout(fusion_dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def omics_tokens(self, omics: torch.Tensor) -> torch.Tensor:
        if self.rna_encoder is not None:
            return self.rna_encoder(omics)
        return omics.float()

    def fused_representation(
        self,
        pathology: torch.Tensor,
        omics: torch.Tensor,
        clinical: torch.Tensor,
        pathology_mask: torch.Tensor | None = None,
        omics_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        path_embed = self.pathology_projection(pathology.float())
        omics_embed = self.omics_projection(self.omics_tokens(omics))
        clinic_embed = self.clinical_projection(clinical.float())
        if omics_mask is not None:
            sample_mask = omics_mask.reshape(-1, 1, 1).to(device=omics_embed.device, dtype=omics_embed.dtype)
            omics_embed = omics_embed * sample_mask

        tokens_omics_path = torch.cat([omics_embed, path_embed], dim=1)
        tokens_clinic_path = torch.cat([clinic_embed, path_embed], dim=1)

        omics_path_mask = None
        if omics_mask is not None:
            omics_keep = omics_mask.reshape(-1, 1).to(device=pathology.device, dtype=torch.bool).expand(-1, self.num_omics)
            path_keep = torch.ones(path_embed.shape[:2], dtype=torch.bool, device=pathology.device)
            omics_path_mask = torch.cat([omics_keep, path_keep], dim=1)

        # Kept faithful to the current SurvPGC scripts for ordinary runs:
        # padding masks are not applied unless an RNA/omics availability mask is provided.
        mm_omics_path = self.omics_pathology_attn(tokens_omics_path, mask=omics_path_mask)
        mm_clinic_path = self.clinical_pathology_attn(tokens_clinic_path, mask=None)

        mm_omics_path = self.layer_norm(self.feed_forward(mm_omics_path))
        mm_clinic_path = self.layer_norm(self.feed_forward(mm_clinic_path))

        omics_post = mm_omics_path[:, : self.num_omics, :].mean(dim=1)
        path_post_omics = mm_omics_path[:, self.num_omics :, :].mean(dim=1)
        if omics_mask is not None:
            sample_mask = omics_mask.reshape(-1, 1).to(device=omics_post.device, dtype=omics_post.dtype)
            omics_post = omics_post * sample_mask
            path_post_omics = path_post_omics * sample_mask
        clinic_post = mm_clinic_path[:, : self.num_clinic, :].mean(dim=1)
        path_post_clinic = mm_clinic_path[:, self.num_clinic :, :].mean(dim=1)
        return torch.cat([omics_post, path_post_omics, clinic_post, path_post_clinic], dim=1)

    def forward_all(
        self,
        pathology: torch.Tensor,
        omics: torch.Tensor,
        clinical: torch.Tensor,
        pathology_mask: torch.Tensor | None = None,
        omics_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self.fused_representation(pathology, omics, clinical, pathology_mask, omics_mask=omics_mask)
        return self.head(fused).squeeze(-1)
