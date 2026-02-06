# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

from dataclasses import dataclass, field
from typing import Type, Optional, Union, List, Tuple
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, FP32LayerNorm, RMSNorm
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import is_torch_version

try:
    from sageattention import sageattn
except ImportError:
    sageattn = None
    print("Warning: SageAttention not found. Please install via `pip install sageattention` for acceleration.")

logger = logging.get_logger(__name__)  


class UlyssesAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group=None, scatter_idx=2, gather_idx=1):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        
        if group is None or dist.get_world_size(group) <= 1:
            return input

        world_size = dist.get_world_size(group)
        
        # 2. Local Split (Scatter)
        # input.chunk creates views. We must ensure they are contiguous for dist.all_to_all
        input_list = [x.contiguous() for x in input.chunk(world_size, dim=scatter_idx)]
        
        # 3. All-to-All
        output_list = [torch.empty_like(x) for x in input_list]
        dist.all_to_all(output_list, input_list, group=group)
        
        # 4. Local Merge (Gather)
        output = torch.cat(output_list, dim=gather_idx).contiguous()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return UlyssesAllToAll.apply(grad_output, ctx.group, ctx.gather_idx, ctx.scatter_idx), None, None, None

def seq_parallel_communication(input_tensor, scatter_idx, gather_idx, group=None):
    return UlyssesAllToAll.apply(input_tensor, group, scatter_idx, gather_idx)


# ==============================================================================
# Modeling
# ==============================================================================

class HunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        sp_group = getattr(attn, "sp_group", None)
        world_size = 1 if sp_group is None else dist.get_world_size(sp_group)

        is_single_stream = (attn.add_q_proj is None) and (encoder_hidden_states is not None)

        if is_single_stream:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key   = key.unflatten(  2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key   = attn.norm_k(key)

        if image_rotary_emb is not None:           
            if is_single_stream: 
                text_len = encoder_hidden_states.shape[1]
                query_img = query[:, :, : -text_len]
                query_txt = query[:, :, -text_len :]
                key_img   = key[:, :, : -text_len]
                key_txt   = key[:, :, -text_len :]
                
                query_img = apply_rotary_emb(query_img, image_rotary_emb)
                key_img   = apply_rotary_emb(key_img, image_rotary_emb)
                
                query = torch.cat([query_img, query_txt], dim=2)
                key   = torch.cat([key_img, key_txt], dim=2)
            else: 
                query = apply_rotary_emb(query, image_rotary_emb)
                key   = apply_rotary_emb(key,   image_rotary_emb)

        encoder_query = None
        encoder_key = None
        encoder_value = None
        
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key   = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key   = encoder_key.unflatten(  2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

        if world_size > 1:
            
            q_img, k_img, v_img = None, None, None
            q_txt, k_txt, v_txt = None, None, None

            if is_single_stream:
                text_len = encoder_hidden_states.shape[1]
                q_img = query[:, :, :-text_len]
                q_txt = query[:, :, -text_len:]
                k_img = key[:, :, :-text_len]
                k_txt = key[:, :, -text_len:]
                v_img = value[:, :, :-text_len]
                v_txt = value[:, :, -text_len:]
            else:
                q_img, k_img, v_img = query, key, value
                if encoder_query is not None:
                    q_txt, k_txt, v_txt = encoder_query, encoder_key, encoder_value

            q_img = seq_parallel_communication(q_img, scatter_idx=1, gather_idx=2, group=sp_group)
            k_img = seq_parallel_communication(k_img, scatter_idx=1, gather_idx=2, group=sp_group)
            v_img = seq_parallel_communication(v_img, scatter_idx=1, gather_idx=2, group=sp_group)

            if q_txt is not None:
                q_txt = seq_parallel_communication(q_txt, scatter_idx=1, gather_idx=2, group=sp_group)
                k_txt = seq_parallel_communication(k_txt, scatter_idx=1, gather_idx=2, group=sp_group)
                v_txt = seq_parallel_communication(v_txt, scatter_idx=1, gather_idx=2, group=sp_group)

            if q_txt is not None:
                query = torch.cat([q_img, q_txt], dim=2)
                key   = torch.cat([k_img, k_txt], dim=2)
                value = torch.cat([v_img, v_txt], dim=2)
            else:
                query, key, value = q_img, k_img, v_img
        
        else:
            if encoder_query is not None:
                query = torch.cat([query, encoder_query], dim=2)
                key = torch.cat([key, encoder_key], dim=2)
                value = torch.cat([value, encoder_value], dim=2)

        if sageattn is not None:
            hidden_states = sageattn(
                query, 
                key, 
                value, 
                attn_mask=attention_mask, 
                is_causal=False, 
                tensor_layout="HND"
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        
        if world_size > 1:
            has_text = (is_single_stream) or (encoder_query is not None)
            
            if has_text:
                local_text_len = encoder_hidden_states.shape[1]
                global_text_len = local_text_len * world_size
                
                global_total_len = hidden_states.shape[2]
                global_img_len = global_total_len - global_text_len
                
                hs_img = hidden_states[:, :, :global_img_len, :]
                hs_txt = hidden_states[:, :, global_img_len:, :]
                
                hs_img = seq_parallel_communication(hs_img, scatter_idx=2, gather_idx=1, group=sp_group)
                hs_txt = seq_parallel_communication(hs_txt, scatter_idx=2, gather_idx=1, group=sp_group)
                
                hidden_states = torch.cat([hs_img, hs_txt], dim=2)
                 
            else:
                 hidden_states = seq_parallel_communication(hidden_states, scatter_idx=2, gather_idx=1, group=sp_group)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -text_len],
                hidden_states[:, -text_len :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

        
class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideoTokenReplaceAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        token_replace_emb: torch.Tensor,
        first_frame_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        token_replace_emb = self.linear(self.silu(token_replace_emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        tr_shift_msa, tr_scale_msa, tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp = token_replace_emb.chunk(
            6, dim=1
        )

        norm_hidden_states = self.norm(hidden_states)
        hidden_states_zero = (
            norm_hidden_states[:, :first_frame_num_tokens] * (1 + tr_scale_msa[:, None]) + tr_shift_msa[:, None]
        )
        hidden_states_orig = (
            norm_hidden_states[:, first_frame_num_tokens:] * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            tr_gate_msa,
            tr_shift_mlp,
            tr_scale_mlp,
            tr_gate_mlp,
        )


class HunyuanVideoTokenReplaceAdaLayerNormZeroSingle(nn.Module):
    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        token_replace_emb: torch.Tensor,
        first_frame_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        token_replace_emb = self.linear(self.silu(token_replace_emb))

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        tr_shift_msa, tr_scale_msa, tr_gate_msa = token_replace_emb.chunk(3, dim=1)

        norm_hidden_states = self.norm(hidden_states)
        hidden_states_zero = (
            norm_hidden_states[:, :first_frame_num_tokens] * (1 + tr_scale_msa[:, None]) + tr_shift_msa[:, None]
        )
        hidden_states_orig = (
            norm_hidden_states[:, first_frame_num_tokens:] * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)

        return hidden_states, gate_msa, tr_gate_msa


class HunyuanVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        guidance_embeds: bool,
        image_condition_type: Optional[str] = None,
    ):
        super().__init__()

        self.image_condition_type = image_condition_type

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

        self.guidance_embedder = None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self, 
        timestep: torch.Tensor, 
        pooled_projection: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb  = self.timestep_embedder(timesteps_proj.to(dtype=dtype))  # (N, D)
        conditioning   = timesteps_emb
        if pooled_projection is not None:
            pooled_projections = self.text_embedder(pooled_projection)
            conditioning       += pooled_projections

        token_replace_emb = None
        if self.image_condition_type == "token_replace":
            token_replace_timestep = torch.zeros_like(timestep)
            token_replace_proj = self.time_proj(token_replace_timestep)
            token_replace_emb  = self.timestep_embedder(token_replace_proj.to(dtype=dtype))
            if pooled_projection is not None:
                token_replace_emb  = token_replace_emb + pooled_projections

        if self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=dtype))
            conditioning = conditioning + guidance_emb

        return conditioning, token_replace_emb


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta
    
    def forward(self, hidden_states: torch.Tensor, t_start: int = 0) -> torch.Tensor:
        # Note: hidden_states passed here should be the GLOBAL shape to calculate correct offsets
        # Or we rely on the caller to calculate for the full grid and slice.

        if hidden_states.ndim == 5:
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]
        elif hidden_states.ndim == 3:
            # Assumes flattened, hard to guess dimensions without explicit args
            # This path usually not taken for global grid generation in this codebase
            batch_size, state_length, _ = hidden_states.shape
            rope_sizes = [state_length, 1, 1]
        else:
            raise ValueError(f'hidden state ndim = {hidden_states.ndim} (shape = {hidden_states.shape}) is invalid !!!')

        t_grid = torch.arange(0, rope_sizes[0], device=hidden_states.device, dtype=torch.float32) + t_start
        h_grid = torch.arange(0, rope_sizes[1], device=hidden_states.device, dtype=torch.float32)
        w_grid = torch.arange(0, rope_sizes[2], device=hidden_states.device, dtype=torch.float32)

        grid_T, grid_H, grid_W = torch.meshgrid(t_grid, h_grid, w_grid, indexing="ij")
        grid = torch.stack([grid_T, grid_H, grid_W], dim=0)  # [3, T', H', W']

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        
        return (freqs_cos, freqs_sin)




class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        
        # In Single Blocks, input is concatenated.
        # For SP, both hidden_states and encoder_hidden_states are already sharded.
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff    = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context    = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = HunyuanVideoTokenReplaceAdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_replace_emb: torch.Tensor = None,
        num_tokens: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate, tr_gate = self.norm(hidden_states, temb, token_replace_emb, num_tokens)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        proj_output = self.proj_out(hidden_states)
        hidden_states_zero = proj_output[:, :num_tokens] * tr_gate.unsqueeze(1)
        hidden_states_orig = proj_output[:, num_tokens:] * gate.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = HunyuanVideoTokenReplaceAdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_replace_emb: torch.Tensor = None,
        num_tokens: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            tr_gate_msa,
            tr_shift_mlp,
            tr_scale_mlp,
            tr_gate_mlp,
        ) = self.norm1(hidden_states, temb, token_replace_emb, num_tokens)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states_zero = hidden_states[:, :num_tokens] + attn_output[:, :num_tokens] * tr_gate_msa.unsqueeze(1)
        hidden_states_orig = hidden_states[:, num_tokens:] + attn_output[:, num_tokens:] * gate_msa.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        hidden_states_zero = norm_hidden_states[:, :num_tokens] * (1 + tr_scale_mlp[:, None]) + tr_shift_mlp[:, None]
        hidden_states_orig = norm_hidden_states[:, num_tokens:] * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states_zero = hidden_states[:, :num_tokens] + ff_output[:, :num_tokens] * tr_gate_mlp.unsqueeze(1)
        hidden_states_orig = hidden_states[:, num_tokens:] + ff_output[:, num_tokens:] * gate_mlp.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class VINOTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        image_condition_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        supported_image_condition_types = ["latent_concat", "token_replace"]
        if image_condition_type is not None and image_condition_type not in supported_image_condition_types:
            raise ValueError(
                f"Invalid `image_condition_type` ({image_condition_type}). Supported ones are: {supported_image_condition_types}"
            )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        if image_condition_type == "token_replace":
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )

        # 4. Single stream transformer blocks
        if image_condition_type == "token_replace":
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )
        else:
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False
    
    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_sequence_parallel_group(self, group):
        """Helper to inject the distributed group into all attention processors."""
        for name, module in self.named_modules():
            if hasattr(module, "attn"):
                # Access the internal processor of the Attention layer
                if hasattr(module.attn, "processor"):
                    # We inject the group into the attention module wrapper
                    # The processor.__call__ checks attn.sp_group
                    module.attn.sp_group = group

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,

        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,

        vae_cond_dict: dict = None,
        pooled_projections: torch.Tensor = None,
        
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        
        if sp_group is not None:
            self.set_sequence_parallel_group(sp_group)
            sp_size = dist.get_world_size(sp_group)
            sp_rank = dist.get_rank(sp_group)
        else:
            sp_size = 1
            sp_rank = 0

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height     = height // p
        post_patch_width      = width  // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width

        temb, token_replace_emb = self.time_text_embed(
            timestep, pooled_projections, guidance, dtype=hidden_states.dtype,
        )

        vae_conds = vae_cond_dict["vae_conds"]
        pos_vs    = vae_cond_dict["pos_vs"]
        pos_ve    = vae_cond_dict["pos_ve"]

        hidden_states_addition = []
        image_rotary_emb_list  = []
        t_start                = []
        add_special_token_from_vlm = True 
        
        for pos_vs_, pos_ve_, vae_conds_, encoder_hidden_states_ in zip(pos_vs, pos_ve, vae_conds, encoder_hidden_states[1]): 
            hidden_states_addition_ = []
            image_rotary_emb_       = []
            t_start_                = 0
            for pos_vs_per, pos_ve_per, vae_conds_per in zip(pos_vs_, pos_ve_, vae_conds_): 
                if add_special_token_from_vlm:
                    hidden_states_addition_.append(encoder_hidden_states_[pos_vs_per : pos_vs_per+1][None])
                
                hidden_states_addition_.append(self.x_embedder(vae_conds_per))
                if add_special_token_from_vlm:
                    hidden_states_addition_.append(encoder_hidden_states_[pos_ve_per : pos_ve_per+1][None])
                    
                if add_special_token_from_vlm:
                    image_rotary_emb_.append(self.rope(encoder_hidden_states_[pos_vs_per : pos_vs_per+1].detach()[None], t_start_))
                    t_start_ += 1
                
                t_delta = vae_conds_per.shape[2] // self.rope.patch_size_t
                image_rotary_emb_.append(self.rope(vae_conds_per, t_start_))
                t_start_ += t_delta
                
                if add_special_token_from_vlm:
                    image_rotary_emb_.append(self.rope(encoder_hidden_states_[pos_ve_per : pos_ve_per+1].detach()[None], t_start_))
                    t_start_ += 1
            
            hidden_states_addition.append(hidden_states_addition_)
            image_rotary_emb_list.append(image_rotary_emb_)
            t_start.append(t_start_)

        encoder_hidden_states_proj = self.context_embedder(encoder_hidden_states[0], timestep, encoder_attention_mask)
        
        for batch_idx, hidden_states_ in enumerate(hidden_states):
            hidden_states_emb = self.x_embedder(hidden_states_[None])
            if len(image_rotary_emb_list) > 0:
                image_rotary_emb_list[batch_idx].append(self.rope(hidden_states_[None], t_start[batch_idx]))
                hidden_states_addition[batch_idx].append(hidden_states_emb)
            else:
                image_rotary_emb_list.append([self.rope(hidden_states_[None])])
                hidden_states_addition.append([hidden_states_emb])
            
            hidden_states_addition[batch_idx] = torch.cat(hidden_states_addition[batch_idx], dim=1)
            image_rotary_emb_list[batch_idx] = [
                torch.cat([x[0] for x in image_rotary_emb_list[batch_idx]], dim=0),
                torch.cat([x[1] for x in image_rotary_emb_list[batch_idx]], dim=0),
            ]

        hidden_states = torch.cat(hidden_states_addition, dim=0) 
        image_rotary_emb = image_rotary_emb_list[0]

        original_seq_len = hidden_states.shape[1]
        original_text_len = encoder_hidden_states_proj.shape[1]
        
        if sp_size > 1:
            # --- Pad Hidden States ---
            if original_seq_len % sp_size != 0:
                pad_len = sp_size - (original_seq_len % sp_size)
                # Pad hidden_states [B, N, C]
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
                
                # Pad RoPE (cos, sin) [N, D]
                cos, sin = image_rotary_emb
                cos = F.pad(cos, (0, 0, 0, pad_len))
                sin = F.pad(sin, (0, 0, 0, pad_len))
                image_rotary_emb = (cos, sin)
                
            # --- Pad Text States ---
            if original_text_len % sp_size != 0:
                text_pad_len = sp_size - (original_text_len % sp_size)
                encoder_hidden_states_proj = F.pad(encoder_hidden_states_proj, (0, 0, 0, text_pad_len))

        latent_len_padded = hidden_states.shape[1]
        text_len_padded = encoder_hidden_states_proj.shape[1]
        total_len = latent_len_padded + text_len_padded
        
        attention_mask = torch.ones(
            batch_size, total_len, device=hidden_states.device, dtype=torch.bool
        )
        
        if sp_size > 1:
            if original_seq_len % sp_size != 0:
                pad_len = sp_size - (original_seq_len % sp_size)
                attention_mask[:, original_seq_len : latent_len_padded] = False
            
            if original_text_len % sp_size != 0:
                text_pad_len = sp_size - (original_text_len % sp_size)
                start_pad_idx = latent_len_padded + original_text_len
                attention_mask[:, start_pad_idx:] = False
                
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        if sp_size > 1:
            seq_len = hidden_states.shape[1]
            chunk_size = seq_len // sp_size
            start_idx = sp_rank * chunk_size
            end_idx = start_idx + chunk_size
            hidden_states = hidden_states[:, start_idx:end_idx, :].contiguous()
            
            # Shard RoPE
            cos, sin = image_rotary_emb
            cos = cos[start_idx:end_idx, :]
            sin = sin[start_idx:end_idx, :]
            image_rotary_emb = (cos, sin)
            
            # Shard Text
            text_len = encoder_hidden_states_proj.shape[1]
            t_chunk_size = text_len // sp_size
            t_start = sp_rank * t_chunk_size
            t_end = t_start + t_chunk_size
            encoder_hidden_states_proj = encoder_hidden_states_proj[:, t_start:t_end, :].contiguous()
            
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for b_idx, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states_proj = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states_proj,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states_proj = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states_proj,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states_proj = block(
                    hidden_states,
                    encoder_hidden_states_proj,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states_proj = block(
                    hidden_states,
                    encoder_hidden_states_proj,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        
        if sp_size > 1:
            gather_list = [torch.zeros_like(hidden_states) for _ in range(sp_size)]
            dist.all_gather(gather_list, hidden_states, group=sp_group)
            hidden_states = torch.cat(gather_list, dim=1)
            
            if original_seq_len % sp_size != 0:
                hidden_states = hidden_states[:, :original_seq_len, :]

        hidden_states = hidden_states[:, -(post_patch_num_frames*post_patch_height*post_patch_width):]
        
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)