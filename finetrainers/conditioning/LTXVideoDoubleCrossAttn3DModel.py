
import math
from typing import Any, Dict, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from diffusers import LTXVideoTransformer3DModel
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models import Attention
from diffusers.models.transformers.transformer_ltx import LTXVideoAttentionProcessor2_0
from diffusers.models.attention import FeedForward

logger = logging.get_logger(__name__)

@maybe_allow_in_graph
class LTXVideoTransformerDoubleCrossAttn3DModel(LTXVideoTransformer3DModel):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    def __init__(self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True):

        super().__init__(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    num_layers=num_layers,
                    activation_fn=activation_fn,
                    qk_norm=qk_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    caption_channels=caption_channels,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias
                )
        
        inner_dim = num_attention_heads * attention_head_dim
        
        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerDoubleCrossAttentionBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )


        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_attention_mask: torch.Tensor,
            num_frames: int,
            height: int,
            width: int,
            rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ) -> torch.Tensor:
          
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

            image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            batch_size = hidden_states.size(0)
            hidden_states = self.proj_in(hidden_states)

            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )

            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)


            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)
    



def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out



@maybe_allow_in_graph
class LTXVideoTransformerDoubleCrossAttentionBlock(nn.Module):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        # Pose condition + img ref block
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn3 = Attention(query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
            )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pose_img_ref_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states_text_cond = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )

        attn_hidden_states_pose_img_ref_cond = self.attn3(
            hidden_states,
            encoder_hidden_states=pose_img_ref_hidden_states,
            image_rotary_emb=None,
            attention_mask=None,
        )

        hidden_states_text_cond = hidden_states + attn_hidden_states_text_cond 
        norm_hidden_states_text_cond = self.norm2(hidden_states_text_cond) * (1 + scale_mlp) + shift_mlp

        hidden_states_pose_img_ref_cond = hidden_states + attn_hidden_states_pose_img_ref_cond
        norm_hidden_states_imf_ref_cond = hidden_states + self.norm3(hidden_states_pose_img_ref_cond)

        ff_output = self.ff(norm_hidden_states_text_cond + norm_hidden_states_imf_ref_cond)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states