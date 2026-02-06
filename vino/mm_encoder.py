
import math
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoTokenizer, Qwen3VLProcessor

from vino.vision_process import process_vision_info


class Qwen3VLTextRMSNorm(nn.Module):
    """RMSNorm used by Qwen3-VL text tower (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VINOConnector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert getattr(config, "connector_type", None) == "mlp", "Only mlp connector is supported."

        self.norm_qwen = Qwen3VLTextRMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.z_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.z_dim * 4, config.z_dim),
        )

    def forward(self, inputs_embeds: torch.Tensor, inputs_embeds_mask: torch.Tensor):
        hidden_states = self.norm_qwen(inputs_embeds)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


try:
    from typing_extensions import Unpack
except Exception:
    Unpack = dict 

try:
    # Newer Transformers layout (expected)
    from transformers.cache_utils import Cache
    from transformers.utils import is_torchdynamo_compiling
    from transformers.modeling_utils import PreTrainedModel  # noqa: F401 (imported for type parity)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel as HFQwen3VLModel,
        Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
        Qwen3VLModelOutputWithPast,
        Qwen3VLCausalLMOutputWithPast,
        check_model_inputs,
        TransformersKwargs,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import Qwen3-VL from transformers. Please ensure you have a recent transformers version "
        "that provides `transformers.models.qwen3_vl`."
    ) from e


class Qwen3VLModel(HFQwen3VLModel):
    """Patched Qwen3VLModel that supports `metaquery`.

    Only the `forward` differs from the upstream Transformers implementation.
    """

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        metaquery: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # -------------------------
        # MetaQuery injection (diff vs upstream)
        # -------------------------
        if metaquery is not None:
            if attention_mask is None:
                raise ValueError("metaquery requires `attention_mask` to be provided.")
            batch_size = attention_mask.shape[0]
            metaquery = metaquery[None].repeat(batch_size, 1, 1)

            metaquery_position_ids = (
                torch.arange(0, metaquery.shape[1])[None, None]
                .repeat(3, batch_size, 1)
                .to(position_ids)
                + position_ids[:, :, -1:]
            )
            position_ids = torch.cat([position_ids, metaquery_position_ids], dim=-1)

            metaquery_attention_mask = attention_mask.new_ones(batch_size, metaquery.shape[1])
            attention_mask = torch.cat([attention_mask, metaquery_attention_mask], dim=-1)

            inputs_embeds = torch.cat([inputs_embeds, metaquery.to(inputs_embeds)], dim=-2)

            if visual_pos_masks is not None:
                metaquery_visual_pos_masks = visual_pos_masks.new_zeros(batch_size, metaquery.shape[1])
                visual_pos_masks = torch.cat([visual_pos_masks, metaquery_visual_pos_masks], dim=-1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration):
    """Patched Qwen3VLForConditionalGeneration that supports `metaquery`.

    Only the `__init__` (to use patched model) and `forward` signature differ.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace the base model with our patched version BEFORE weights are loaded by from_pretrained.
        self.model = Qwen3VLModel(config)
        # Re-run post_init to ensure weight tying is consistent with the patched model.
        if hasattr(self, 'post_init'):
            self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        metaquery: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            metaquery=metaquery,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )


# ============================================================================
# Wrapper used by VINO
# ============================================================================

class VINOQwen3VL(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.ckpt_path,
            padding_side="left",
            trust_remote_code=True,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            config.ckpt_path,
            padding_side="left",
        )
        self.mm_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            config.ckpt_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        chat_template_new = (
            "{% set image_counter = namespace(value=0) %}{% set video_counter = namespace(value=0) %}"
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant aimed at generating an output video caption.<|im_end|>\n{% endif %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_counter.value = image_counter.value + 1 %}Image {{ image_counter.value }}: <|vision_start|><|image_pad|><|vision_end|>"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_counter.value = video_counter.value + 1 %}Video {{ video_counter.value }}: <|vision_start|><|video_pad|><|vision_end|>"
            "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
            "{% endfor %}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        self.processor.chat_template = chat_template_new

        if getattr(config, "metaquery_config", None) is not None:
            num_queries = config.metaquery_config.num_queries
            hidden_size = self.mm_encoder.model.language_model.config.hidden_size
            self.metaquery = nn.Parameter(torch.zeros(num_queries, hidden_size))
            nn.init.normal_(self.metaquery, std=1 / math.sqrt(hidden_size))
        else:
            self.metaquery = None

        if getattr(config, "connector_config", None) is not None:
            self.connector = VINOConnector(config.connector_config)
            connector_config_vae_branch = deepcopy(config.connector_config)
            connector_config_vae_branch.z_dim = connector_config_vae_branch.hy_dim
            self.connector_vae_branch = VINOConnector(connector_config_vae_branch)
        else:
            self.connector = None
            self.connector_vae_branch = None

    @torch.no_grad()
    def get_embed_mm(
        self,
        prompt,
        ref_image_paths,
        ref_video_path,  # NOTE: only support input video num = 1
        vlm_cond_image_path_to_pil_image_dict=None,
        device=None,
    ):
        """Compute multimodal embeddings for VINO.

        Args:
            prompt: str
            ref_image_paths: str like "['a.png','b.png']" or []
            ref_video_path: path or None
            vlm_cond_image_path_to_pil_image_dict: optional {path: PIL.Image} cache
        """
        if vlm_cond_image_path_to_pil_image_dict is None:
            vlm_cond_image_path_to_pil_image_dict = {}

        vlm_cond_image_numbers = []
        messages = []

        if ref_image_paths is None:
            ref_image_paths = []
        elif isinstance(ref_image_paths, str):
            assert ref_image_paths.startswith("[") and ref_image_paths.endswith("]"), "ref_image_paths format is wrong!"
            ref_image_paths = eval(ref_image_paths)
        elif isinstance(ref_image_paths, (list, tuple)):
            # allow direct list in case caller already parsed it
            ref_image_paths = list(ref_image_paths)
        else:
            raise ValueError("ref_image_paths format is wrong!")

        vlm_cond_image_numbers.append(len(ref_image_paths))
        content = [
            {
                "type": "image",
                "image": vlm_cond_image_path_to_pil_image_dict.get(image_path, image_path),
            }
            for image_path in ref_image_paths
        ]

        if ref_video_path is not None:
            content = content + [
                {
                    "type": "video",
                    "video": ref_video_path,
                    "min_pixels": 256 * 32 * 32,
                    "max_pixels": 16384 * 32 * 32,
                }
            ]

        if prompt is not None:
            content = content + [{"type": "text", "text": f"{prompt}"}]

        messages.append([{"role": "user", "content": content}])

        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"[check texts] {texts}")

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        batch_dict = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs=video_kwargs,
        )

        input_ids = batch_dict["input_ids"]
        attn_mask = batch_dict["attention_mask"]
        id_vs = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        id_ve = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        mask_vs = (input_ids == id_vs) & (attn_mask == 1)
        mask_ve = (input_ids == id_ve) & (attn_mask == 1)
        pos_vs = [torch.where(mask_vs[i])[0].tolist() for i in range(input_ids.size(0))]
        pos_ve = [torch.where(mask_ve[i])[0].tolist() for i in range(input_ids.size(0))]

        prompt_attention_mask = batch_dict["attention_mask"].to(device)

        if self.metaquery is not None:
            batch_size = prompt_attention_mask.shape[0]
            metaquery_prompt_attention_mask = prompt_attention_mask.new_ones(batch_size, self.metaquery.shape[0])
            prompt_attention_mask = torch.cat([prompt_attention_mask, metaquery_prompt_attention_mask], dim=-1)

        out = self.mm_encoder(**batch_dict.to(device), output_hidden_states=True, metaquery=self.metaquery)
        prompt_embeds = out["hidden_states"][-2]

        if self.connector is not None:
            prompt_embeds_vlm_branch = self.connector(prompt_embeds, prompt_attention_mask)
            prompt_embeds_vae_branch = self.connector_vae_branch(prompt_embeds, prompt_attention_mask)
        else:
            prompt_embeds_vlm_branch = prompt_embeds
            prompt_embeds_vae_branch = prompt_embeds

        for idx, vlm_cond_image_number in enumerate(vlm_cond_image_numbers):
            pos_vs[idx] = pos_vs[idx][: vlm_cond_image_number + 1]
            pos_ve[idx] = pos_ve[idx][:vlm_cond_image_number] + pos_ve[idx][-1:]

        return [prompt_embeds_vlm_branch, prompt_embeds_vae_branch], prompt_attention_mask, pos_vs, pos_ve
