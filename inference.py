import ast
import io
import os
import random
import sys
import inspect
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import fire
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

sys.path.append(os.getcwd())


from vino.mm_encoder import VINOQwen3VL
from vino.mmdit_sp_sage import VINOTransformer
from vino.test_dataset import VINODataset

from diffusers import AutoencoderKLHunyuanVideo as VINOVAE



# ==============================================================================
# Configuration Classes
# ==============================================================================

@dataclass
class VINOConfig:
    """Base configuration class with update utility."""
    def update(self, cfg: Dict[str, Any], strict: bool = True):
        for k, v in cfg.items():
            if not hasattr(self, k):
                if strict:
                    raise KeyError(f"Unknown config key: {k}")
                else:
                    continue
            setattr(self, k, v)
        return self


@dataclass
class VINODatasetConfig(VINOConfig):
    json_path: Optional[str] = None
    height: int = 480
    width:  int = 848
    num_frames: int = 85
    vae_temporal_scale_factor: int = 4
    vae_spatial_scale_factor:  int = 8
    transformer_path_size: int = 2
    sample_fps: int = -1


@dataclass
class VINOMetaQueryConfig(VINOConfig):
    num_queries: int = 256


@dataclass
class VINOConnectorConfig(VINOConfig):
    connector_type: str = 'mlp'
    hidden_size: int = 2560
    z_dim: int = 4096
    hy_dim: int = 3072


@dataclass
class VINOQwen3VLConfig(VINOConfig):
    ckpt_path: Optional[str] = None
    metaquery_config: VINOMetaQueryConfig = field(default_factory=VINOMetaQueryConfig)
    connector_config: VINOConnectorConfig = field(default_factory=VINOConnectorConfig)


@dataclass
class VINOVAEConfig(VINOConfig):
    ckpt_path: Optional[str] = None


@dataclass
class VINOTransformerConfig(VINOConfig):
    ckpt_path: Optional[str] = None


@dataclass
class VINOInferenceConfig(VINOConfig):
    guidance_scale: float = 7.0
    guidance_scale_image: float = 1.0
    negative_prompt: str = "low quality, blurry, low resolution, jpeg artifacts, noisy, bad anatomy, bad proportions, deformed face, extra limbs, extra fingers, fused fingers, malformed hands, unnatural pose, distorted eyes, poor lighting, overexposed, underexposed, bad composition, cropped, duplicate, ghosting, motion blur, watermark, logo, text, nsfw"
    negative_prompt_video: str = "low quality, blurry, flickering, temporal inconsistency, ghosting, motion blur, jitter, unstable structure, distorted anatomy, extra limbs, extra fingers, malformed hands, warped face, inconsistent identity, poor lighting, overexposed, underexposed, bad composition, cropped, duplicate frames, color flicker, watermark, logo, text, nsfw"
    num_inference_steps: int = 50
    timestep_shift: float = 5.0
    height: int = 640
    width: int = 640
    num_frames: int = 85


# ==============================================================================
# Pipeline Logic (Integrated)
# ==============================================================================

class ModelInputPayload:
    """
    Internal data class to manage Transformer input arguments to reduce dictionary chaos.
    """
    def __init__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        vae_conds: List[Any],
        pos_vs: Any,
        pos_ve: Any
    ):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.vae_cond_dict = {
            'vae_conds': vae_conds,
            'pos_vs': pos_vs,
            'pos_ve': pos_ve,
        }


class VINOInferencePipeline(nn.Module):
    def __init__(self, config, transformer, mm_encoder, vae):
        super().__init__()
        self.config = config
        self.transformer = transformer 
        self.mm_encoder  = mm_encoder 
        self.vae         = vae

    def _init_gaussian_latents(
        self, 
        bs: int, 
        ch: int, 
        frames: int, 
        h: int, 
        w: int, 
        generator: Optional[torch.Generator], 
        device: torch.device
    ) -> torch.Tensor:
        """
        Initialize Gaussian noise latents, including distributed synchronization logic.
        """
        # Calculate compression dimensions
        t_compress = self.vae.config.temporal_compression_ratio
        s_compress = self.vae.config.spatial_compression_ratio
        
        reduced_frames = (frames - 1) // t_compress + 1
        reduced_h = h // s_compress
        reduced_w = w // s_compress

        shape = (bs, ch, reduced_frames, reduced_h, reduced_w)
        
        # Strictly maintain the original type: bfloat16
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.bfloat16)

        # Maintain distributed broadcast logic
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(latents, src=0)
            
        return latents

    def _parse_image_paths(self, path_str: Optional[str]) -> List[str]:
        """Safely parse image path strings."""
        if not path_str:
            return []
        try:
            return eval(path_str)
        except Exception:
            return []

    def _encode_visual_context(self, path_list: List[str], img_map: Dict, device: torch.device) -> List[torch.Tensor]:
        """Encode reference images into VAE Latents."""
        encoded_latents = []
        for path in path_list:
            if path not in img_map:
                raise ValueError(f"Image {path} missing from provided dictionary.")
            
            # Image Preprocessing: [H, W, C] -> [1, C, 1, H, W]
            raw_np = np.array(img_map[path])
            tensor = torch.from_numpy(raw_np)[None, None].permute(0, 4, 1, 2, 3)
            tensor = (tensor / 127.5 - 1.0).to(dtype=self.vae.dtype, device=device)

            # VAE Encoding
            dist_sample = self.vae.encode(tensor).latent_dist.sample()
            scaled_latent = dist_sample.to(dtype=tensor.dtype) * self.vae.config.scaling_factor
            encoded_latents.append(copy.deepcopy(scaled_latent))
        return encoded_latents

    def _resolve_condition_inputs(
        self,
        prompt: Optional[str],
        is_video_task: Optional[str],
        img_paths_str: Optional[str],
        vid_path: Optional[str],
        vid_tensor: Optional[torch.Tensor],
        img_map: Dict,
        device: torch.device
    ) -> ModelInputPayload:
        """
        Unified processing for Positive, Negative-Text, and Negative-Image embedding generation.
        """
        if prompt is None:
            if is_video_task:
                prompt = self.config.negative_prompt_video
            else:
                prompt = self.config.negative_prompt
                
        # 1. Get Multimodal Embeddings (Text + Vision Info)
        embeds, mask, pos_vs, pos_ve = self.mm_encoder.get_embed_mm(
            prompt, img_paths_str, vid_path, 
            vlm_cond_image_path_to_pil_image_dict=img_map, 
            device=device
        )

        # 2. Get VAE Latents (Visual Signals)
        # Parse paths
        path_list = self._parse_image_paths(img_paths_str)
        vae_latents_list = self._encode_visual_context(path_list, img_map, device)

        # Handle Reference Video (if exists)
        if vid_path is not None and vid_tensor is not None:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            vid_input = vid_tensor.to(dtype=self.vae.dtype, device=device)[None].permute(0, 2, 1, 3, 4)
            vid_encoded = self.vae.encode(vid_input).latent_dist.sample()
            vid_encoded = vid_encoded.to(dtype=vid_input.dtype) * self.vae.config.scaling_factor
            vae_latents_list.append(vid_encoded)

        # Wrap in list (matching original code structure)
        final_vae_conds = [vae_latents_list] if vae_latents_list else []
        
        return ModelInputPayload(embeds, mask, final_vae_conds, pos_vs, pos_ve)

    def _adjust_uncond_vae_payload(self, payload: ModelInputPayload, ref_payload: ModelInputPayload):
        """
        Reproduce the VAE conds slicing logic for the Uncond branch.
        """
        adjusted_conds = []
        for idx, cond_list in enumerate(ref_payload.vae_cond_dict['vae_conds']):
            # Get the length of pos_vs in the current payload (negative)
            current_vs_len = len(payload.vae_cond_dict['pos_vs'][idx])
            # Slice
            adjusted_conds.append(cond_list[-current_vs_len:])

        payload.vae_cond_dict['vae_conds'] = adjusted_conds

    def _compute_guided_velocity(
        self,
        v_cond: torch.Tensor,
        v_uncond: Optional[torch.Tensor],
        v_uncond_img: Optional[torch.Tensor],
        scale_txt: float,
        scale_img: float
    ) -> torch.Tensor:
        """Pure math function: Calculate the velocity vector after CFG combination."""
        # Case 1: Text Guidance Only (Standard CFG)
        if v_uncond is not None and v_uncond_img is None:
            return v_uncond + scale_txt * (v_cond - v_uncond)

        # Case 2: Image + Text Guidance (Double CFG)
        # Formula: Uncond_Img + Scale_Img * (Uncond - Uncond_Img) + Scale_Txt * (Text - Uncond)
        if v_uncond_img is not None:
            term1 = v_uncond_img
            term2 = scale_img * (v_uncond - v_uncond_img)
            term3 = scale_txt * (v_cond - v_uncond)
            return term1 + term2 + term3
        
        # Case 3: No Guidance
        return v_cond

    def _get_scheduler_timesteps(self, steps: int, shift: float, device: torch.device):
        """Generate timestep sequence."""
        raw_times = torch.linspace(1.0, 0, steps + 1, device=device)
        # Time shifting formula
        warped_times = shift * raw_times / (1 - raw_times + shift * raw_times)
        dts = warped_times[:-1] - warped_times[1:]
        timesteps = warped_times[:-1]
        return timesteps, dts

    @torch.no_grad()
    def __call__(
        self,
        guidance_scale: float = 7.0,
        guidance_scale_image: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 1,
        prompt: str = None,
        ref_image_paths: str = None,
        ref_image_pils: Optional[List[Image.Image]] = None,
        ref_image_have_base: bool = False,
        ref_video_path: str = None,
        ref_video: torch.FloatTensor = None,
        task: str = None,
        vlm_cond_image_path_to_pil_image_dict: dict = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # --- 1. Environment & Config Preparation ---
        device        = self.transformer.device
        use_text_cfg  = guidance_scale > 1.0
        use_image_cfg = guidance_scale_image > 1.0

        # Validation
        assert isinstance(prompt, str)
        assert (ref_image_paths is None) or isinstance(ref_image_paths, str)

        # --- 2. Build Model Conditions ---
        
        # A. Positive Condition
        pos_inputs = self._resolve_condition_inputs(
            prompt, num_frames>1, ref_image_paths, ref_video_path, ref_video, 
            vlm_cond_image_path_to_pil_image_dict, device
        )

        # B. Negative Text Condition (Uncond)
        neg_text_inputs = None
        if use_text_cfg:
            neg_text_inputs = self._resolve_condition_inputs(
                None, num_frames>1, ref_image_paths, ref_video_path, ref_video,
                vlm_cond_image_path_to_pil_image_dict, device
            )
            # Apply slicing logic to fix vae_conds
            self._adjust_uncond_vae_payload(neg_text_inputs, pos_inputs)

        # C. Negative Image Condition
        neg_img_inputs = None
        if use_image_cfg:
            # Handle ref_image_have_base logic: Keep only the last image as base
            img_paths_for_neg = ref_image_paths
            if ref_image_have_base:
                paths = self._parse_image_paths(ref_image_paths)
                # if len(paths) > 1:
                assert len(paths) > 1
                img_paths_for_neg = str(paths[-1:])
            else:
                img_paths_for_neg = None

            neg_img_inputs = self._resolve_condition_inputs(
                None, num_frames>1, img_paths_for_neg, ref_video_path, ref_video,
                vlm_cond_image_path_to_pil_image_dict, device
            )
            # Apply slicing logic, referencing pos_inputs latents
            self._adjust_uncond_vae_payload(neg_img_inputs, pos_inputs)

        # --- 3. Initialize Latents ---
        latents = self._init_gaussian_latents(
            bs=1,  # Hardcoded
            ch=self.transformer.config.in_channels,
            frames=num_frames,
            h=height,
            w=width,
            generator=generator,
            device=device
        )

        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f'[latents] {latents.shape}')

        # --- 4. Prepare Scheduler ---
        timesteps, delta_ts = self._get_scheduler_timesteps(
            num_inference_steps, self.config.timestep_shift, device
        )

        # --- 5. Denoising Loop ---
        # Set up group for distributed environment
        world_group = dist.group.WORLD if (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1) else None
        is_master = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

        loop_iter = tqdm(enumerate(timesteps), total=len(timesteps), disable=not is_master)

        for i, t_val in loop_iter:
            # Construct input
            current_t = t_val.view(1).expand(latents.size(0)).to(device)
            # Original code used a hardcoded 1000.0 guidance tensor
            dummy_guidance = torch.full((1,), 1000.0, device=device)

            # Define forward pass closure
            def forward_model(payload: ModelInputPayload):
                return self.transformer(
                    latents,
                    timestep=current_t * 999,
                    encoder_hidden_states=payload.hidden_states,
                    encoder_attention_mask=payload.attention_mask,
                    vae_cond_dict=payload.vae_cond_dict,
                    guidance=dummy_guidance,
                    return_dict=False,
                    sp_group=world_group,
                )[0]

            # 1. Compute Positive
            velocity_pos = forward_model(pos_inputs)
            
            # 2. Compute Negative Text (if needed)
            velocity_uncond = None
            if use_text_cfg:
                velocity_uncond = forward_model(neg_text_inputs)
            
            # 3. Compute Negative Image (if needed)
            velocity_uncond_img = None
            if use_image_cfg:
                velocity_uncond_img = forward_model(neg_img_inputs)

            # 4. Fuse Velocity Fields (CFG)
            final_velocity = self._compute_guided_velocity(
                velocity_pos, velocity_uncond, velocity_uncond_img,
                guidance_scale, guidance_scale_image
            )

            # 5. Update Latents (Euler Step)
            latents = latents - delta_ts[i] * final_velocity

        # --- 6. Video Decode ---
        # Normalize scaling
        z = latents / self.vae.config.scaling_factor
        
        # Check if VAE supports direct temporal decoding
        forward_params = inspect.signature(self.vae.forward).parameters
        if "num_frames" in forward_params:
            video = self.decode_latents_with_temporal_decoder(z)
        else:
            video = self.vae.decode(z, return_dict=False)[0]

        return video


# ==============================================================================
# Helper Functions & Main
# ==============================================================================

def load_checkpoint_weights(model: torch.nn.Module, state_dict: Dict[str, Any], prefix: str):
    """
    Helper to load weights that might have a specific prefix (e.g. from a wrapped training model).
    """
    filtered_state_dict = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = key.replace(prefix, '', 1)
            filtered_state_dict[new_key] = state_dict[key]
    
    if len(filtered_state_dict) > 0:
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"[Warning] Loading {prefix} weights: {len(unexpected)} unexpected keys found.")


def process_batch(pipeline: VINOInferencePipeline, batch: Dict[str, Any], generator: torch.Generator) -> np.ndarray:
    """
    Process a single batch to generate video/image content.
    """
    task = batch['task']
    
    # Prepare Visual Language Model condition images
    vlm_cond_image_path_to_pil_image_dict = {}
    if "ref_image_paths" in batch:
        # Securely parse string representation of list
        try:
            paths = ast.literal_eval(batch["ref_image_paths"]) if isinstance(batch["ref_image_paths"], str) else batch["ref_image_paths"]
        except (ValueError, SyntaxError):
            # Fallback for simple list strings or if already a list
            paths = eval(batch["ref_image_paths"])
            
        for ref_image_path, ref_image_pil in zip(paths, batch["ref_image_pils"]):
            vlm_cond_image_path_to_pil_image_dict[ref_image_path] = ref_image_pil

    # Dynamic resolution adjustment based on reference inputs
    if task in ['ti2i', 'i2v']:
        if "ref_image_pils" not in batch:
            raise ValueError(f"Task {task} requires 'ref_image_pils' in batch.")
        base_image = batch["ref_image_pils"][-1]
        pipeline.config.width  = base_image.size[0]
        pipeline.config.height = base_image.size[1]
        if task == 'ti2i':
            pipeline.config.num_frames = 1
            
    if task in ['tv2v', 'tiv2v']:
        if "ref_video" not in batch:
            raise ValueError(f"Task {task} requires 'ref_video' in batch.")
        pipeline.config.width      = batch['ref_video'].shape[-1]
        pipeline.config.height     = batch['ref_video'].shape[-2]
        pipeline.config.num_frames = batch['ref_video'].shape[-4]
        

    # Run Inference
    output = pipeline(
        guidance_scale=pipeline.config.guidance_scale,
        guidance_scale_image=pipeline.config.guidance_scale_image,
        height=pipeline.config.height,
        width=pipeline.config.width,
        num_frames=pipeline.config.num_frames,
        num_inference_steps=pipeline.config.num_inference_steps,
        prompt=batch['caption'],
        ref_image_paths=batch.get('ref_image_paths', None),
        ref_image_pils=batch.get('ref_image_pils', None),
        ref_image_have_base=batch.get('ref_image_have_base', None),
        ref_video_path=batch.get('ref_video_path', None),
        ref_video=batch.get('ref_video', None),
        task=task,
        vlm_cond_image_path_to_pil_image_dict=vlm_cond_image_path_to_pil_image_dict,
        generator=generator, 
    )

    # Post-process: Normalize to [0, 255] and rearrange for saving
    output = (output.detach().cpu().to(torch.float32).numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    output = rearrange(output, "b c t h w -> b t h w c")

    return output


def save_video_to_buffer(images: np.ndarray, fps: int = 24, format: str = "mp4", crf: int = 12) -> bytes:
    """Encodes a sequence of images into a video buffer."""
    video_stream = io.BytesIO()
    ffmpeg_params = ["-crf", str(crf)]
    
    with imageio.get_writer(
        video_stream, 
        fps=fps, 
        format=format, 
        codec="libx264", 
        ffmpeg_params=ffmpeg_params, 
        pixelformat="yuv420p"
    ) as writer:
        for frame in images:
            writer.append_data(frame)

    return video_stream.getvalue()


@torch.no_grad()
def main(
    # Model Paths
    ckpt_path:        str = './checkpoints/SOTAMak1r/VINO-weight/vino.ckpt',
    mm_encoder_path:  str = './checkpoints/Qwen/Qwen3-VL-4B-Instruct',
    vae_path:         str = './checkpoints/hunyuanvideo-community/HunyuanVideo',
    transformer_path: str = './checkpoints/hunyuanvideo-community/HunyuanVideo',
    
    # Dataset Config
    json_path: str = '',

    # Inference Config
    seed: int = None,
    output_path: str = './output',
    output_height: int = 640,
    output_width: int = 640,
    output_num_frames: int = 1,
    guidance_scale: float = 7.0,
    guidance_scale_image: float = 1.0,
    negative_prompt: str = None,
    negative_prompt_video: str = None,
):
    """
    Main inference entry point for VINO model generation.
    """
    # ---- 1. Distributed Environment Setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        print(f"[Init] Process group initialized on rank {local_rank}.")
    
    device = torch.device(f"cuda:{local_rank}")
    output_num_frames = min(89, max(1, output_num_frames))

    # ---- 2. Initialize Data Pipeline ----
    dataset_cfg = VINODatasetConfig()
    dataset_cfg.update({
        "json_path": json_path,
        "num_frames": output_num_frames,
    })
    dataset = VINODataset(dataset_cfg)

    # ---- 3. Initialize Models ----
    print(f"[Init] Loading models...")
    
    # MM Encoder
    qwen_cfg = VINOQwen3VLConfig()
    qwen_cfg.update({"ckpt_path": mm_encoder_path})
    mm_encoder = VINOQwen3VL(qwen_cfg)

    # VAE
    vae_cfg = VINOVAEConfig()
    vae_cfg.update({"ckpt_path": vae_path})
    vae = VINOVAE.from_pretrained(vae_cfg.ckpt_path, subfolder='vae')

    # Transformer
    transformer_cfg = VINOTransformerConfig()
    transformer_cfg.update({"ckpt_path": transformer_path})
    
    # Load transformer config from pretrained folder
    tr_config_obj = VINOTransformer.load_config(transformer_cfg.ckpt_path, subfolder="transformer")
    transformer = VINOTransformer.from_config(tr_config_obj)

    # Inference Pipeline
    pipeline_cfg = VINOInferenceConfig()
    pipeline_cfg.update({
        "height": output_height,
        "width": output_width,
        "num_frames": output_num_frames,
        "guidance_scale": 5.0 if output_num_frames == 1 else 7.0,
        "guidance_scale_image": guidance_scale_image,
        "timestep_shift": 5.0 if output_num_frames == 1 else 7.0,
        "negative_prompt": negative_prompt if negative_prompt is not None else  pipeline_cfg.negative_prompt,
        "negative_prompt_video": negative_prompt_video if negative_prompt_video is not None else  pipeline_cfg.negative_prompt_video,
    })
    pipeline = VINOInferencePipeline(pipeline_cfg, transformer, mm_encoder, vae)
    
    # ---- 4. Load Model Weights ----
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"[Init] Loading checkpoint from {ckpt_path}")
    full_state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # Load specific sub-modules
    load_checkpoint_weights(pipeline.transformer, full_state_dict, "transformer.")
    load_checkpoint_weights(pipeline.mm_encoder, full_state_dict, "mm_encoder.")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

    # ---- 5. Generation Loop ----
    print(f"[Run] Starting inference on {len(dataset)} samples...")
    os.makedirs(output_path, exist_ok=True)

    for batch in tqdm(dataset, disable=(local_rank != 0)):
        # Setup Generator
        if seed == None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        # Run Inference
        video_output = process_batch(pipeline, batch, generator)
        
        # Save Results
        save_as_video = video_output.shape[1] > 1
        ext = '.mp4' if save_as_video else '.png'
        file_name = f"{batch['index']}{ext}"
        save_path = os.path.join(output_path, file_name)

        video_data = video_output[0] # Unbatch

        if save_as_video:
            video_bytes = save_video_to_buffer(video_data, fps=24)
            with open(save_path, "wb") as f:
                f.write(video_bytes)
        else:
            img_pil = Image.fromarray(video_data[0])
            img_pil.save(save_path)

    print("[Done] Inference finished.")


if __name__ == "__main__":
    fire.Fire(main)