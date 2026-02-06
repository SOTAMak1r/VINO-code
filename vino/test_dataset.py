import ast
import math
import os
import random
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union, Callable

import cv2
import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset


class VINODataset(TorchDataset):
    def __init__(self, config: SimpleNamespace):
        self._cfg = self._sanitize_config(config)

        self._spatial_unit = self._cfg.vae_spatial_scale_factor * self._cfg.transformer_path_size
        self._max_pixels   = self._cfg.height * self._cfg.width
        self._normalize_op = torch.jit.script(transforms.Normalize([127.5], [127.5]))
        
        self._data_source  = HFDataset.from_pandas(pd.read_json(self._cfg.json_path))

    @staticmethod
    def _sanitize_config(cfg: SimpleNamespace) -> SimpleNamespace:
        unit = cfg.vae_spatial_scale_factor * cfg.transformer_path_size
        if cfg.height % unit != 0 or cfg.width % unit != 0:
            raise ValueError(f"Dimensions ({cfg.height}x{cfg.width}) must differ by unit {unit}")

        if (cfg.num_frames - 1) % cfg.vae_temporal_scale_factor != 0:
            raise ValueError(f"Frame count mismatch for temporal scale {cfg.vae_temporal_scale_factor}")
        
        if not hasattr(cfg, 'json_path') or not cfg.json_path:
            raise ValueError("Missing JSON path")
        return cfg

    @staticmethod
    def _calc_target_resolution(w: float, h: float, max_area: int) -> Tuple[int, int]:
        ratio = w / h
        h_new = math.sqrt(max_area / ratio)
        return round(h_new), round(h_new * ratio)

    @staticmethod
    def _solve_optimal_grid(src_w: int, src_h: int, target_area: int, unit: int) -> Tuple[int, int]:
        r_tgt = src_w / src_h
        h_approx = math.sqrt(target_area / r_tgt)
        k_base = max(1, int(round(h_approx / unit)))

        best_metric = (float('inf'), float('inf'), 0, 0) 

        search_space = range(k_base - 256, k_base + 257)

        for k in search_space:
            if k <= 0: continue

            h_can = k * unit
            w_raw = r_tgt * h_can
            
            w_candidates = [
                max(unit, int(math.floor(w_raw / unit)) * unit),
                max(unit, int(math.ceil(w_raw / unit)) * unit)
            ]
            
            for w_can in set(w_candidates):
                area_err = abs((h_can * w_can) - target_area)
                ratio_err = abs((w_can / h_can) - r_tgt)
                
                if (area_err, ratio_err) < best_metric[:2]:
                    best_metric = (area_err, ratio_err, h_can, w_can)

        return best_metric[2], best_metric[3]

    def _ingest_video(self, fpath: str) -> Tuple[torch.Tensor, float]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[dataset error] can not find {fpath}")

        # Phase 1: Metadata Probe
        probe = cv2.VideoCapture(fpath)
        if not probe.isOpened():
            raise IOError(f"[dataset error] can not open {fpath}")
        h_raw, w_raw = probe.get(cv2.CAP_PROP_FRAME_HEIGHT), probe.get(cv2.CAP_PROP_FRAME_WIDTH)
        probe.release()

        # Phase 2: Decode Setup
        tgt_h, tgt_w = self._calc_target_resolution(w_raw, h_raw, self._max_pixels)
        vr = decord.VideoReader(fpath, ctx=decord.cpu(0), width=tgt_w, height=tgt_h)

        # Phase 3: Frame Schedule Logic
        v_len, v_fps = len(vr), vr.get_avg_fps()
        n_req = self._cfg.num_frames

        # Clamp frames to temporal scale alignment
        n_out = min(n_req, v_len)
        n_out = (n_out - 1) // self._cfg.vae_temporal_scale_factor * self._cfg.vae_temporal_scale_factor + 1

        # Stride calculation
        if n_out == 1 or self._cfg.sample_fps == -1:
            stride = 1
        else:
            dyn_stride = max(1, round(v_fps / self._cfg.sample_fps))
            cap_stride = max(1, round((v_len - 1) / (n_out - 1)))
            stride = min(dyn_stride, cap_stride)

        # Start position strategies
        span = (n_out - 1) * stride + 1
        valid_start_max = max(0, v_len - span)
        
        pick_fn   = lambda _: 0
        start_idx = pick_fn(valid_start_max)
        indices   = range(start_idx, start_idx + n_out * stride, stride)

        # Phase 4: Extraction
        if max(indices) >= v_len:
             raise AssertionError(f"Frame index overflow: {max(indices)} >= {v_len}")

        buffer = torch.as_tensor(vr.get_batch(indices).asnumpy())

        if len(buffer) != n_out:
            raise AssertionError(f"Frame count mismatch: {len(buffer)} != {n_out}")

        return buffer, v_fps

    def _transform_tensor(self, pixel_tensor: torch.Tensor, perform_resize: bool) -> torch.Tensor:
        x = pixel_tensor.float().permute(0, 3, 1, 2)
        _, _, h, w = x.shape

        if perform_resize:
            th, tw = self._calc_target_resolution(w, h, self._max_pixels)
            x = F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=False, antialias=True)
            curr_h, curr_w = th, tw
        else:
            curr_h, curr_w = h, w

        u = self._spatial_unit
        valid_h = (curr_h // u) * u
        valid_w = (curr_w // u) * u
        
        x = x[:, :, :valid_h, :valid_w]
        
        return self._normalize_op(x)

    def _letterbox_image(self, img: Image.Image) -> Image.Image:
        w_curr, h_curr = img.size
        
        # Solve strict grid constraints
        h_box, w_box = self._solve_optimal_grid(w_curr, h_curr, self._max_pixels, self._spatial_unit)
        
        scale_f = min(w_box / w_curr, h_box / h_curr)
        w_rsz, h_rsz = round(w_curr * scale_f), round(h_curr * scale_f)
        
        # Composite
        canvas = Image.new("RGB", (w_box, h_box), (255, 255, 255))
        canvas.paste(
            img.resize((w_rsz, h_rsz), Image.LANCZOS), 
            ((w_box - w_rsz) // 2, (h_box - h_rsz) // 2)
        )
        return canvas

    def __len__(self) -> int:
        return len(self._data_source)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._data_source[index]
        
        # Base schema
        sample = {
            "index": row['index'],
            "task": row.get('task'), 
            "caption": row['caption'],
        }

        # Branch 1: Video Reference
        if 'ref_video_path' in row and row['ref_video_path'] is not None:
            v_path = row['ref_video_path']
            raw_frames, fps = self._ingest_video(v_path)
            proc_frames = self._transform_tensor(raw_frames, perform_resize=False)
            
            sample.update({
                "ref_video_path": v_path,
                "ref_video": proc_frames,
            })

        # Branch 2: Image Reference
        if 'ref_image_paths' in row and row['ref_image_paths'] is not None:
            # Safe parsing
            p_val = row['ref_image_paths']
            
            if isinstance(p_val, list):
                paths = p_val
            else:
                raise TypeError(f"ref_image_paths expects list, got {type(p_val)}")

            # Process stack
            img_stack = []
            for p in paths:
                with Image.open(p) as f:
                    img_stack.append(self._letterbox_image(f.convert("RGB")))

            sample.update({
                "ref_image_paths": str(paths), 
                "ref_image_pils": img_stack,
                "ref_image_have_base": row.get('ref_image_have_base', False)
            })

        return sample
