
<p align="center">
<img src="assets/readme/logo.png" width="400"/>
<p>

<!-- <p align="center">
<h2 align="center">A Unified Visual Generator with Interleaved OmniModal Context</h2>
</p> -->

<p align="center">
💜 <a href="https://sotamak1r.github.io/VINO-web/"><b>Project Page</b></a> &nbsp&nbsp ｜ 
&nbsp&nbsp 📑 <a href="http://arxiv.org/abs/2601.02358"><b>Paper</b></a> &nbsp&nbsp  | 
&nbsp&nbsp🤗 <a href="https://huggingface.co/SOTAMak1r/VINO-weight"><b>Hugging Face</b></a>&nbsp&nbsp | 
&nbsp&nbsp 📺 <a href="[https://www.youtube.com/watch?v=4jXkEk3r2pw](https://www.youtube.com/watch?v=4jXkEk3r2pw)"><b>YouTube</b></a>&nbsp&nbsp
<br>

---

**[VINO: A Unified Visual Generator with Interleaved OmniModal Context](http://arxiv.org/abs/2601.02358)** 


Welcome to the official repository for **VINO**.

**VINO** is a unified visual generation framework that breaks down the barriers between image generation, video generation, and editing. Powered by a robust **Vision-Language Model (VLM)** and **Multi-Modal Diffusion Transformer (MMDiT)** architecture, VINO seamlessly interprets interleaved multi-modal inputs to achieve superior consistency and controllability.

**Key Features:**

* 👍 **All-in-One Unified Model**: A single model weight supports all tasks including Text-to-Image, Text-to-Video, Image-to-Video, and extensive Image/Video Editing.
* 👍 **OmniModal Context**: Deeply integrated with VLM to handle multi-image references, long-context instructions, and mixed-modal inputs for precise Instruction Following.
* 👍 **Advanced Control**: Supports sophisticated editing capabilities such as **Element Clone**, **Camera Control**, and **Style Transfer**.

## 🎬 Video Demo

<div align="center">
<video src="https://github.com/user-attachments/assets/82b91467-6b0f-424e-a086-75bc49781600" autoplay muted loop playsinline></video>
</div> 

## 🔥 News

* **[2026.02.09]** 🚀 We have officially released the **VINO** inference code and full model weights!
* **[2026.01.06]** 📑 The **VINO** paper is now available on [arXiv](http://arxiv.org/abs/2601.02358).
* **[2025.12.09]** 🌐 The project page is live.

## 🛠️ Installation

We recommend using Anaconda to create an isolated Python environment:

```bash
# Clone the repository
git clone https://github.com/SOTAMak1r/VINO-code.git
cd VINO-code

# Create environment
conda create -n vino python=3.10
conda activate vino

# Install dependencies
pip install -r requirements.txt
pip install "flash_attn==2.7.4.post1" --no-build-isolation

# [Optional] faster inference 🧨
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
python setup.py install
```

## 📥 Model Download

VINO uses a unified weight design, so you do not need to download different models for different tasks.

| Models | Download Link | Description |
| --- | --- | --- |
| **VINO** | 🤗 [Huggingface](https://huggingface.co/SOTAMak1r/VINO-weight) | Contains MMDiT weights and learnable tokens |
| **HunyuanVideo** | 🤗 [Huggingface](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Contains VAE weights |
| **Qwen3VL** | 🤗 [Huggingface](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Contains VLM weights |

We recommend using the script for automatic downloading:

```bash
python download.py --ak your_own_huggingface_ak
```

## 🚀 Inference & Quick Start

VINO supports various generation modes. We have integrated all functionalities into `inference.py`, allowing you to switch tasks easily by modifying parameters.

| Category | Command Flag | Task Name | Capability |
|--------|--------------|-----------|------------|
| **Generation** | `t2i` | Text → Image | image synthesis from natural language |
|  | `t2v` | Text → Video | text-to-video generation |
|  | `i2v` | Image → Video | Animate a single image with motion & dynamics |
|  | `ti2v` | Multi-Image → Video | Video generation conditioned on multiple reference images |
| **Editing** | `ti2i` | Text-Instructed Image Edit | Instruction-based image editing |
|  | `ti2i_baseimg` | Image-Guided Image Edit | Edit image with an explicit reference image |
|  | `tv2v` | Text-Instructed Video Edit | Instruction-based video editing |
|  | `tiv2v` | Image-Guided Video Edit | Edit video using an additional image reference |
| **Control / Transfer** | `tiv2v_clone` | Element Cloning | Clone motion, camera, or expression from reference |

### ✨ Highlights
- One unified interface for **generation, editing, and control**
- Supports **single-image, multi-image, and video** conditioning
- Instruction-driven, reference-driven, or hybrid control
- Easily extensible to new tasks

> In short: **if it’s a visual generation or editing task, VINO can handle it.**


---

### 1. Text-to-Image Generation (T2I)

Generate high-fidelity single-frame images.

```bash
torchrun --nproc_per_node=1 inference.py \
    --json_path ./assets/test_data/tasks/t2i.json \
    --output_height 640 --output_width 640 \
    --output_path output/t2i --seed 666
```

### 2. Text-to-Video Generation (T2V)

Generate coherent video clips. We utilize Sequence Parallelism to support high-resolution generation.

```bash
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/t2v.json \
    --output_height 480 --output_width 848 --output_num_frames 85 \
    --output_path output/t2v --seed 666

```

### 3. Image-to-Video Generation (I2V)

Animate static images. Supports both Caption descriptions and Instruction control.

```bash
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/i2v.json \
    --output_height 480 --output_width 848 --output_num_frames 85 \
    --output_path output/i2v --seed 666 \
     --negative_prompt_video ''
```

### 4. Image Editing (TI2I)

Perform image editing tasks using two distinct conditioning methods: pure text instruction or a combination of instruction and reference image.
Use `--guidance_scale_image` to control the strength of the visual reference.

```bash
# instruction
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/ti2i.json \
    --output_path output/ti2i --seed 666

# instruction + reference image
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/ti2i_baseimg.json \
    --guidance_scale_image 3.0 \
    --output_path output/ti2i_baseimg --seed 666
```

### 5. Multi-Image Reference Video Generation (TI2V)

You can input multiple reference images, and the model will understand the relationship between them to generate a video.

```bash
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/ti2v.json \
    --guidance_scale_image 3.0 \
    --output_height 480 --output_width 848 --output_num_frames 81 \
    --output_path output/ti2v --seed 666 \
    --negative_prompt_video ''
```

### 6. Video Editing (TV2V / TIV2V)

Supports video editing using only text instructions (`tv2v`) or combined with reference images (`tiv2v`).
The script automatically detects the aspect ratio of the input video.

```bash
# Instruction-based Editing
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/tv2v.json \
    --output_num_frames 81 \
    --output_path output/tv2v --seed 666 \
    --negative_prompt_video ''

# Image-based Editing
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/tiv2v.json \
    --output_num_frames 81 \
    --guidance_scale_image 3.0 \
    --output_path output/tiv2v --seed 666 
```

### 7. Element & Camera Clone

VINO allows for unique "Cloning" effects, such as keeping the scene static while only moving the camera (Bullet Time effect).

> 💡 **Tip**: To achieve the **Bullet Time** effect (freezing the scene while moving the camera), we recommend using a specific negative prompt to suppress object dynamics:
> `negative_prompt`: "moving objects, dynamic elements, animation, character movement, walking, running, talking, ... static camera, still image"

```bash
torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/tiv2v_clone.json \
    --output_num_frames 81 \
    --output_path output/tiv2v_clone --seed 666 \
    --negative_prompt_video ''

torchrun --nproc_per_node=8 inference.py \
    --json_path ./assets/test_data/tasks/tiv2v_camclone.json \
    --output_num_frames 81 \
    --output_path output/tiv2v_camclone_neg --seed 666 \
    --negative_prompt_video 'moving objects, dynamic elements, animation, character movement, walking, running, talking, blinking, living, breathing, flowing water, wind, distortion, morphing, mutating, shifting, jittery, flickering, frame interpolation artifacts, static camera, still image, frozen camera, low resolution, blurry, watermark, text, bad composition'

```

### 8. Understanding then Generation

Leveraging the VLM's powerful comprehension capabilities, you can first caption the input and then generate, creating an "Understanding-then-Generation" pipeline.

```bash
# Generate refined captions using VLM
python understand.py \
    --json_path ./assets/test_data/tasks/und_before.json \
    --output_path ./assets/test_data/tasks/und_after.json 
```

> 💡 Tip: You can customize the understanding task by modifying the prompt templates inside `understand.py.`


## ⚙️ Configuration Details

Our `inference.py` provides extensive parameter configurations. You can check the defaults in `VINOInferenceConfig`:

* **`--guidance_scale`**: Text guidance scale (CFG). 
* **`--guidance_scale_image`**: Image guidance scale for Image-Conditioned tasks. Recommended range: `1.0` - `3.0`.
* **`--timestep_shift`**: Timestep shift scheduler adjustment.
* **`--output_num_frames`**: Number of frames to generate.


## 📄 License

* **Codebase**: Apache License 2.0
* **Model Weights**: CC BY-NC 4.0 (Non-Commercial Use Only)

## 📝 Citation

If you find VINO useful for your research, please consider citing our paper:

```bibtex
@article{chen2026vino,
  title={VINO: A Unified Visual Generator with Interleaved OmniModal Context},
  author={Chen, Junyi and He, Tong and Fu, Zhoujie and Wan, Pengfei and Gai, Kun and Ye, Weicai},
  journal={arXiv preprint arXiv:2601.02358},
  year={2026}
}
```
