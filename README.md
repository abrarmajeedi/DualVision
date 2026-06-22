# DualVision: RGB–Infrared Multimodal Large Language Models for Robust Visual Reasoning

[![Paper](https://img.shields.io/badge/CVPR%20Findings%202026-Paper-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2026F/html/Majeedi_DUALVISION_RGB-Infrared_Multimodal_Large_Language_Models_for_Robust_Visual_Reasoning_CVPRF_2026_paper.html)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://abrarmajeedi.github.io/dualvision)

Official repository for our **CVPR Findings 2026** paper:
**"DualVision: RGB–Infrared Multimodal Large Language Models for Robust Visual Reasoning"**

[Abrar Majeedi](https://abrarmajeedi.github.io)<sup>1</sup>,
Zhiyuan Ruan<sup>2</sup>,
Ziyi Zhao<sup>2</sup>,
Hongcheng Wang<sup>2</sup>,
Jianglin Lu<sup>3</sup>,
Yin Li<sup>1</sup>

<sup>1</sup>University of Wisconsin–Madison &nbsp;&nbsp; <sup>2</sup>Amazon &nbsp;&nbsp; <sup>3</sup>Northeastern University

---

## Overview

Multimodal large language models (MLLMs) achieve impressive performance on visual reasoning with RGB imagery, yet remain fragile under common degradations such as fog, blur, and low-light conditions. Infrared (IR) imaging is a long-established complement to RGB and offers inherent robustness in these settings, but its integration into MLLMs is underexplored.

**DualVision** is a lightweight fusion module that efficiently incorporates IR–RGB information into MLLMs via patch-level **multi-scale localized cross-attention**. Each RGB patch token attends only to spatially corresponding IR regions, injecting complementary IR cues where they are most relevant, with low compute overhead and broad compatibility with existing MLLMs.

<p align="center">
  <img src="https://abrarmajeedi.github.io/dualvision/img/highlevel_cropped.png" width="80%">
</p>

## Key Contributions

- **DualVision fusion module**: a lightweight IR–RGB fusion design built on multi-scale 2D local cross-attention, drop-in compatible with existing MLLMs.
- **DV-204K**: a dataset of ~25K aligned IR–RGB image pairs with ~204K modality-aware QA annotations for instruction tuning.
- **DV-500**: a curated benchmark of 500 IR–RGB image pairs with 500 QA pairs for evaluating cross-modal reasoning under degradations.
- **~75% compute savings** vs. naïve concatenation: RGB-IR concatenation costs O(4N²); DualVision fuses to N tokens at O(N²).
- **Strong empirical results** under blur, low-light, and fog, outperforming open- and closed-source MLLMs (LLaVA 1.5, Qwen2-VL, LLaVA-Next Interleave, LLaMA-4 Scout, Claude Sonnet 3.5v2, Claude Opus 4) across clean and degraded settings.

## Method

<p align="center">
  <img src="https://abrarmajeedi.github.io/dualvision/img/main_fig.png" width="100%">
</p>

- **2D Local Cross-Attention**: each RGB token attends to IR tokens within a radius-r region centered at its location, enforcing spatially aligned fusion.
- **Multi-Scale Design**: multiple local cross-attention blocks applied sequentially with progressively larger radii, capturing multi-scale interactions while preserving locality.
- **Computational Efficiency**: fuses both modalities into N tokens (4× fewer than concatenation), reducing attention cost by ~75%.

## Repository structure

```
DualVision/
├── core/                 # training/evaluation code (built on LLaVA)
│   ├── llava/            # model, training, eval, and the DualVision fusion module
│   ├── scripts/          # train.sh, eval.sh, zero3.json
│   ├── requirements.txt
│   └── pyproject.toml
└── data/                 # DualVision QA annotations (images downloaded separately)
    └── v1/               # DV-204K: the dataset used in the paper
        ├── train.json    # DV-204K instruction-tuning conversations
        └── dv500.jsonl   # DV-500 benchmark
```

## Datasets

This release provides **DV-204K** and **DV-500**, the datasets used in the paper.

We build our annotations on top of publicly available aligned IR–RGB image pairs. The
underlying **images** must be downloaded from their official sources:

- [LLVIP](https://bupt-ai-cz.github.io/LLVIP/)
- [HDRT](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001824)

The **QA annotations** are released here under `data/v1/`:

| dataset | file | description |
|---------|------|-------------|
| **DV-204K** | `data/v1/train.json` | 49,976 instruction-tuning conversations (~204K modality-aware QA pairs) over ~25K aligned IR–RGB pairs. |
| **DV-500**  | `data/v1/dv500.jsonl` | 500 yes/no questions over 500 IR–RGB pairs for evaluating cross-modal reasoning under degradations. |

Every training sample feeds the model **both modalities** (RGB + IR) regardless of which
one the question targets.

### Arranging the images

Annotations reference images by relative path (`HDRT/...`, `LLVIP/...`). Download the two
image sets and arrange them under a single root:

```
<DATA_ROOT>/
├── HDRT/{visible,infrared}/      # visible = RGB, infrared = IR
└── LLVIP/{visible,infrared}/
```

Point the scripts at this directory via the `DATA_ROOT` environment variable.

## Setup

```bash
conda create -n dualvision python=3.10 -y
conda activate dualvision

cd core
# Install CUDA-matched torch first (example: CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                       # installs the `llava` package
# Optional but recommended (after torch):
pip install flash-attn==2.5.8 --no-build-isolation
```

## Fusion strategies

The fusion module combines the per-modality CLIP features before the LLM:

| strategy      | description |
|---------------|-------------|
| `local_xattn` | **DualVision**: RGB queries attend to a local IR window (3 blocks, radii 1/2/3). |
| `concat`      | concatenate RGB + IR tokens (baseline). |

## Training

```bash
cd core
# Usage: bash scripts/train.sh <strategy>
#   strategy: local_xattn or concat   (default: local_xattn)
DATA_ROOT=/path/to/images bash scripts/train.sh local_xattn
```

- LoRA finetune of `liuhaotian/llava-v1.5-7b`, 8× GPU via DeepSpeed ZeRO-3, bf16, 2 epochs.
- **Robustness augmentation:** at load time each RGB image has a 25% (`--degradation_prob`)
  chance of a random degradation (blur / fog / low-light); the IR image is never degraded.
- Checkpoints, logs, and eval predictions are written under
  `runs/<strategy>_<version>/<timestamp>/`.
- `train.sh` automatically runs the evaluation below when training finishes.

**Image mode & resolution.** Runs in image mode (`--use_features False`): raw RGB and IR
jpgs are pushed through the frozen CLIP tower (`openai/clip-vit-large-patch14-336`) at run
time, **336×336**, padded to square (no cropping), yielding 577 tokens per modality.

## Evaluation

`scripts/eval.sh` evaluates a trained checkpoint on **DV-500** across 13 conditions: the
clean original plus 12 RGB degradations (IR always clean):

```
blur{5,10,15,20}   bright{10,20,30,50}   fog{070,085,092,097}
```

```bash
cd core
DATA_ROOT=/path/to/images bash scripts/eval.sh <ckpt_path>
```

Each condition writes a predictions `*.jsonl` next to the answer file. Accuracy is exact
yes/no match (first yes/no token in the generation). Conditions run `NUM_GPUS` at a time.

> The checkpoint directory name must contain `llava` and `lora` (a requirement of the
> underlying LLaVA loader). `train.sh` handles this automatically.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{majeedi2026dualvision,
  title={DualVision: RGB-Infrared Multimodal Large Language Models for Robust Visual Reasoning},
  author={Abrar Majeedi and Zhiyuan Ruan and Ziyi Zhao and Hongcheng Wang and Jianglin Lu and Yin Li},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

## Acknowledgement

Our code is built on the excellent [LLaVA](https://github.com/haotian-liu/LLaVA) repository.
The DualVision additions are the fusion module
(`core/llava/model/multimodal_encoder/fusion_encoder.py`, `local_xattn.py`), the
dual-modality data path in `core/llava/train/train.py`, and
`core/llava/model/llava_arch.py::encode_dual_pixels`.

## Contact

For questions, please email majeedi@wisc.edu
