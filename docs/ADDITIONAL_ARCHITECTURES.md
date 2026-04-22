# Additional Architectures for Future Pipeline Modes

This document covers neural network architectures that are relevant to the goals of this project but are **not currently implemented** in either `mb/models/frameworks/keras/architectures.py` or `mb/models/frameworks/pytorch/architectures.py`, and are not addressed in `JEPA_LEARNING_STRATEGIES.md`.

The document is organised by use case, matching the future prediction modes mentioned in the JEPA strategies document: classification, object detection, segmentation, generation, and multi-modal.

---

## Currently implemented (for reference)

ResNet (18/34/50/101/152), EfficientNet (B0–B3), MobileNet (V2/V3), DenseNet (121/169/201), VGG (16/19).

All are CNN-based ImageNet-pretrained classifiers. The ViT family is mentioned in the JEPA document as a prerequisite for I-JEPA but is not yet implemented.

---

## 1. Image Classification — missing architecture families

### 1.1 Vision Transformer (ViT) family

ViT (Dosovitskiy et al., 2020, Google Brain) was the first demonstration that a pure transformer — no convolutions — could match or exceed CNNs on image classification when trained at scale. Images are split into fixed-size patches (typically 16×16 pixels); each patch is linearly embedded into a token; a standard transformer encoder processes the token sequence.

Key variants:
- **ViT-B/16, ViT-L/16, ViT-H/14** — Base, Large, Huge, denominator is patch size in pixels
- **DeiT** (Data-efficient Image Transformers, Touvron et al., 2021, Meta AI) — ViT trained with knowledge distillation from a CNN teacher; achieves strong performance on ImageNet-scale data without the original ViT's need for JFT-scale pretraining

**Why it matters for this project**: ViT is a prerequisite for I-JEPA (see `JEPA_LEARNING_STRATEGIES.md`). It also produces spatially-indexed patch tokens that are directly useful for detection and segmentation without modification. `timm` provides all standard ViT and DeiT variants with pretrained weights.

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=N)
model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=N)
```

### 1.2 Swin Transformer

Swin Transformer (Liu et al., 2021, Microsoft Research) solves ViT's two main practical limitations: fixed-resolution inputs and quadratic attention complexity. It uses a hierarchical structure (like ResNet's feature pyramid) and computes attention within local shifted windows rather than globally. This makes it suitable for dense prediction tasks (detection, segmentation) as well as classification.

Variants: Swin-T (Tiny), Swin-S (Small), Swin-B (Base), Swin-L (Large); also Swin-V2 with improved stability at high resolution.

**Why it matters**: Swin is the most widely used backbone for object detection and segmentation models (e.g., Mask R-CNN with Swin backbone). If the pipeline adds detection or segmentation modes, Swin is a natural shared backbone. Available in `timm` and `transformers`.

```python
import timm
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=N)
```

### 1.3 ConvNeXt

ConvNeXt (Liu et al., 2022, Meta AI) is a "modernised ResNet" — a pure CNN redesigned to incorporate the training recipes and design choices that make ViT strong (larger kernels, GELU activations, layer norm, inverted bottleneck). ConvNeXt-Base matches or outperforms Swin-B at similar compute.

ConvNeXt V2 (Woo et al., 2023) adds a masked autoencoder pre-training stage (FCMAE) that further improves downstream performance.

**Why it matters**: ConvNeXt is a drop-in improvement on ResNet for classification that also serves as a strong backbone for detection and segmentation. It does not require patch embeddings or positional encodings, so it integrates more easily than ViT into the existing CNN-oriented pipeline. Available in `timm`.

```python
import timm
model = timm.create_model('convnext_base', pretrained=True, num_classes=N)
model = timm.create_model('convnextv2_base', pretrained=True, num_classes=N)
```

### 1.4 EfficientNetV2

EfficientNetV2 (Tan & Le, 2021, Google Brain) improves on EfficientNet with progressive training (start with small images and mild augmentation, scale up during training), smaller but faster building blocks (Fused-MBConv), and revised scaling rules. EfficientNetV2-S trains 4× faster than EfficientNet-B7 with better accuracy.

Variants: V2-S, V2-M, V2-L, V2-XL. Available in `timm` and `keras.applications` (from TF 2.8+).

---

## 2. Object Detection

Detection adds a **spatial localisation** requirement: not just "what class is in this image" but "where is it, with a bounding box." All architectures below expect a backbone (typically one of the classification architectures above) plus a detection head.

### 2.1 DETR — Detection Transformer

DETR (Carion et al., 2020, Meta AI) was the first end-to-end transformer-based detector. It removes anchor boxes and NMS entirely: a CNN backbone extracts features, transformer encoder-decoder processes them, and a fixed set of "object queries" predict boxes and class labels directly.

**Why it matters**: The cleanest conceptual fit with a transformer-based pipeline (ViT backbone → DETR head). No anchor tuning, no NMS threshold tuning. Variants: Deformable DETR (faster convergence), DINO-DETR (state-of-the-art on COCO at time of writing).

Available via HuggingFace `transformers`:
```python
from transformers import DetrForObjectDetection
```

### 2.2 YOLO family

YOLO (You Only Look Once) models are single-stage detectors optimised for real-time inference. The current generation is:
- **YOLOv8** (Ultralytics, 2023) — clean Python API, supports detection, segmentation, pose estimation, and classification from the same interface
- **YOLO11** (Ultralytics, 2024) — successor to YOLOv8 with improved efficiency

**Why it matters**: If the pipeline ever supports "deploy a detector to run on a webcam or edge device," YOLO is the standard choice. The `ultralytics` Python package wraps everything:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # nano, small, medium, large, xlarge
```

### 2.3 Faster R-CNN / RetinaNet (torchvision)

Both are available in `torchvision.models.detection` with pretrained weights and a documented fine-tuning interface:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn_v2
```

Faster R-CNN is a two-stage detector (region proposal + classification); RetinaNet is a one-stage detector with a Focal Loss to handle class imbalance. Both use FPN (Feature Pyramid Network) backbones. These are the most commonly used reference detectors for custom datasets.

---

## 3. Image Segmentation

Segmentation assigns a class label to every pixel (semantic segmentation) or identifies individual object masks (instance segmentation).

### 3.1 U-Net

U-Net (Ronneberger et al., 2015) is an encoder-decoder architecture with skip connections between the encoder and decoder at each spatial scale. Originally designed for biomedical image segmentation, it remains the dominant architecture in medical imaging and any domain with limited data and high-resolution outputs.

**Why it matters**: If the pipeline adds segmentation support for scientific imaging, medical imaging, or satellite imagery, U-Net is almost always the starting point. Multiple implementations available:
- `segmentation-models-pytorch` (`pip install segmentation-models-pytorch`) — wraps U-Net and other decoders with any `timm` backbone
- `monai` — medical imaging-focused, has 3D U-Net variants

```python
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', classes=N)
```

### 3.2 SegFormer

SegFormer (Xie et al., 2021, NVIDIA) combines a hierarchical transformer encoder (similar to Swin but simpler) with a lightweight MLP decoder. Avoids positional encodings, making it naturally adaptable to arbitrary resolutions. Available in HuggingFace `transformers`:

```python
from transformers import SegformerForSemanticSegmentation
```

### 3.3 Mask R-CNN

Mask R-CNN (He et al., 2017, Meta AI) extends Faster R-CNN with a third head that predicts a binary mask for each detected object. This gives instance segmentation (individual object masks, not just bounding boxes). Available in `torchvision`:

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
```

---

## 4. Image Generation

### 4.1 Variational Autoencoder (VAE)

A VAE (Kingma & Welling, 2013) encodes images into a continuous latent distribution and learns to decode samples from that distribution back to images. The encoder and decoder are jointly trained with a reconstruction loss plus a KL divergence term that regularises the latent space. VAEs produce smooth, interpolatable latent spaces, which makes them useful for controlled generation and as a component inside larger generative systems (e.g., the VAE inside Stable Diffusion is a latent-space compressor).

No canonical pip package for VAEs — they are typically implemented directly. The `diffusers` library (HuggingFace) contains a production-quality VAE used in Stable Diffusion.

### 4.2 Diffusion Models — U-Net / DiT backbone

Diffusion models (Ho et al., 2020 DDPM; Song et al., 2020 DDIM) learn to reverse a Gaussian noise process iteratively. Two backbone architectures are used:

- **U-Net with cross-attention** — the backbone used in Stable Diffusion and most practical diffusion models. The time step and optional conditioning (text, class label) are injected via cross-attention layers at each scale.
- **DiT (Diffusion Transformer, Peebles & Xie, 2022)** — replaces the U-Net with a pure transformer. Scales better with model size. Used in Sora and the most recent high-resolution generation models.

Available via `diffusers` (`pip install diffusers`), which provides pretrained models, noise schedulers, and pipelines.

**Relevance to JEPA**: The latent space learned by I-JEPA is directly compatible with latent diffusion — the I-JEPA encoder could serve as the perceptual encoder for a latent diffusion model, replacing the VAE encoder.

### 4.3 GAN — Generative Adversarial Networks

GANs (Goodfellow et al., 2014) train a generator and discriminator in competition. While largely superseded by diffusion models for image quality, they remain relevant for:
- Real-time generation (single forward pass, no iterative denoising)
- Video generation (e.g., StyleGAN-V)
- Domain adaptation and data augmentation

**StyleGAN2/3** (NVIDIA) is the practical standard. Available at `NVlabs/stylegan3` (GitHub). The `torchgan` pip package provides building blocks for custom GAN architectures.

### 4.4 Flux and the MM-DiT Double-Stream Architecture

Flux (Black Forest Labs, 2024) is the current state of the art in open-weight text-to-image generation. Its architectural foundation is **MM-DiT** (Multimodal Diffusion Transformer), introduced in Stable Diffusion 3 (Esser et al., 2024), which extends DiT with a **double-stream block** design.

#### The double-stream block

Standard DiT concatenates image latent tokens and text conditioning tokens into a single sequence and processes them through the same transformer weights. This forces both modalities through an identical MLP, constraining what each can express internally.

A double-stream block maintains **two parallel transformer streams** — one for image tokens, one for text tokens — with independent weight matrices:

1. Each stream computes its own Q, K, V projections using stream-specific weights.
2. For the **attention step**, K and V from both streams are concatenated, giving every image token full visibility of every text token and vice versa (bidirectional cross-modal attention).
3. After attention, the streams are **split again** and each passes through its own feedforward MLP and normalisation layers.

The result: the two modalities interact semantically (via shared attention) while developing independent internal representations (via separate MLPs). Image patches and language tokens have fundamentally different statistical structure — separate MLPs let each specialise, while joint attention keys them together semantically.

#### Flux architecture specifics

Flux.1 uses a hybrid of double-stream and single-stream blocks:

```
[Patch embed + text encode]
        ↓
[19 × double-stream MM-DiT blocks]   ← image and text in parallel, joint attention
        ↓
[38 × single-stream DiT blocks]      ← streams merged, standard transformer
        ↓
[Decode latents → image]
```

Additional details:
- **Training objective**: flow matching (predicts the velocity field of a probability flow ODE) rather than DDPM noise prediction — smoother gradients, fewer inference steps needed
- **Text encoding**: dual encoder — CLIP-L (short-prompt semantic grounding) + T5-XXL (5B parameters, rich linguistic structure); the two encodings are concatenated as the text stream
- **VAE**: 16-channel latent space (vs 4-channel in SD 1.x), preserving higher spatial fidelity
- **Position encoding**: RoPE (Rotary Position Embeddings) applied to image patch tokens for spatial awareness independent of resolution

#### Flux variants

| Variant | Parameters | Notes | Licence |
|---|---|---|---|
| Flux.1-dev | 12B | Full model, best quality | Non-commercial |
| Flux.1-schnell | 12B | Distilled to 4 denoising steps | Apache 2.0 |
| Flux.1-pro | 12B | Black Forest Labs API only | Commercial |
| **Chroma** | ~8B | Community variant; removes CLIP encoder entirely, conditions solely on T5-XXL, modifies guidance mechanism; stronger instruction-following at the cost of requiring T5-XXL | Community |

Chroma's removal of the CLIP encoder is architecturally significant: the double-stream block's text stream receives only T5 embeddings, which carry richer syntactic and semantic structure than CLIP's contrastively-trained embeddings. Other community variants fine-tune only the double-stream blocks (which handle image-text interaction) on specific aesthetic domains while keeping the single-stream blocks frozen — a natural split that maps well onto LoRA training (see Section 6).

#### Available via `diffusers`

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")
image = pipe("a photo of a cat", num_inference_steps=28, guidance_scale=3.5).images[0]
```

#### Relevance to this pipeline

If image generation is added as a prediction mode, Flux represents the current practical ceiling for open-weight text-to-image generation. The MM-DiT double-stream block is also architecturally adjacent to the multi-modal directions in `JEPA_LEARNING_STRATEGIES.md` — joint attention over image and language tokens in a generative setting is a structural parallel to the joint-embedding objective in JEPA.

---

## 5. Video

Video extends image architectures along the temporal dimension. The two primary tasks — **video classification** (assigning a label to a clip) and **video generation** (producing a clip from a prompt or conditioning signal) — make different architectural demands, but share a common foundation: how to efficiently model correlations across both space and time.

---

### 5.1 Video Classification

Video classification assigns a category label to a short clip. The core challenge is that naive frame-by-frame processing discards temporal relationships (motion, causality, rhythm), while processing all frames jointly is quadratically expensive.

#### SlowFast Networks

SlowFast (Feichtenhofer et al., 2019, Meta AI) processes video through two parallel pathways:

- **Slow pathway**: high spatial resolution, low frame rate (e.g., 8 frames) — captures fine-grained appearance
- **Fast pathway**: low spatial resolution, high frame rate (e.g., 32 frames) — captures rapid motion

Lateral connections fuse information between the pathways at each stage. The result is a model that specialises its capacity: the slow pathway learns what objects look like, the fast pathway learns how they move. SlowFast is the standard baseline for action recognition on Kinetics-400/600.

Available in `pytorchvideo` (`pip install pytorchvideo`, Meta):
```python
from pytorchvideo.models import create_slowfast
```

#### Video Swin Transformer

Video Swin (Liu et al., 2022, Microsoft Research) extends the Swin Transformer's shifted-window attention into the temporal dimension. Local 3D windows of patches (spatial height × spatial width × temporal depth) are processed with self-attention; windows shift across both space and time between layers to allow cross-window communication. This gives full spatio-temporal modelling at sub-quadratic cost.

Variants: Swin-T-3D, Swin-S-3D, Swin-B-3D, Swin-L-3D. Available in `timm` and `transformers`.

#### TimeSformer and ViViT

Both apply standard ViT to video by tokenising frames into patches, with different strategies for handling the added temporal dimension:

- **TimeSformer** (Bertasius et al., 2021, Facebook): "divided space-time attention" — applies spatial self-attention and temporal self-attention in alternating separate blocks, rather than joint 3D attention. Computationally efficient and strong on Kinetics.
- **ViViT** (Arnab et al., 2021, Google): offers several attention factorisation variants, the most practical being the "factorised encoder" which separately encodes spatial and temporal information before fusing them.

Both are available in HuggingFace `transformers`.

#### VideoMAE — Pre-training Backbone for Video

VideoMAE (Tong et al., 2022) applies masked autoencoder pre-training to video, masking an aggressive ~90% of spatio-temporal patch tubes and training the model to reconstruct the missing regions. The high masking ratio is necessary because video is highly temporally redundant — neighbouring frames are similar enough that a low masking rate gives the model trivial reconstructions to learn from. VideoMAE pre-training produces representations that transfer well to downstream classification tasks. Available in HuggingFace `transformers`:

```python
from transformers import VideoMAEForVideoClassification
```

---

### 5.2 Video Generation — Core Architectural Extensions

Video generation inherits the diffusion model framework from image generation but must additionally model temporal coherence — frames must flow smoothly rather than being independently generated. Three architectural innovations underpin all modern video generators:

#### Temporal attention

The simplest extension: take an existing image DiT or U-Net and insert **temporal attention layers** between spatial layers. Each spatial feature token attends to its counterpart tokens at other timesteps in the clip. AnimateDiff uses this approach — temporal attention modules are trained independently and can be inserted into any existing Stable Diffusion model as adapters, inheriting all its existing LoRAs and fine-tunes.

```
[Frame 1 spatial features]──┐
[Frame 2 spatial features]──┤ → temporal self-attention → residual
[Frame N spatial features]──┘
```

#### 3D VAE

Image generation compresses each frame independently through a 2D VAE. A 3D VAE extends the encoder and decoder with temporal convolutions and attention, compressing a video clip into a latent tensor with both spatial and temporal dimensions. This enables the diffusion transformer to operate on a compact spatio-temporal latent rather than on raw frames or per-frame latents. CogVideoX, HunyuanVideo, and Wan all use a 3D VAE. Temporally-aware compression is what makes high-quality motion coherence possible at reasonable computational cost.

#### Video DiT — space-time patch tokens

Full 3D transformers treat the video latent as a sequence of **space-time patch tokens**: a volume of shape (time × height × width) is divided into 3D patches, linearly embedded, and processed by a standard transformer. Position embeddings (typically RoPE extended to 3D) encode each token's location in space and time. The denoising transformer then operates on this full sequence, learning correlations across all spatial positions at all timesteps jointly.

This is the architecture used by Wan, HunyuanVideo, Open-Sora, and most current high-quality video generators. It is a natural extension of the 2D video DiT architecture already described in Section 4.2.

---

### 5.3 Video Generation Models

The landscape as of 2024–2025 is notably dominated by models from Chinese research labs, reflecting both the volume of investment and the commercial motivation of companies (ByteDance, Alibaba, Tencent, Kuaishou) that operate large video platforms with the training data and infrastructure to match.

#### AnimateDiff (temporal U-Net, 2023)

AnimateDiff (Guo et al., 2023) is a **motion module** — a set of temporal attention layers that can be grafted into any Stable Diffusion 1.5 or XL model without retraining the base model. This makes it compatible with the entire ecosystem of SD LoRAs and fine-tunes. The motion module is trained on video clips to learn frame-to-frame coherence; at inference time the spatial layers remain as in the original SD model. Practical for short clips (16–24 frames) and stylised animation, less so for photorealistic long-form video.

#### Stable Video Diffusion (SVD, 2023)

Stability AI's SVD is an **image-to-video** model — it conditions on a single input frame and generates a short video by adding temporal attention to the SD U-Net backbone. Not a text-to-video model in the full sense; it generates plausible motion from a given starting frame. Available via `diffusers`.

#### Open-Sora / Open-Sora-Plan

Community open-source reproductions of the Sora architecture (OpenAI, 2024), which was described but not released. Both use a Video DiT approach — a full 3D transformer operating on space-time patch tokens, trained with a flow matching objective. Neither matches the quality of proprietary Chinese models at present but provide accessible, modifiable codebases. Available on GitHub.

#### CogVideoX (Zhipu AI / THUDM, 2024)

CogVideoX is a strong open-weight text-to-video model from Zhipu AI (Tsinghua University affiliation). Key architecture details:

- **3D causal VAE**: encodes temporal-spatial structure while preserving causal ordering (a frame can only attend to previous frames in the VAE), which aids autoregressive extension
- **Expert transformer**: a Video DiT with modality-specific expert routing, enabling specialised processing of different token types
- **Text encoder**: a bilingual T5 model (supporting both Chinese and English)
- Available via HuggingFace `diffusers`:

```python
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
```

#### HunyuanVideo (Tencent, 2024)

HunyuanVideo is a 13B parameter open-weight model and one of the strongest publicly available video generators as of early 2025. Architecture details:

- **Full-attention 3D DiT**: joint spatial and temporal attention across all tokens; no separated or windowed attention
- **Dual text encoder**: CLIP-L for visual-semantic grounding, plus a **bilingual large language model (LLM)** encoder derived from Tencent's HunYuan LLM family that processes both Chinese and English natively — the LLM encoder provides rich syntactic and semantic structure beyond what CLIP can represent
- **Flow matching** training objective (same as Flux)
- Available on HuggingFace, unofficial `diffusers` integration available

#### Wan 2.1 (Alibaba / 通义万象, 2025)

Wan 2.1 achieves state-of-the-art results on multiple video generation benchmarks and is available open-weight. Its most architecturally distinctive feature from a prompting standpoint is its text encoder:

- **umt5-xxl** (Universal Multilingual T5, 10B parameters): a multilingual T5 variant covering 100+ languages, trained on multilingual web data. This is a substantially larger and more linguistically capable text encoder than the CLIP+T5 combination used in image generation models.
- **3D causal VAE** and full Video DiT backbone
- Available via `diffusers`

#### Other notable open models

- **Mochi-1** (Genmo, 2024): an asymmetric DiT where the denoising network is larger than in comparable models; strong motion quality. Open-weight, Apache 2.0.
- **LTX-Video** (Lightricks, 2024): optimised for efficient inference; can run at lower VRAM than most Video DiTs. Available via `diffusers`.

---

### 5.4 Multilingual Text Encoders and Prompting

The dominance of Chinese video generation models has a concrete architectural consequence: the text encoders used in these models are not the English-centric CLIP and T5 that underpin Western image generators. Understanding this is necessary for both prompting these models effectively and for designing a pipeline that will handle them correctly.

#### Why Chinese models use different text encoders

Western text-to-image models (Stable Diffusion, Flux) use CLIP trained on English-centric image-caption pairs and T5/T5-XXL trained primarily on English web text. These encoders produce strong embeddings for English prompts but have limited vocabulary coverage for Chinese, Japanese, Korean, and other languages — character-level tokenisation in those languages is fragmented and poorly represented in training data.

Chinese video models are built by teams whose primary users write Chinese, whose internal annotation pipelines produce Chinese captions, and whose large-scale pre-training data includes Chinese video and text. This creates a natural motivation to use text encoders that are natively multilingual or Chinese-language-primary from the start.

#### Encoder choices in current Chinese video models

| Model | Text encoder | Languages | Notes |
|---|---|---|---|
| Wan 2.1 | **umt5-xxl** (10B) | 100+ including Chinese, English, Japanese | Largest and most multilingual; richer embeddings than T5-XXL |
| HunyuanVideo | CLIP-L + **bilingual HunYuan LLM** | Chinese + English | LLM encoder provides deep linguistic understanding |
| CogVideoX | Bilingual T5 | Chinese + English | Smaller but capable bilingual variant |
| AnimateDiff / SVD | CLIP-L (English-dominant) | English primarily | Western models; limited Chinese support |
| Flux.1 | CLIP-L + T5-XXL | English primarily | English-dominant; limited Chinese |

**umt5-xxl** is worth understanding specifically. It is a 10B parameter T5 model trained on a multilingual corpus spanning 100+ languages with explicit efforts to balance representation across languages. Compared to standard T5-XXL (also 10B, but English-dominant), it produces meaningfully better embeddings for non-English text, and Chinese descriptions of visual scenes translate directly into strong conditioning signals without requiring English as an intermediary.

The **bilingual LLM encoder** approach used by HunyuanVideo represents a step further: a full causal language model is used as the text encoder rather than an encoder-only T5. This means the text representation carries the kind of world knowledge, syntactic structure, and compositional reasoning that autoregressive LLMs are trained on, rather than just the contrastive or reconstruction-based representations of T5. This enables more precise following of complex, compositional, or culturally-specific prompts.

#### Practical prompting implications

For a user working with these models in English:
- **Wan 2.1 and CogVideoX** process English prompts well due to their multilingual encoders, but detailed or culturally-specific descriptions may produce better results when written in Chinese, particularly for scene compositions that were frequent in the Chinese-language training data.
- **HunyuanVideo's** LLM encoder handles English reliably — the bilingual LLM was explicitly trained on both languages — but complex artistic direction phrases (specific photographic styles, cultural references) may still benefit from Chinese equivalents.
- Translation preprocessing is a practical option: using a capable LLM to translate or re-express the user's prompt in Chinese before passing it to the video model can improve adherence for these models. This is distinct from naive machine translation — the goal is to produce a description that reads naturally in Chinese, not a literal translation.

#### Pipeline integration considerations

For a pipeline supporting video generation:

1. **Language detection**: accept Chinese and English prompts natively; detect the language and route accordingly rather than assuming English
2. **Translation as an optional pre-processing step**: offer an optional LLM-based prompt translation/rewriting step for models that benefit from Chinese input
3. **Tokenisation budget awareness**: Chinese text is more information-dense per character than English. A 50-token Chinese prompt can carry substantially more semantic content than a 50-token English prompt — the token budget for video models (typically 256–512 tokens) is used differently across languages
4. **Model-specific text encoder loading**: unlike image models where CLIP+T5 is nearly universal, video models require loading the correct text encoder (umt5-xxl for Wan, the HunYuan LLM for HunyuanVideo, etc.); the pipeline's model configuration must track which encoder each video model requires

#### Connection to V-JEPA

V-JEPA (see `JEPA_LEARNING_STRATEGIES.md`, Section 7) is the self-supervised pre-training counterpart of the video generation models described here. Where video generation models learn to synthesise new clips from text or image conditioning, V-JEPA learns video representations by predicting masked spatio-temporal regions in embedding space. The same 3D patch tokenisation and temporal attention mechanisms appear in both contexts — a backbone pre-trained with V-JEPA could serve as the video encoder for a generation model's conditioning signal or as a classification backbone, making it the natural SSL complement to the supervised video models above.

---

## 6. Multi-modal

### 5.1 CLIP

CLIP (Contrastive Language-Image Pre-Training, Radford et al., 2021, OpenAI) trains a dual-encoder (image + text) with a contrastive objective that aligns their embedding spaces. The result is a model that can classify images by similarity to text descriptions without any task-specific fine-tuning ("zero-shot classification").

**Why it matters**: CLIP image encoders are among the strongest general-purpose visual representations available. They can be used as backbone feature extractors for any downstream task. Available in HuggingFace `transformers` and via `openai/CLIP` (GitHub, also pip-installable):

```python
import clip
model, preprocess = clip.load('ViT-B/32', device='cuda')
# or:
from transformers import CLIPModel
```

### 5.2 SigLIP

SigLIP (Zhai et al., 2023, Google) replaces CLIP's softmax contrastive loss with a sigmoid loss, enabling better performance on smaller batches and with independent image-text pairs rather than in-batch negatives. Currently provides some of the best open visual encoders. Available in HuggingFace `transformers`.

---

## 7. Parameter-Efficient Fine-Tuning (PEFT) and LoRA

The architectures in this document are large — ViT-L has 307M parameters, Flux.1-dev has 12B. Full fine-tuning on all weights is expensive and requires large labelled datasets. **Parameter-efficient fine-tuning (PEFT)** techniques adapt a pretrained model to a specific concept, style, or domain by training only a small number of additional parameters while keeping the base model frozen.

LoRA is the dominant PEFT method in practice and is the primary mechanism by which users customise large generation models such as Flux and Stable Diffusion.

### 6.1 LoRA — Low-Rank Adaptation

LoRA (Hu et al., 2021, Microsoft) is based on the observation that the weight updates needed for fine-tuning tend to have low intrinsic rank. For a pretrained weight matrix **W** ∈ ℝ^(d×k), instead of learning a full ΔW, LoRA decomposes the update into two small matrices:

```
ΔW = B · A     where B ∈ ℝ^(d×r),  A ∈ ℝ^(r×k),  rank r << min(d, k)
```

During training, **W** is frozen. Only **A** and **B** are updated. At inference the adapted weight is:

```
W' = W + α · (B · A)
```

where α is a scaling factor (commonly written as `lora_alpha / r`). With r=4 on a 1024×1024 weight matrix, trainable parameters drop from ~1M to ~8K — roughly a 125× reduction.

For image generation models a LoRA is typically trained on 10–100 images representing a concept (a specific subject, an art style, a product). The trained **BA** matrices are saved as a small checkpoint file (~10–200 MB) that users apply on top of the base model. Multiple LoRAs can be combined additively:

```
W' = W + α₁·B₁A₁ + α₂·B₂A₂ + ...
```

The per-LoRA α weight controls how strongly each adapter steers the output — a continuous knob between the base model and the target concept.

### 6.2 LoRA variants

| Variant | Decomposition | When to prefer |
|---|---|---|
| **LoRA** (standard) | B·A low-rank matrices | General purpose; well-understood |
| **LoHa** | Hadamard product of two low-rank pairs (W1 ⊙ W2) | More expressive per parameter; good for style transfer |
| **LoKr** | Kronecker product decomposition | Captures structured weight patterns well |
| **DoRA** (Weight-Decomposed LoRA) | Decomposes W into magnitude × direction; updates direction via LoRA, magnitude separately | More stable training; better generalisation |
| **LyCORIS** | Umbrella project implementing LoHa, LoKr, and others | Use when standard LoRA lacks expressiveness |

### 6.3 LoRA in the context of this pipeline

LoRA is not specific to image generation — it applies wherever a large pretrained model needs to be adapted efficiently:

| Use case | How LoRA applies |
|---|---|
| **Image generation** (Flux, SD3) | Train on concept images to steer output; primary user customisation mechanism for generation models |
| **Classification** (ViT, CLIP, SigLIP) | Fine-tune large vision encoders on a new label taxonomy without full retraining |
| **SSL / JEPA encoders** | Adapt a pretrained I-JEPA context encoder to a new domain with a fraction of the compute |
| **Multi-modal** (CLIP, SigLIP) | Extend the text-image embedding space to a new concept vocabulary |

For **Flux specifically**: the double-stream blocks (which handle image-text interaction) and the single-stream blocks (which refine the merged representation) represent natural LoRA targets. Community practice trains LoRAs on the attention projection layers (`q_proj`, `k_proj`, `v_proj`) and sometimes the MLP layers within the double-stream blocks, often leaving single-stream blocks at a lower rank or untouched, since the double-stream blocks encode the cross-modal binding of the concept.

Adding LoRA support to this pipeline for any model type means:
1. Wrapping a loaded backbone with a PEFT config before training begins
2. Saving only the LoRA weights as the training checkpoint (not the full model)
3. At inference, loading the base model and merging the adapter

### 6.4 Available packages

**`peft`** (HuggingFace, `pip install peft`) is the standard library for LoRA on any HuggingFace model:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, config)
# Only LoRA weights are trained; base_model weights are frozen
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 307,494,912 || trainable%: 0.096
```

**`diffusers`** includes LoRA training scripts for Flux and Stable Diffusion under `examples/dreambooth/` in the repository. These handle the full training loop including dataset preparation, scheduler setup, and LoRA checkpoint saving.

**`kohya_ss`** (GitHub: `kohya-ss/sd-scripts`) is the community standard for training generation LoRAs, with both a script interface and a GUI (`kohya_ss`). Supports standard LoRA, LoHa, LoKr, and LyCORIS variants. Not on PyPI, but the most widely used tool for Flux/SD LoRA training in practice.

---

## 8. Recurrent-Depth Transformers — Mythos / OpenMythos

### Background and provenance

**Mythos** was an unreleased Anthropic model described in public technical communication but never open-sourced. Its central innovation is the **Recurrent-Depth Transformer (RDT)**, sometimes called a *Looped Transformer*: instead of stacking hundreds of unique layers, a single transformer block is recycled and executed multiple times within one forward pass, sharing its weights across all iterations. **OpenMythos** (`kyegomez/OpenMythos`, GitHub) is an independent community reproduction of this design based on the public description, and is the only available open implementation.

This section documents the RDT architecture as realised in OpenMythos, its component design choices, and where it is relevant to the goals of this project.

---

### 8.1 Core architecture — the Recurrent-Depth Transformer

A standard transformer of depth D contains D distinct sets of weights. An RDT collapses this into three phases:

```
[Prelude]              ← P standard transformer layers, executed once
      ↓
[Recurrent Block]      ← 1 transformer block, weights shared, executed up to K times
      ↓
[Coda]                 ← C standard transformer layers, executed once
```

The Prelude encodes the input into a hidden state `h_0`. At each loop iteration `t`, the recurrent block updates the hidden state:

```
h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
```

Where:
- `h_t` is the current hidden state (the "thinking" state being refined across iterations)
- `e` is the encoded input from the Prelude, **re-injected at every loop** to prevent the hidden state from drifting away from the input signal
- `A` and `B` are learned scalar matrices that gate how much the previous state and the input each contribute

After K iterations, the Coda processes the final hidden state into output logits.

#### Why this matters

A model trained on K loops can be run at inference time with K′ > K loops on harder examples — at the cost of proportionally more compute but with no additional parameters. This is **inference-time compute scaling**, analogous in effect to chain-of-thought reasoning: the model can "think longer" about a difficult input by iterating more. Benchmarks on compositional reasoning tasks (multi-hop chains) show that an RDT trained on 5-hop chains generalises to 10-hop chains by increasing loop count at inference, whereas a standard transformer fails on out-of-distribution hop counts.

---

### 8.2 Spectral stability

Training looped models risks two failure modes: the hidden state either explodes or collapses to zero across iterations. OpenMythos addresses this structurally by parameterising `A` as a **continuous negative diagonal matrix** and approximating the discrete update with a Zero-Order Hold or Euler scheme:

```
A = -exp(a_raw)   (elementwise, guaranteeing A < 0)
discrete A = exp(A·Δt)   (always in (0, 1))
```

This guarantees that the **spectral radius ρ(A) < 1 by construction**, regardless of learning rate — the hidden state is always a contractive map of itself, ensuring stability without the gradient clipping or careful initialisation required by vanilla RNNs. The constraint does not need to be imposed as a penalty term; it is enforced in the parameterisation.

---

### 8.3 Attention variants — MLA and GQA

The recurrent block's self-attention layer supports two modes, switchable via config:

**MLA (Multi-head Latent Attention)** — introduced in DeepSeek-V2/V3. Rather than storing full-rank K and V caches for each head, MLA compresses the key-value state into a low-rank latent vector and reconstructs K/V on the fly:

```
[c_KV] = W_DKV · h       (down-project to low-rank latent)
K = W_UK · c_KV          (up-project to key heads)
V = W_UV · c_KV          (up-project to value heads)
Q = W_DQ · h → W_UQ      (separate query compression)
```

Key-value cache at inference stores only `c_KV` (the compressed latent) rather than full K and V matrices, reducing KV cache memory roughly in proportion to `kv_lora_rank / (n_heads × head_dim)`. MLA also separates positional information: a RoPE-encoded component of the query/key is concatenated to the compressed non-positional component, so position encoding does not contaminate the cached latent.

**GQA (Grouped Query Attention)** — simpler alternative: `n_kv_heads` key/value heads shared across groups of query heads. Faster to implement, lower memory than MHA, but does not achieve MLA's compression ratio. Appropriate for smaller variants where KV cache is not the bottleneck.

---

### 8.4 Sparse Mixture of Experts feedforward

The feedforward sublayer within the recurrent block is a **sparse MoE**:

- `n_experts` expert MLPs are defined in total
- `n_shared_experts` are **always activated** for every token (domain-general capacity)
- From the remaining experts, `n_experts_per_tok` are selected via a learned router (top-K gating) per token
- `expert_dim` controls each expert's internal hidden dimension

This allows the model to specialise different experts for different types of content (e.g. code, reasoning, factual recall) while only paying the cost of `n_shared_experts + n_experts_per_tok` experts per forward pass — keeping FLOPs proportional to the activated subset rather than the full expert count. The architecture is directly analogous to the MoE design in Mixtral, DeepSeek-V2, and other efficiency-focused LLMs.

---

### 8.5 Available variants

OpenMythos ships five pre-configured size presets:

| Variant | Parameters | Notes |
|---|---|---|
| `mythos_1b` | ~1B | Entry-level; fits in consumer VRAM |
| `mythos_3b` | ~3B | Practical fine-tuning target |
| `mythos_7b` | ~7B | Competitive with Llama-class models |
| `mythos_70b` | ~70B | Large-scale; multi-GPU required |
| `mythos_1t` | ~1T | Research-scale; not practical without clusters |

All variants share the same RDT structure; they differ in `dim`, `n_heads`, `n_experts`, `max_loop_iters`, and layer counts.

#### Minimal configuration

```python
from openmythos import Mythos, MythosConfig

config = MythosConfig(
    vocab_size=50_257,
    dim=2048,
    n_heads=16,
    max_seq_len=8192,
    max_loop_iters=8,        # train with 4–8, can increase at inference
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=2,
    n_experts_per_tok=2,
    expert_dim=512,
    lora_rank=16,
    attn_type="mla",         # or "gqa"
)
model = Mythos(config)

# Standard forward pass (fixed loop count)
logits = model(input_ids, n_loops=4)

# Inference-time compute scaling — same weights, more thinking
logits_hard = model(input_ids, n_loops=12)
```

---

### 8.6 Relevance to this project

The Mythos / RDT architecture is primarily a **language model** design. Its direct integration into the current vision-focused pipeline is limited, but there are several meaningful points of contact:

#### As a text encoder for multimodal generation

The video generation section (Section 5) describes the move from CLIP+T5 text encoders toward larger, richer language model encoders (HunyuanVideo's bilingual LLM encoder, Wan 2.1's umt5-xxl). An RDT-based text encoder with MoE feedforward and configurable loop depth represents the logical next step in this direction: a compact model (3B–7B activated parameters) capable of deeper compositional reasoning than T5-class encoders, with the ability to scale text encoding compute independently of model size at inference time.

#### Inference-time compute scaling as a general principle

The RDT loop mechanism is architecturally agnostic — the same principle (recycle a block N times, re-inject input at every step) can be applied to:
- **Vision Transformers** (ViT-RDT): patch tokens as the hidden state, the full image embedding re-injected as `e`, the loop count scaling spatial reasoning depth for dense tasks
- **Video encoders**: temporal hidden states iterated across the loop, enabling the model to "replay" a clip conceptually before predicting
- Any sequence model where the depth of reasoning should be proportional to the difficulty of the instance, not fixed by the architecture

#### Component-level relevance

Even if the full RDT is not adopted, two components are independently relevant:
- **MLA** is the current state-of-the-art KV cache compression technique and directly applicable to any attention-heavy model (large ViTs, video DiTs, multimodal encoders) where KV memory is a bottleneck
- **Sparse MoE** is applicable to any part of the pipeline where a single dense feedforward layer is a capacity bottleneck — including Flux's single-stream blocks or a large multimodal encoder

#### Practical caveats

OpenMythos is a community implementation of an unreleased model. Its weight checkpoints have not been trained to production quality. Using it in this project means training from scratch, which requires significant compute and data (the reference training run targets 30B tokens on FineWeb-Edu). It is therefore more relevant as an **architectural reference and research direction** than as a plug-and-play pretrained backbone. The most practical near-term use case is experimenting with small variants (mythos_1b, mythos_3b) as text encoders or sequence reasoning modules alongside pretrained vision backbones.

---

## 9. Packages summary

| Package | Primary use | Install |
|---|---|---|
| `timm` | ViT, Swin, ConvNeXt, EfficientNetV2, 700+ classification backbones with pretrained weights | `pip install timm` |
| `transformers` | ViT, Swin, DETR, SegFormer, CLIP, SigLIP, with fine-tuning interfaces | `pip install transformers` |
| `torchvision.models.detection` | Faster R-CNN, RetinaNet, Mask R-CNN | included in `torchvision` |
| `ultralytics` | YOLOv8, YOLO11, detection/segmentation/pose | `pip install ultralytics` |
| `segmentation-models-pytorch` | U-Net, FPN, PSPNet with any timm backbone | `pip install segmentation-models-pytorch` |
| `diffusers` | Stable Diffusion, Flux, DDPM/DDIM, DiT, VAE, LoRA training scripts | `pip install diffusers` |
| `peft` | LoRA, LoHa, DoRA, and other PEFT adapters for any HuggingFace model | `pip install peft` |
| `lightly` | SimCLR, VICReg, Barlow Twins, I-JEPA, V-JEPA SSL | `pip install lightly` |
| `clip` (OpenAI) | CLIP image+text encoders | `pip install clip` (via GitHub) |
| `monai` | 3D U-Net, medical imaging segmentation | `pip install monai` |
| `kohya_ss` | GUI + scripts for LoRA/LoHa/LoKr/LyCORIS training on Flux and SD | GitHub only (`kohya-ss/sd-scripts`) |
| `pytorchvideo` | SlowFast, MViT, video data loading utilities (Meta AI) | `pip install pytorchvideo` |
| `openmythos` | Recurrent-Depth Transformer (Mythos architecture): looped transformer, MLA, MoE | GitHub only (`kyegomez/OpenMythos`) |

---

## References

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words* (ViT). [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron et al. (2021). *Training data-efficient image transformers* (DeiT). [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Liu et al. (2021). *Swin Transformer*. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Liu et al. (2022). *A ConvNet for the 2020s* (ConvNeXt). [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Tan & Le (2021). *EfficientNetV2*. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
- Carion et al. (2020). *End-to-End Object Detection with Transformers* (DETR). [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- He et al. (2017). *Mask R-CNN*. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Ronneberger et al. (2015). *U-Net*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Xie et al. (2021). *SegFormer*. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Peebles & Xie (2022). *Scalable Diffusion Models with Transformers* (DiT). [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (Stable Diffusion 3 / MM-DiT). [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
- Black Forest Labs (2024). *FLUX.1*. [github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- Feichtenhofer et al. (2019). *SlowFast Networks for Video Recognition*. [arXiv:1812.03982](https://arxiv.org/abs/1812.03982)
- Liu et al. (2022). *Video Swin Transformer*. [arXiv:2106.13230](https://arxiv.org/abs/2106.13230)
- Bertasius et al. (2021). *Is Space-Time Attention All You Need for Video Understanding?* (TimeSformer). [arXiv:2102.05095](https://arxiv.org/abs/2102.05095)
- Arnab et al. (2021). *ViViT: A Video Vision Transformer*. [arXiv:2103.15691](https://arxiv.org/abs/2103.15691)
- Tong et al. (2022). *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*. [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
- Guo et al. (2023). *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning*. [arXiv:2307.04725](https://arxiv.org/abs/2307.04725)
- Yang et al. (2024). *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer*. [arXiv:2408.06072](https://arxiv.org/abs/2408.06072)
- Kong et al. (2024). *HunyuanVideo: A Systematic Framework For Large Video Generation Model*. [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)
- Wan Team, Alibaba (2025). *Wan: Open and Advanced Large-Scale Video Generative Models*. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Khosla et al. (2020). *Supervised Contrastive Learning* (SupCon). [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
- Assran et al. (2021). *Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments* (PAWS). [arXiv:2104.13963](https://arxiv.org/abs/2104.13963)
- Liu et al. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (MLA, MoE). [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Giannou et al. (2023). *Looped Transformers as Programmable Computers*. [arXiv:2301.13196](https://arxiv.org/abs/2301.13196)
- Dehghani et al. (2018). *Universal Transformers* (foundational looped/recurrent-depth transformer). [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
- Gomez, K. (2025). *OpenMythos: Open-source Recurrent-Depth Transformer*. [github.com/kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos)
