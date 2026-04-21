# Self-Supervised & Joint Embedding Learning Strategies

This document explores how modern self-supervised learning (SSL) strategies — specifically those covered in the JEPA family of ideas — could be integrated into this training pipeline. The pipeline currently performs supervised transfer learning (frozen → unfrozen fine-tuning on pretrained backbones). The strategies below represent complementary or alternative approaches that can produce richer, more generalizable representations.

---

## Background: Why These Strategies Matter Here

The current pipeline trains image classifiers by fine-tuning ImageNet-pretrained backbones on labelled data. That works well, but it has two structural limitations:

1. **Label dependency**: Every image must be labelled. SSL strategies learn useful representations from unlabelled data, which is almost always cheaper to collect.
2. **Representation quality ceiling**: ImageNet pretraining encodes ImageNet priors. If the target domain differs (e.g. medical imaging, satellite imagery, product photos), SSL pre-training on domain-specific unlabelled data can produce a better starting backbone than ImageNet weights.

As the pipeline expands to image generation, object detection, or other prediction modes, these strategies become even more relevant — they are not classification-specific.

---

## 1. Sample-Contrastive Learning

### Concept

Sample-contrastive methods learn by pulling together representations of different augmented **views of the same image** while pushing apart representations of **views from different images**. The unit of contrast is the image sample.

### SimCLR

SimCLR (Chen et al., 2020) is the canonical example. The training loop for one step is:

1. For each image `x` in a batch, sample two augmented views: `x_i` and `x_j` (e.g. random crop + colour jitter + grayscale + blur).
2. Pass both views through the shared encoder `f(·)` to get representations `h_i`, `h_j`.
3. Pass representations through a small projection MLP `g(·)` to get `z_i`, `z_j`.
4. Apply the NT-Xent (normalised temperature-scaled cross-entropy) loss: maximise cosine similarity between `(z_i, z_j)` while minimising it against all other `2(N-1)` views in the batch.
5. After pre-training, **discard the projection head** and fine-tune the encoder `f(·)` on labelled data.

#### What would change in this codebase

| Component | Current state | Required change |
|---|---|---|
| `data_loader.py` (both frameworks) | Conservative augmentations (flip, small rotation, mild colour jitter) | Add a `SimCLRTransform` that produces two correlated views per image; augmentations must be much stronger (large random crop, strong colour distortion, Gaussian blur, optional grayscale) |
| `architectures.py` | Backbone + classification head | Add optional projection MLP (2–3 layers, output dim ~128–2048); this head is only active during SSL pre-training |
| `trainer.py` / `PyTorchTrainer` | Frozen → unfrozen supervised phases | Add a **Phase 0: SSL pre-training** stage before the existing frozen phase; the supervised phases follow unchanged once Phase 0 completes |
| `hyperparams.py` | No SSL params | Add `ssl_epochs`, `ssl_temperature` (NT-Xent τ), `ssl_projection_dim`; these are only active when SSL pre-training is enabled |
| `run_args.py` | No SSL mode flag | Add `ssl_mode: Optional[str]` (e.g. `"simclr"`, `"vicreg"`, `"barlow_twins"`, `"ijepa"`) |

#### Practical notes

- SimCLR is **batch-size sensitive**: performance improves significantly with large batches (4096+ is ideal). On constrained hardware, MoCo (a momentum-contrast variant) achieves similar results with smaller batches by maintaining a queue of negatives.
- The strong augmentation set used in SSL pre-training is different from what's used in supervised fine-tuning — the data loader needs to toggle between the two.
- Phase 0 runs on the **raw training images without labels**. The `ImageFolderDataset` class can support this by ignoring the label/subdirectory structure.

---

## 2. Dimension-Contrastive Learning

Dimension-contrastive methods avoid using negative pairs entirely. Instead of contrasting across samples, they prevent representation collapse by enforcing statistical constraints **across the dimensions of the embedding**.

### VICReg

VICReg (Bardes et al., 2022 — Meta AI / Yann LeCun's group) defines a three-term loss over two views `Z_A` and `Z_B` (each is a matrix of shape `[batch, embedding_dim]`):

1. **Invariance** (`s`): MSE between `Z_A` and `Z_B` — the encoder should produce the same embedding for both views.
2. **Variance** (`v`): Penalise any dimension whose standard deviation within the batch falls below a threshold γ (default 1). This prevents all embeddings from collapsing to the same point.
3. **Covariance** (`c`): Penalise the off-diagonal elements of the covariance matrix of `Z`. This decorrelates dimensions and prevents dimensional collapse (many redundant dimensions encoding the same information).

```
L = λ·s(Z_A, Z_B) + μ·[v(Z_A) + v(Z_B)] + ν·[c(Z_A) + c(Z_B)]
```

Default coefficients: λ=25, μ=25, ν=1.

#### Advantages over SimCLR in this context

- No need for large batches or negative pair mining — the variance term handles collapse prevention globally.
- Simpler to implement: no temperature tuning.
- Tends to be more robust to augmentation choices than sample-contrastive methods.

#### What would change in this codebase

The data loading and training phase changes are the same structure as SimCLR above. The only substantive difference is the loss function:

| Component | SimCLR | VICReg |
|---|---|---|
| Loss function | NT-Xent (requires negative pairs within batch) | VICReg (three regularisation terms, no negatives) |
| Batch size sensitivity | High | Low — works well at standard batch sizes |
| Temperature hyperparameter | Required | Not required |
| New hyperparams | `ssl_temperature`, `ssl_projection_dim` | `vic_lambda`, `vic_mu`, `vic_nu`, `ssl_projection_dim` |

VICReg is arguably the easier first SSL method to add to this pipeline precisely because it doesn't require large batches and has no awkward negative-mining logic.

---

### Barlow Twins

Barlow Twins (Zbontar et al., 2021 — also Meta AI) takes a different but related angle. Given embeddings `Z_A` and `Z_B`, it computes the **cross-correlation matrix** `C` between the two branches (shape `[embedding_dim, embedding_dim]`):

```
C_ij = Σ_b z_A_bi · z_B_bj  (after batch-normalising each dimension)
```

The loss pushes `C` toward the identity matrix:

- **Diagonal** (`C_ii → 1`): the same feature should respond identically to both views (invariance).
- **Off-diagonal** (`C_ij → 0, i≠j`): different features should be as uncorrelated as possible (redundancy reduction).

This is Barlow's "redundancy reduction" principle applied to self-supervised learning. The connection to VICReg: VICReg's covariance term is essentially the same constraint, expressed via the intra-branch covariance matrix instead of the cross-branch correlation matrix.

#### Relationship to VICReg

The two methods are close enough that they could share most of the same integration code (projection head, dual-view data loader, SSL training phase), with only the loss function differing. A clean implementation would define an `SSLLoss` base class/protocol and plug in `VICRegLoss` or `BarlowTwinsLoss` as configured.

---

## 3. JEPA — Joint Embedding Predictive Architecture

### Concept

JEPA represents a fundamental shift in *what the model is asked to learn*. Instead of:

> "Produce the same representation for two augmented views"

JEPA asks:

> "Given a partial view (context), **predict the representation** of a different partial view (target)"

The critical distinction is that the prediction happens **in embedding space, not pixel space**. This is what separates JEPA from masked autoencoders (MAE) like the original BERT for images. Reconstructing pixels requires the model to spend capacity on low-level texture details that are not useful for high-level understanding. Predicting in latent space forces the model to learn abstract, semantic features.

### I-JEPA (Image JEPA — Assran et al., 2023, Meta AI)

I-JEPA is the primary published image instantiation of the JEPA concept:

1. Divide the image into a grid of patches.
2. Sample a **context region** (a large random subset of patches, with the target regions masked out of context).
3. Sample several **target blocks** (rectangular regions of patches).
4. Pass the context patches through a **context encoder** `f_θ`.
5. Pass the *full* image through a **target encoder** `f_ξ` (EMA-updated copy of context encoder, not directly trained).
6. A lightweight **predictor** `g_φ` takes the context representation + positional tokens for each target block and predicts the target encoder's embedding for those blocks.
7. Loss: L2 between predictor output and target encoder output (stop-gradient on the target encoder side; gradients flow only through the context encoder and predictor).

The EMA target encoder is the same technique used in BYOL — it prevents collapse without requiring negatives or explicit variance constraints.

#### Why I-JEPA is particularly relevant for image classification

- It naturally produces patch-level representations that capture **spatial structure** — useful if the pipeline later supports detection, segmentation, or generation tasks.
- The masking strategy (predict missing regions from context) is a strong inductive bias for learning object-level semantics rather than texture.
- Performance scales well with ViT (Vision Transformer) backbones but the predictor concept is architecture-agnostic.

#### What would change in this codebase

I-JEPA requires more invasive changes than VICReg or SimCLR:

| Component | Change required |
|---|---|
| `architectures.py` | Need a patch-based encoder. ViT is the natural fit. The existing CNN backbones (ResNet, EfficientNet) do not produce spatially-indexed patch tokens natively — they would need either a patch embedding wrapper or replacement with a ViT architecture. |
| New: `architectures/predictor.py` | A narrow transformer predictor (typically 12 layers, narrower than the context encoder) that takes context tokens + positional queries and outputs target token predictions. |
| `data_loader.py` | Need a `JEPAMaskingTransform` that generates the context mask and a list of target block coordinates per image. These coordinates are needed by the predictor. |
| `trainer.py` | New SSL Phase 0 with the EMA update loop (Exponential Moving Average of context encoder weights into target encoder). The EMA momentum typically starts at ~0.996 and anneals toward 1.0 over training. |
| `hyperparams.py` | `jepa_context_scale`, `jepa_target_aspect_ratio`, `jepa_num_targets`, `jepa_ema_momentum` |

#### Practical trade-offs vs. contrastive/dimension-contrastive methods

| | SimCLR / Barlow / VICReg | I-JEPA |
|---|---|---|
| Architecture requirement | Works with existing CNNs | Works best with ViT; CNNs need patching wrappers |
| Augmentation sensitivity | High (augmentation choice is critical) | Lower (masking strategy matters more than pixel-level augmentation) |
| Implementation complexity | Low–Medium | High |
| Scalability | Good | Excellent (scales with model size) |
| Downstream task flexibility | Classification-optimised | Better for dense tasks (detection, segmentation) as well as classification |

---

## 4. LeJEPA

LeJEPA is Yann LeCun's most recent extension of the JEPA concept. The "Le" prefix reflects LeCun's continued evolution of the world model / energy-based model framework that JEPA is part of. At time of writing, the full technical details of LeJEPA are not yet widely published, but the conceptual direction extends JEPA toward:

- **Hierarchical prediction**: Multiple levels of abstraction predict at different granularities (patch-level up to scene-level), closer to how LeCun describes a full world model stack.
- **Multi-modal grounding**: Bridging visual and other modalities (language, action) through the joint embedding space — less directly applicable to this pipeline's current scope, but relevant if audio, text, or sensor data is ever added as an input modality.
- **Planning and imagination**: Using the predictor as a forward model to simulate future states — directly applicable to video or temporal-sequence extensions.

For this pipeline, LeJEPA's relevance is forward-looking: the architecture choices made now (patch-based encoders, separate predictor networks, EMA target encoders) are exactly the building blocks that LeJEPA and any future JEPA variant will use. Building in I-JEPA today is building the foundation for LeJEPA tomorrow.

---

## 5. Integration Roadmap

The cleanest way to add these to the existing pipeline without breaking its current supervised-fine-tuning workflow:

### Phase 0: SSL Pre-training (new, optional)

Insert before the existing frozen phase. Controlled by a new `ssl_mode` field in `TrainingRunArgs`. When set, the trainer runs SSL pre-training on the training images (labels ignored), then discards the projection head / predictor before handing the backbone to the existing frozen → unfrozen supervised phases.

```
[Phase 0: SSL pre-training]   ← new, optional
    ↓ (discard projection head / predictor)
[Phase 1: Frozen fine-tuning] ← existing, unchanged
    ↓
[Phase 2: Unfrozen fine-tuning] ← existing, unchanged
```

### Augmentation strategy

The existing train transforms in `data_loader.py` are appropriate for supervised fine-tuning. SSL pre-training needs a separate, stronger augmentation pipeline. The `create_data_loaders()` function would need a `mode` parameter (`"supervised"` vs `"ssl"`) to select the appropriate transforms — or a dedicated `create_ssl_data_loaders()` factory.

### Recommended implementation order

1. **VICReg** — lowest complexity, no batch-size constraints, uses existing CNN backbones. Best first step.
2. **Barlow Twins** — same infrastructure as VICReg, just a different loss. Near-free once VICReg is in.
3. **SimCLR / MoCo** — requires large-batch or queue infrastructure, but well-understood.
4. **I-JEPA** — requires ViT support (new architecture family) and predictor network. Significant but high-value work.
5. **LeJEPA extensions** — builds on I-JEPA infrastructure.

### New hyperparameters summary

| Parameter | Applies to | Description |
|---|---|---|
| `ssl_mode` | all SSL | `None` \| `"vicreg"` \| `"barlow_twins"` \| `"simclr"` \| `"ijepa"` |
| `ssl_epochs` | all SSL | Epochs for Phase 0 pre-training |
| `ssl_projection_dim` | contrastive + dim-contrastive | Output dimension of projection MLP |
| `ssl_temperature` | SimCLR | NT-Xent temperature τ |
| `vic_lambda` / `vic_mu` / `vic_nu` | VICReg | Invariance / variance / covariance loss weights |
| `barlow_lambda` | Barlow Twins | Off-diagonal penalty weight |
| `jepa_context_scale` | I-JEPA | Fraction of patches used as context |
| `jepa_num_targets` | I-JEPA | Number of target blocks to predict per image |
| `jepa_ema_momentum` | I-JEPA | Starting EMA momentum for target encoder |

---

## 6. Relevance to Future Model Types

| Future model type | Most applicable SSL strategy | Reason |
|---|---|---|
| Image generation | VICReg / I-JEPA | Learning disentangled, structured latent spaces directly improves generative model quality |
| Object detection | I-JEPA / JEPA family | Patch-level representations with spatial awareness are directly useful for localisation |
| Image segmentation | I-JEPA | Same spatial patch structure; context-prediction maps well to dense prediction tasks |
| Video classification | V-JEPA (video variant of I-JEPA) | Temporal prediction in latent space; same predictor architecture |
| Multi-modal (image + text) | LeJEPA / CLIP-style contrastive | Joint embedding across modalities; JEPA principles extend naturally |

---

## References

- Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Bardes, Ponce, LeCun (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- Zbontar et al. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA). [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- LeCun (2022). *A Path Towards Autonomous Machine Intelligence*. [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf) — foundational JEPA concept paper
