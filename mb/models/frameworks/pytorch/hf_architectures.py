"""
Hugging Face ``transformers`` vision backbones for the PyTorch trainer.

``transformers`` classification models return an ``ImageClassifierOutput`` dataclass and
take their input as ``pixel_values``, while the training and evaluation loops here call
``model(inputs)`` and feed the result straight to a loss function that expects a tensor.
:class:`HFImageClassifier` bridges that gap so a Hugging Face backbone is usable through
the existing architecture registry with no changes to the loops.

``transformers`` is an optional dependency. Registration is guarded, so an install without
it simply does not offer these architectures rather than failing to import.

The SigLIP2 entries here are the **fixed-resolution** checkpoints. Their ``config.json``
declares ``model_type: siglip``, so they load through the SigLIP classes rather than the
``Siglip2`` ones, which serve the variable-resolution (naflex) variant. Resolving the class
through ``AutoModelForImageClassification`` keeps that detail with the checkpoint instead of
hardcoding it here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Union

import torch.nn as nn

from mb.models.frameworks.registry import register_architecture
from mb.models.types import ArchitectureType, FrameworkType, LabelMode
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

_FW = FrameworkType.PYTORCH

# ``transformers`` config field recording what kind of head this is. The training loops
# compute their own loss, so this is metadata travelling with the checkpoint rather than a
# loss selector — but it has to agree with the loss that was actually applied.
_PROBLEM_TYPES: Dict[LabelMode, str] = {
    LabelMode.SINGLE_LABEL: "single_label_classification",
    LabelMode.MULTI_LABEL: "multi_label_classification",
}

SINGLE_LABEL_PROBLEM_TYPE = _PROBLEM_TYPES[LabelMode.SINGLE_LABEL]


def problem_type_for(label_mode: Union[LabelMode, str, None]) -> str:
    """``transformers`` ``problem_type`` string for a label mode (unknown values fall back)."""
    mode = LabelMode.try_from(label_mode) or LabelMode.get_default()
    return _PROBLEM_TYPES[mode]

# First release carrying SigLIP2; the fixed-resolution checkpoints named below were
# published against it.
MIN_TRANSFORMERS_VERSION = "4.50.0"

# Canonical architecture id -> Hugging Face hub id.
SIGLIP2_HUB_IDS: Dict[str, str] = {
    ArchitectureType.SIGLIP2_BASE_PATCH16_224.value: "google/siglip2-base-patch16-224",
    ArchitectureType.SIGLIP2_BASE_PATCH16_256.value: "google/siglip2-base-patch16-256",
    ArchitectureType.SIGLIP2_BASE_PATCH16_384.value: "google/siglip2-base-patch16-384",
}

# Vision-tower geometry for the ``-base-patch16-*`` checkpoints, used only to build an
# equivalent randomly-initialized model offline. Everything else is a SigLIP base default.
_SIGLIP2_PATCH_SIZE = 16


class HFImageClassifier(nn.Module):
    """
    Adapter presenting a ``transformers`` classification model as a plain module.

    :meth:`forward` returns the logits tensor, and :meth:`head_parameters` names the
    classification head so the frozen training phase does not have to guess at attribute
    names on the wrapped model.
    """

    def __init__(self, model: nn.Module, architecture: str, num_classes: int) -> None:
        super().__init__()
        self.hf = model
        self.architecture = architecture
        self.num_classes = num_classes

    def forward(self, x):
        return self.hf(pixel_values=x).logits

    @property
    def classifier(self) -> nn.Module:
        """The classification head, for callers that reach for ``model.classifier``."""
        return self.hf.classifier

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the classification head only (everything else is the backbone)."""
        return self.hf.classifier.parameters()

    def set_problem_type(self, label_mode: Union[LabelMode, str, None]) -> None:
        """
        Record the label mode on the wrapped config.

        Set after construction rather than passed to the factory: the factory signature is
        shared with the torchvision ones, which forward unrecognized keywords straight into
        their constructors and would reject it. The value is metadata — the training loop
        computes its own loss — but a consumer that loads this checkpoint through
        ``transformers`` and supplies labels will use it to pick a loss, so it has to agree
        with what was actually trained.
        """
        self.hf.config.problem_type = problem_type_for(label_mode)


def _require_transformers():
    """Import ``transformers``, raising a message that names the missing dependency."""
    try:
        from transformers import AutoModelForImageClassification
    except ImportError as e:
        raise ImportError(
            _(
                "transformers>={version} is required for Hugging Face backbones. "
                "Install it with: pip install -e \".[transformers]\""
            ).format(version=MIN_TRANSFORMERS_VERSION)
        ) from e
    return AutoModelForImageClassification


def create_siglip2(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    *,
    label_mode: Union[LabelMode, str, None] = None,
    **kwargs: Any,
) -> HFImageClassifier:
    """
    Create a fixed-resolution SigLIP2 classifier.

    Args:
        architecture: One of the ``siglip2_base_patch16_*`` ids.
        num_classes: Number of output classes.
        pretrained: True downloads the checkpoint from the Hugging Face hub (or reads the
            local cache). False builds an equivalent randomly-initialized model and makes
            **no network access**, which is what keeps the test suite runnable offline.
        label_mode: Selects the ``problem_type`` recorded on the config. Defaults to
            single-label, matching the pipeline default.

    Returns:
        The model wrapped so it returns a logits tensor.
    """
    arch_s = architecture.value if isinstance(architecture, ArchitectureType) else str(architecture).strip().lower()
    hub_id = SIGLIP2_HUB_IDS.get(arch_s)
    if hub_id is None:
        raise ValueError(
            f"Unknown SigLIP2 architecture: {arch_s}. Supported: {sorted(SIGLIP2_HUB_IDS)}"
        )

    problem_type = problem_type_for(label_mode)
    AutoModelForImageClassification = _require_transformers()

    if pretrained:
        # ignore_mismatched_sizes lets an already-fine-tuned checkpoint be re-headed for a
        # different class count instead of raising. The base checkpoints named here carry no
        # classification head at all, so it changes nothing for them; it matters when
        # someone points this at a downstream fine-tune.
        model = AutoModelForImageClassification.from_pretrained(
            hub_id,
            num_labels=num_classes,
            problem_type=problem_type,
            ignore_mismatched_sizes=True,
        )
        logger.info("Created %s from %s with %d classes", arch_s, hub_id, num_classes)
        return HFImageClassifier(model, arch_s, num_classes)

    # Offline: build the same architecture from a config rather than fetching one. The
    # image size drives the patch grid, so it has to match the named checkpoint.
    from mb.models.preprocessing import preprocessing_spec_for

    image_size = preprocessing_spec_for(arch_s).image_size
    from transformers import SiglipConfig

    config = SiglipConfig(
        vision_config={"image_size": image_size, "patch_size": _SIGLIP2_PATCH_SIZE},
        num_labels=num_classes,
        problem_type=problem_type,
    )
    model = AutoModelForImageClassification.from_config(config)
    logger.info(
        "Created %s (randomly initialized, %dpx) with %d classes", arch_s, image_size, num_classes
    )
    return HFImageClassifier(model, arch_s, num_classes)


def _make_siglip2_factory(arch_name: str):
    return lambda num_classes, pretrained=True, **kwargs: create_siglip2(
        arch_name, num_classes, pretrained, **kwargs
    )


def transformers_available() -> bool:
    """
    Whether ``transformers`` is installed, without importing it.

    ``find_spec`` only locates the module; importing it costs seconds and a large amount of
    memory. This runs during package import, which DataLoader worker processes also pay, so
    it must stay cheap — the real import happens when a factory is actually called.
    """
    from importlib.util import find_spec

    try:
        return find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def register_hf_architectures() -> bool:
    """
    Register the Hugging Face backbones, if ``transformers`` is installed.

    Returns:
        True if they were registered, False if the dependency is missing.
    """
    if not transformers_available():
        logger.debug("transformers not installed; Hugging Face architectures not registered")
        return False

    for arch in (
        ArchitectureType.SIGLIP2_BASE_PATCH16_224,
        ArchitectureType.SIGLIP2_BASE_PATCH16_256,
        ArchitectureType.SIGLIP2_BASE_PATCH16_384,
    ):
        try:
            register_architecture(_FW, arch, _make_siglip2_factory(arch.value))
        except ValueError:
            # Already registered (module re-imported); keep the first registration.
            pass
    return True


register_hf_architectures()
