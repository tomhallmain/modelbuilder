"""Argument parsers for ``mb data``."""

from __future__ import annotations

from pathlib import Path

from mb.cli.parsing.common import MODEL_TYPE_CLI_CHOICES
from mb.models.types import ModelBuildStepCommand
from mb.utils.constants import ModelBuilderTaskType
from mb.pipeline_config import gather_pipeline_defaults
from mb.utils.translations import _


def register(subparsers) -> None:
    _gather_def = gather_pipeline_defaults()
    data_parser = subparsers.add_parser(
        ModelBuilderTaskType.DATA.value,
        help=_("Data processing operations"),
        description=_("Data processing operations for preparing image datasets"),
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        help=_("Data subcommands"),
        metavar="SUBCOMMAND",
    )

    # mb data gather
    gather_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.GATHER.value,
        help=_("Gather images from source directories"),
        description=_(
            "Gather images from source directories into a target directory, "
            "with deduplication and optional weighting by subdirectory."
        ),
    )
    gather_parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help=_("Source directory containing images"),
    )
    gather_parser.add_argument(
        "--subdirs",
        nargs="+",
        required=True,
        help=_("Subdirectories to process"),
    )
    gather_parser.add_argument(
        "--target-count",
        type=int,
        default=_gather_def["target_count"],
        help=_(
            "Target number of images to gather (default: from pipeline data.gather.default_target_count). "
            "Treated as a limit, not an exact requirement."
        ),
    )
    gather_parser.add_argument(
        "--target-dir",
        type=Path,
        default=_gather_def["target_dir"],
        help=_("Target directory for gathered images (default: from pipeline data.gather.default_target_dir)"),
    )
    gather_parser.add_argument(
        "--rejected-dir",
        type=Path,
        default=_gather_def["rejected_dir"],
        help=_("Rejected directory for manually rejected images (default: data.gather.default_rejected_dir)"),
    )
    gather_parser.add_argument(
        "--subdir-weights",
        type=str,
        help=_(
            'Relative weights for subdirectories: "subdir1:weight1,subdir2:weight2" '
            '(e.g. "neutral:4,drawing:1" ≈ 80%%/20%%, or "neutral:0.8,drawing:0.2"). '
            "Weights are normalized automatically; any positive numbers work."
        ),
    )
    gather_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=_gather_def["raw_data_dir"],
        help=_("Root directory for raw data (default: data.raw_data_dir in pipeline config)"),
    )
    gather_parser.add_argument(
        "--model-type",
        default=None,
        choices=MODEL_TYPE_CLI_CHOICES,
        help=_(
            "Pipeline model type (default: model.default_type). "
            "When image_classification, gather also considers configured video extensions."
        ),
    )

    # mb data convert
    convert_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.CONVERT.value,
        help=_("Convert images to specified format"),
        description=_(
            "Convert images in the raw data directory to a specified format (e.g., JPEG). "
            "Large images are automatically resized to prevent memory issues."
        ),
    )
    convert_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help=_("Raw data directory (default: raw_data)"),
    )
    convert_parser.add_argument(
        "--format",
        choices=["jpeg", "jpg"],
        default="jpeg",
        help=_(
            "Target format for converted outputs (default: jpeg). "
            "The converter is currently JPEG-oriented; non-default values may be ignored until implemented."
        ),
    )
    convert_parser.add_argument(
        "--model-type",
        default=None,
        choices=MODEL_TYPE_CLI_CHOICES,
        help=_(
            "Pipeline model type (default: model.default_type). "
            "When image_classification, videos and multi-frame GIFs get a random frame as JPEG."
        ),
    )
    convert_parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help=_(
            "Allow convert to run even if the raw-data drive appears to have insufficient free space "
            "(heuristic estimate; not recommended)."
        ),
    )
    convert_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=_(
            "Unified snapshot run ID to update (snapshot_<run_id>.json under raw data). "
            "Omit to start a new snapshot with a new run ID."
        ),
    )

    # mb data deduplicate
    dedup_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.DEDUPLICATE.value,
        help=_("Remove duplicate images"),
        description=_(
            "Remove duplicate images within and across class directories. "
            "Uses perceptual hashing to identify duplicates and moves them to a review directory."
        ),
    )
    dedup_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help=_(
            "Root raw data directory containing class subdirectories (default: raw_data). "
            "Typical layouts include multiple class folders under this root."
        ),
    )
    dedup_parser.add_argument(
        "--list-only",
        action="store_true",
        help=_(
            "Scan and print duplicate groups as indented JSON (no removals). "
            "Useful for manual review workflows."
        ),
    )
    dedup_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=_(
            "Optional unified snapshot run ID to update with deduplication metadata. "
            "If omitted, the latest loadable snapshot under raw data is used when available."
        ),
    )

    # mb data upscale
    upscale_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.UPSCALE.value,
        help=_("Upscale small images"),
        description=_(
            "Upscale images that are smaller than a minimum dimension threshold. "
            "Small images are moved to a review directory for manual inspection before upscaling."
        ),
    )
    upscale_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help=_("Raw data directory (default: raw_data)"),
    )
    upscale_parser.add_argument(
        "--review-dir",
        type=Path,
        help=_(
            "Review directory containing small images to upscale "
            "(default: <raw-data-dir>/small_images_review). "
            "Upscaled outputs go under <review-dir>/upscaled_small_images."
        ),
    )

    # mb data create-dataset
    dataset_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.CREATE_DATASET.value,
        help=_("Create train/test dataset splits"),
        description=_(
            "Create training and test dataset splits from raw data. "
            "Validates images, removes corrupted files, filters by size, "
            "and creates balanced train/test splits with hash-based filenames."
        ),
    )
    dataset_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help=_("Raw data directory (default: raw_data)"),
    )
    dataset_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=_("Output data directory (default: data)"),
    )
    dataset_parser.add_argument(
        "--test-per-class",
        type=int,
        default=None,
        help=_(
            "Number of items per class in the test split when mode is fixed; also the anchor for "
            "dataset-weighted mode (default: data.test_per_class from pipeline YAML)."
        ),
    )
    dataset_parser.add_argument(
        "--test-split-mode",
        choices=["fixed", "dataset-weighted"],
        default=None,
        help=_(
            "fixed = test_per_class images per class; dataset-weighted = modulated counts from "
            "class size vs total (default: data.test_split_mode in pipeline YAML, else fixed)."
        ),
    )
    dataset_parser.add_argument(
        "--test-small-class-threshold",
        type=int,
        default=None,
        help=_(
            "With dataset-weighted mode: classes with fewer images than this use a proportional "
            "test count; larger classes use anchor + anchor×(class_share). "
            "Omit to use --test-per-class as the threshold (default: pipeline data.test_small_class_threshold)."
        ),
    )
    dataset_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=_(
            "Random seed for reproducibility (default: data.seed from pipeline YAML when set, else non-deterministic)"
        ),
    )
    dataset_parser.add_argument(
        "--run-id",
        type=str,
        help=_("Run ID of unified snapshot to update (auto-detects latest if not provided)"),
    )
    dataset_parser.add_argument(
        "--balance-train",
        action="store_true",
        help=_(
            "Balance the training set to the smallest class size (default: off, keeps natural proportions)."
        ),
    )
    dataset_parser.add_argument(
        "--max-train-per-class",
        type=int,
        help=_(
            "Maximum training items per class (no limit if omitted; keeps natural proportions below the cap)."
        ),
    )
    dataset_parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help=_("Allow running on external/removable storage (not recommended)"),
    )
    dataset_parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help=_(
            "Allow create-dataset even if the output data drive appears to have insufficient free space "
            "(heuristic estimate; not recommended)."
        ),
    )
    dataset_parser.add_argument(
        "--model-type",
        default=None,
        choices=MODEL_TYPE_CLI_CHOICES,
        help=_(
            "Pipeline model type (default: model.default_type). image_classification builds "
            "train/test ImageFolder splits from class subdirectories (all other options above "
            "apply); image_generation_lora instead copies every image directly under "
            "--raw-data-dir (no class folders, no train/test split) plus its optional .txt "
            "caption sidecar into --data-dir, for LoRA fine-tuning of an image-generation model."
        ),
    )

    # mb data fix-jpeg-extension-mismatch
    fix_jpeg_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH.value,
        help=_("Rename mislabeled .jpg sources and rebuild CONVERTED JPEGs"),
        description=_(
            "Finds non-JPEG bytes under .jpg/.jpeg names in class source trees (same discovery as convert), "
            "writes corrected JPEGs, and removes stale copies under CONVERTED and small_images_review "
            "only after a successful write. Animated GIFs use the same random-frame + visual_media_review "
            "layout as convert when the model type is image_classification."
        ),
    )
    fix_jpeg_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help=_(
            "Raw data directory (default: data.raw_data_dir from the pipeline config after --config; "
            "same as gather/convert when omitted)."
        ),
    )
    fix_jpeg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=_("Log planned repairs only; do not rename files or write outputs."),
    )
    fix_jpeg_parser.add_argument(
        "--json",
        dest="report_json",
        action="store_true",
        help=_(
            "Print newline-delimited JSON (stdout): dry-run or live repair. By default only actionable "
            "mismatches are listed; use -v to include policy-skipped PNG/WebP/BMP/TIFF under .jpg. "
            "Live repair emits one line per successful fix when applicable."
        ),
    )
    fix_jpeg_parser.add_argument(
        "--pillow",
        dest="report_pillow",
        action="store_true",
        help=_("With --dry-run and --json: include Pillow format and GIF metadata in each JSON object."),
    )
    fix_jpeg_parser.add_argument(
        "--quiet",
        dest="report_quiet",
        action="store_true",
        help=_(
            "With --dry-run: omit verbose per-file log lines (use with --json for machine-readable output only)."
        ),
    )
    fix_jpeg_parser.add_argument(
        "-v",
        "--verbose",
        dest="fix_jpeg_verbose",
        action="store_true",
        help=_(
            "With --json: include policy-skipped static-format mismatches in JSON output. "
            "In text mode: log each policy-skipped file and class progress for every folder; "
            "with live repair, log each skipped file. Useless with --quiet. "
            "Separate from top-level mb -v (global logging)."
        ),
    )
    fix_jpeg_parser.add_argument(
        "--include-static-format-mismatches",
        action="store_true",
        help=_(
            "Also rename/repair mislabeled .jpg/.jpeg whose bytes are PNG, WebP, BMP, or TIFF. "
            "By default those are counted and summarized per class only (GIF and animated-IC cases are always repaired)."
        ),
    )
    fix_jpeg_parser.add_argument(
        "--model-type",
        default=None,
        choices=MODEL_TYPE_CLI_CHOICES,
        help=_("Pipeline model type (default: model.default_type)."),
    )
    fix_jpeg_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=_("RNG seed for random GIF frame selection (optional)."),
    )

    # mb data estimate-space
    estimate_space_parser = data_subparsers.add_parser(
        ModelBuildStepCommand.ESTIMATE_SPACE.value,
        help=_("Estimate disk space needed for convert or create-dataset"),
        description=_(
            "Walks source files (same rules as convert/dataset) and compares a rough byte estimate "
            "to free space on the target volume. Exits non-zero if the estimate exceeds free space."
        ),
    )
    estimate_space_parser.add_argument(
        "--operation",
        choices=[
            ModelBuildStepCommand.CONVERT.value,
            ModelBuildStepCommand.CREATE_DATASET.value,
        ],
        required=True,
        help=_("Which step to estimate for"),
    )
    estimate_space_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help=_("Raw data directory (default: raw_data)"),
    )
    estimate_space_parser.add_argument(
        "--data-dir",
        type=Path,
        help=_("Output data directory (required when operation is create-dataset)"),
    )
