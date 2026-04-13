"""
Command-line interface for Model Builder.

This module provides the main CLI entry point and subcommand structure.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

from mb import __version__
from mb.pipeline_config import (
    data_class_layout_defaults,
    gather_pipeline_defaults,
    get_pipeline_config,
    reload_pipeline_config,
)
from mb.utils.logging_setup import setup_logging
from mb.utils.translations import _

# Import data processing modules
from mb.data.gather import ImageGatherer
from mb.data.convert import ImageConverter
from mb.data.deduplicate import ImageDeduplicator
from mb.data.upscale import ImageUpscaler
from mb.data.dataset import DatasetCreator
from mb.info_inspect import dataset_info_text, model_info_text

# Import training modules
from mb.training.run_args import TrainingRunArgs, load_training_run_args_json
from mb.models.types import (
    ArchitectureType,
    ConversionTargetFormat,
    FrameworkType,
    InfoSubcommand,
    ModelBuildStepCommand,
    ModelType,
)

# CLI ``--model-type`` for gather/convert (all declared pipeline types).
_MODEL_TYPE_CLI_CHOICES = [m.value for m in ModelType]

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="mb",
        description=_("Model Builder - A unified CLI for building ML models"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    _gather_def = gather_pipeline_defaults()

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help=_("Path to configuration file (YAML)"),
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help=_("Enable verbose logging"),
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help=_("Available commands"),
        metavar="COMMAND",
    )
    
    # Data subcommands
    data_parser = subparsers.add_parser(
        "data",
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
        "gather",
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
        choices=_MODEL_TYPE_CLI_CHOICES,
        help=_(
            "Pipeline model type (default: model.default_type). "
            "When image_classification, gather also considers configured video extensions."
        ),
    )

    # mb data convert
    convert_parser = data_subparsers.add_parser(
        "convert",
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
        choices=_MODEL_TYPE_CLI_CHOICES,
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
        "deduplicate",
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
        "upscale",
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
        "create-dataset",
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
        default=1000,
        help=_(
            "Number of items per class in the test split when mode is fixed (default: 1000). "
            "For dataset-weighted mode this is the anchor scale — see --test-split-mode."
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
        help=_("Random seed for reproducibility"),
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

    # mb data fix-jpeg-extension-mismatch
    fix_jpeg_parser = data_subparsers.add_parser(
        "fix-jpeg-extension-mismatch",
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
        choices=_MODEL_TYPE_CLI_CHOICES,
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
        "estimate-space",
        help=_("Estimate disk space needed for convert or create-dataset"),
        description=_(
            "Walks source files (same rules as convert/dataset) and compares a rough byte estimate "
            "to free space on the target volume. Exits non-zero if the estimate exceeds free space."
        ),
    )
    estimate_space_parser.add_argument(
        "--operation",
        choices=["convert", "create-dataset"],
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

    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help=_("Train a model"),
        description=_(
            "Train a machine learning model using the specified framework and architecture. "
            "Supports transfer learning with frozen/unfrozen training phases."
        ),
    )
    train_parser.add_argument(
        "--model-type",
        choices=["image_classification"],
        default="image_classification",
        help=_("Model type (default: image_classification)"),
    )
    train_parser.add_argument(
        "--framework",
        choices=[f.value for f in FrameworkType],
        help=_("Framework to use (default: from config)"),
    )
    train_parser.add_argument(
        "--architecture",
        help=_("Model architecture (e.g., resnet34)"),
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        help=_("Data directory (default: from config)"),
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help=_("Output directory for models (default: from config)"),
    )
    train_parser.add_argument(
        "--frozen-epochs",
        type=int,
        help=_("Number of frozen training epochs (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-epochs",
        type=int,
        help=_("Number of unfrozen training epochs (default: from config)"),
    )
    train_parser.add_argument(
        "--frozen-lr",
        type=float,
        help=_("Learning rate for frozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-lr-max",
        type=float,
        help=_("Maximum learning rate for unfrozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--unfrozen-lr-min",
        type=float,
        help=_("Minimum learning rate for unfrozen phase (default: from config)"),
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help=_("Batch size (default: from config or auto-detect)"),
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        help=_("Image size (default: 224)"),
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help=_("Number of data loading workers (default: from config)"),
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help=_("Path to checkpoint to resume training from"),
    )
    train_parser.add_argument(
        "--run-id",
        type=str,
        help=_("Run ID of unified snapshot to update (auto-detects latest if not provided)"),
    )
    train_parser.add_argument(
        "--skip-snapshot-update",
        action="store_true",
        help=_("Skip updating the unified snapshot with training data"),
    )
    train_parser.add_argument(
        "--train-args-json",
        type=Path,
        metavar="PATH",
        help=_(
            "Load TrainingRunArgs from JSON (see mb.training.run_args); other train flags are ignored"
        ),
    )
    
    # Convert command
    convert_model_parser = subparsers.add_parser(
        "convert",
        help=_("Convert model between formats"),
        description=_(
            "Convert a trained model between different formats. "
            "Supports PyTorch -> ONNX, PyTorch -> SafeTensors, and Keras -> ONNX."
        ),
    )
    convert_model_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=_("Input model file"),
    )
    convert_model_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=_("Output model file"),
    )
    convert_model_parser.add_argument(
        "--framework",
        choices=[f.value for f in FrameworkType],
        help=_("Source framework (auto-detected if not specified)"),
    )
    convert_model_parser.add_argument(
        "--target",
        choices=[f.value for f in ConversionTargetFormat],
        required=True,
        help=_("Target format (onnx or safetensors)"),
    )
    convert_model_parser.add_argument(
        "--architecture",
        help=_("Model architecture (required for PyTorch -> ONNX conversion, e.g., 'resnet34')"),
    )
    convert_model_parser.add_argument(
        "--num-classes",
        type=int,
        help=_("Number of output classes (required for PyTorch -> ONNX conversion)"),
    )
    convert_model_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=_("Input image size (default: 224, used for ONNX conversion)"),
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help=_("Show information about models or datasets"),
        description=_(
            "Display information about trained models or datasets, including metadata, "
            "architecture details, and dataset statistics."
        ),
    )
    info_subparsers = info_parser.add_subparsers(
        dest="info_command",
        help=_("Info subcommands"),
        metavar="SUBCOMMAND",
    )
    
    # mb info model
    info_model_parser = info_subparsers.add_parser(
        "model",
        help=_("Show model information"),
        description=_(
            "Display detailed information about a trained model, including architecture, "
            "framework, number of parameters, and training metadata."
        ),
    )
    info_model_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help=_("Path to model file"),
    )
    
    # mb info dataset
    info_dataset_parser = info_subparsers.add_parser(
        "dataset",
        help=_("Show dataset information"),
        description=_(
            "Display statistics about a dataset, including class distributions, "
            "image counts, and data directory structure."
        ),
    )
    info_dataset_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=_("Path to data directory"),
    )
    
    return parser


def handle_data_gather(args):
    """Handle 'mb data gather' command."""
    try:
        reload_pipeline_config(getattr(args, "config", None), force=True)
        # Validate subdirectories
        if not args.subdirs:
            logger.error(_("Please specify valid subdirectories using --subdirs argument"))
            return 1
        
        # Parse subdirectory weights if provided
        subdir_weights = {}
        if hasattr(args, 'subdir_weights') and args.subdir_weights:
            for pair in args.subdir_weights.split(','):
                if ':' not in pair:
                    logger.error(
                        _("Invalid weight format: {pair}. Expected format: subdir:weight").format(
                            pair=pair
                        )
                    )
                    return 1
                subdir, weight_str = pair.split(':', 1)
                subdir = subdir.strip()
                weight = float(weight_str.strip())
                if weight < 0:
                    logger.error(
                        _("Weight must be non-negative: {subdir}={weight}").format(
                            subdir=subdir, weight=weight
                        )
                    )
                    return 1
                subdir_weights[subdir] = weight
            
            # Validate that weighted subdirectories exist in subdirs
            invalid_weights = set(subdir_weights.keys()) - set(args.subdirs)
            if invalid_weights:
                logger.error(
                    _("Subdirectories in weights not found in --subdirs: {weights}").format(
                        weights=invalid_weights
                    )
                )
                return 1
        
        layout = data_class_layout_defaults()
        mt = ModelType.from_pipeline_value(
            getattr(args, "model_type", None) or get_pipeline_config().get("model.default_type")
        )
        # Create and run gatherer
        gatherer = ImageGatherer(
            source_dir=str(args.source_dir),
            valid_subdirs=args.subdirs,
            target_dir=Path(args.target_dir),
            target_count=args.target_count,
            rejected_dir=Path(args.rejected_dir) if args.rejected_dir is not None else None,
            subdir_weights=subdir_weights if subdir_weights else None,
            class_qualifying_subdir=layout.get("class_qualifying_subdir"),
            model_type=mt,
        )
        
        # Store raw data directory
        gatherer.raw_data_dir = args.raw_data_dir
        
        success = gatherer.run()
        return 0 if success else 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error in data gather: {e}", exc_info=True)
        return 1


def handle_data_convert(args):
    """Handle 'mb data convert' command."""
    # TODO: Pass args.format into ImageConverter (or downstream) when non-JPEG targets are supported.
    # Today ImageConverter is JPEG-oriented; --format is accepted but not yet applied.
    try:
        reload_pipeline_config(getattr(args, "config", None), force=True)
        mt = ModelType.from_pipeline_value(
            getattr(args, "model_type", None) or get_pipeline_config().get("model.default_type")
        )
        converter = ImageConverter(raw_data_dir=args.raw_data_dir, model_type=mt)
        success = converter.run(
            skip_space_check=getattr(args, "skip_space_check", False),
            run_id=getattr(args, "run_id", None),
        )
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data convert: {e}", exc_info=True)
        return 1


def handle_data_deduplicate(args):
    """Handle 'mb data deduplicate' command."""
    try:
        deduplicator = ImageDeduplicator(raw_data_dir=args.raw_data_dir)
        success = deduplicator.run(
            list_only=bool(getattr(args, "list_only", False)),
            run_id=getattr(args, "run_id", None),
        )
        if success and getattr(args, "list_only", False):
            print(deduplicator.duplicate_groups_as_json())
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data deduplicate: {e}", exc_info=True)
        return 1


def handle_data_upscale(args):
    """Handle 'mb data upscale' command."""
    try:
        # Determine review directory
        if hasattr(args, 'review_dir') and args.review_dir:
            review_dir = args.review_dir
        else:
            review_dir = args.raw_data_dir / "small_images_review"
        
        upscaler = ImageUpscaler(review_dir=review_dir)
        success = upscaler.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data upscale: {e}", exc_info=True)
        return 1


def handle_data_create_dataset(args):
    """Handle 'mb data create-dataset' command."""
    try:
        from mb.utils.storage import check_target_external_storage, check_same_drive
        from mb.data.dataset import confirm_user_action, normalize_test_split_mode

        reload_pipeline_config(getattr(args, "config", None), force=True)
        pc = get_pipeline_config()
        tsm = getattr(args, "test_split_mode", None)
        if tsm is None:
            tsm = pc.get("data.test_split_mode")
        test_split_mode = normalize_test_split_mode(tsm)
        tst = getattr(args, "test_small_class_threshold", None)
        if tst is None:
            tst = pc.get("data.test_small_class_threshold")
            if tst is not None:
                tst = int(tst)

        # Storage checks
        if check_target_external_storage(logger, args.data_dir, override=getattr(args, 'allow_external_storage', False)):
            logger.error(_("Process terminated due to external storage detection."))
            return 1
        
        # User confirmation for same drive case
        if check_same_drive(args.raw_data_dir, args.data_dir):
            if not confirm_user_action(logger, args):
                return 1
        
        # Set random seed if provided
        if hasattr(args, 'seed') and args.seed is not None:
            import random
            random.seed(args.seed)
            logger.info(f"Using random seed: {args.seed}")
        
        creator = DatasetCreator(
            raw_data_dir=args.raw_data_dir,
            data_dir=args.data_dir,
            test_per_class=args.test_per_class,
            balance_train=getattr(args, 'balance_train', False),
            max_train_per_class=getattr(args, 'max_train_per_class', None),
            run_id=getattr(args, 'run_id', None),
            skip_space_check=getattr(args, "skip_space_check", False),
            test_split_mode=test_split_mode,
            test_small_class_threshold=tst,
        )
        
        success = creator.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data create-dataset: {e}", exc_info=True)
        return 1


def handle_data_fix_jpeg_extension_mismatch(args):
    """Handle ``mb data fix-jpeg-extension-mismatch``."""
    import random

    try:
        reload_pipeline_config(getattr(args, "config", None), force=True)
        from mb.data.find_jpeg_extension_mismatches import repair_mislabeled_jpeg_extensions
        from mb.pipeline_config import gather_pipeline_defaults

        raw_data_dir = getattr(args, "raw_data_dir", None)
        if raw_data_dir is None:
            raw_data_dir = gather_pipeline_defaults()["raw_data_dir"]
        else:
            raw_data_dir = Path(raw_data_dir)

        dry_run = bool(getattr(args, "dry_run", False))
        report_json = bool(getattr(args, "report_json", False))
        report_pillow = bool(getattr(args, "report_pillow", False))
        report_quiet = bool(getattr(args, "report_quiet", False))
        if (report_pillow or report_quiet) and not dry_run:
            logger.error(_("--pillow and --quiet require --dry-run"))
            return 1
        if report_pillow and not report_json:
            logger.error(_("--pillow requires --json"))
            return 1

        mt = ModelType.from_pipeline_value(
            getattr(args, "model_type", None) or get_pipeline_config().get("model.default_type")
        )
        seed = getattr(args, "seed", None)
        rng = random.Random(seed) if seed is not None else None
        ok, stats = repair_mislabeled_jpeg_extensions(
            raw_data_dir,
            model_type=mt,
            dry_run=dry_run,
            json_lines=report_json,
            dry_run_pillow=report_pillow,
            dry_run_quiet=report_quiet,
            json_lines_to_logger=False,
            include_static_format_mismatches=bool(
                getattr(args, "include_static_format_mismatches", False)
            ),
            verbose=bool(getattr(args, "fix_jpeg_verbose", False)),
            rng=rng,
        )
        if not ok:
            return 1
        if dry_run and stats.actionable_mismatches_found > 0:
            return 1
        return 0
    except Exception as e:
        logger.error(f"Error in data fix-jpeg-extension-mismatch: {e}", exc_info=True)
        return 1


def handle_data_estimate_space(args):
    """Handle ``mb data estimate-space``."""
    try:
        reload_pipeline_config(getattr(args, "config", None), force=True)
        mt = ModelType.from_pipeline_value(get_pipeline_config().get("model.default_type"))
        if args.operation == "convert":
            from mb.space_estimate import run_convert_estimate

            report = run_convert_estimate(args.raw_data_dir, mt)
            print(report.message)
            logger.info(report.message)
            return 0 if report.ok else 2
        if args.data_dir is None:
            logger.error(_("--data-dir is required when operation is create-dataset"))
            return 1
        from mb.space_estimate import run_create_dataset_estimate

        report = run_create_dataset_estimate(args.raw_data_dir, args.data_dir)
        print(report.message)
        logger.info(report.message)
        return 0 if report.ok else 2
    except Exception as e:
        logger.error(f"Error in data estimate-space: {e}", exc_info=True)
        return 1


def handle_train(args):
    """Handle 'mb train' command."""
    try:
        from mb.training.trainer import ModelTrainer

        # Pipeline YAML (--config) for model/data/training/paths defaults
        reload_pipeline_config(getattr(args, "config", None), force=True)
        pipeline = get_pipeline_config()

        if getattr(args, "train_args_json", None):
            run_args = load_training_run_args_json(args.train_args_json)
            mt_cfg = ModelType.from_pipeline_value(pipeline.get("model.default_type"))
            if mt_cfg != ModelType.IMAGE_CLASSIFICATION:
                logger.error(_("Unsupported model type from config: {t}").format(t=mt_cfg.value))
                return 1
            model_type = ModelType.IMAGE_CLASSIFICATION
            data_dir = run_args.data_dir
            if not data_dir.exists():
                logger.error(_("Data directory does not exist: {path}").format(path=data_dir))
                return 1
            output_dir = run_args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            trainer = ModelTrainer(
                framework=run_args.framework,
                model_type=model_type,
                pipeline_config=pipeline,
            )
            supported_archs = trainer.get_supported_architectures()
            if run_args.architecture.value not in supported_archs:
                logger.error(
                    _("Architecture '{arch}' not supported for framework '{fw}'").format(
                        arch=run_args.architecture.value, fw=run_args.framework.value
                    )
                )
                logger.info(f"Supported architectures: {supported_archs}")
                return 1
            logger.info(
                f"Starting training from JSON ({run_args.framework.value}/{run_args.architecture.value})"
            )
            model_path = trainer.train(run_args)
            logger.info(
                _("Training completed successfully. Model saved to: {path}").format(path=model_path)
            )
            return 0
        
        # Determine framework
        framework_raw = args.framework or pipeline.get("model.default_framework", FrameworkType.get_default().value)
        fw = FrameworkType.try_from(framework_raw)
        if fw is None:
            logger.error(_("Unsupported framework: {fw}").format(fw=framework_raw))
            return 1
        # Determine model type
        mt = ModelType.from_pipeline_value(args.model_type or pipeline.get("model.default_type"))
        if mt != ModelType.IMAGE_CLASSIFICATION:
            logger.error(_("Unsupported model type: {t}").format(t=mt.value))
            return 1
        model_type = mt

        # Determine architecture
        arch_raw = args.architecture or pipeline.get("model.default_architecture", ArchitectureType.get_default().value)
        arch = ArchitectureType.try_from(arch_raw)
        if arch is None:
            logger.error(_("Unsupported architecture: {arch}").format(arch=arch_raw))
            return 1
        
        # Determine data directory
        data_dir = args.data_dir or Path(pipeline.get('data.data_dir', 'data'))
        if not data_dir.exists():
            logger.error(_("Data directory does not exist: {path}").format(path=data_dir))
            return 1
        
        # Determine output directory
        output_dir = args.output_dir or Path(pipeline.get('paths.models_dir', 'data/models'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare CLI hyperparameters
        cli_hyperparams = {}
        if args.frozen_epochs is not None:
            cli_hyperparams['frozen_epochs'] = args.frozen_epochs
        if args.unfrozen_epochs is not None:
            cli_hyperparams['unfrozen_epochs'] = args.unfrozen_epochs
        if args.frozen_lr is not None:
            cli_hyperparams['frozen_lr'] = args.frozen_lr
        if args.unfrozen_lr_max is not None:
            cli_hyperparams['unfrozen_lr_max'] = args.unfrozen_lr_max
        if args.unfrozen_lr_min is not None:
            cli_hyperparams['unfrozen_lr_min'] = args.unfrozen_lr_min
        if args.batch_size is not None:
            cli_hyperparams['batch_size'] = args.batch_size
        if args.image_size is not None:
            cli_hyperparams['image_size'] = args.image_size
        if args.num_workers is not None:
            cli_hyperparams['num_workers'] = args.num_workers
        
        # Create trainer
        trainer = ModelTrainer(
            framework=fw,
            model_type=model_type,
            pipeline_config=pipeline,
        )

        # Check if architecture is supported
        supported_archs = trainer.get_supported_architectures()
        if arch.value not in supported_archs:
            logger.error(
                _("Architecture '{arch}' not supported for framework '{fw}'").format(
                    arch=arch.value, fw=fw.value
                )
            )
            logger.info(f"Supported architectures: {supported_archs}")
            return 1

        # Train model
        logger.info(f"Starting training with {fw.value}/{arch.value}")
        run_args = TrainingRunArgs(
            framework=fw,
            architecture=arch,
            data_dir=data_dir,
            output_dir=output_dir,
            resume_from=args.resume_from,
            run_id=getattr(args, "run_id", None),
            update_snapshot=not getattr(args, "skip_snapshot_update", False),
            cli_hyperparams=dict(cli_hyperparams),
        )
        model_path = trainer.train(run_args)
        
        logger.info(
            _("Training completed successfully. Model saved to: {path}").format(path=model_path)
        )
        return 0
        
    except Exception as e:
        logger.error(_("Training failed: {err}").format(err=e), exc_info=args.verbose)
        return 1


def handle_convert(args):
    """Handle 'mb convert' command."""
    try:
        from mb.conversion.converters import convert_model
        
        # Validate arguments
        if not args.input.exists():
            logger.error(_("Input model file not found: {path}").format(path=args.input))
            return 1
        
        # Check if architecture/num_classes are needed
        source_framework = args.framework
        if source_framework is None:
            from mb.conversion.converters import detect_model_framework
            source_framework = detect_model_framework(args.input)
            if source_framework is None:
                logger.error(_("Could not detect source framework. Please specify --framework"))
                return 1
        
        if source_framework == FrameworkType.PYTORCH.value and args.target == ConversionTargetFormat.ONNX.value:
            if args.architecture is None or args.num_classes is None:
                logger.error(
                    _("PyTorch -> ONNX conversion requires --architecture and --num-classes")
                )
                return 1
        
        # Perform conversion
        logger.info(f"Converting {args.input} ({source_framework}) -> {args.output} ({args.target})")
        
        success = convert_model(
            input_path=args.input,
            output_path=args.output,
            source_framework=source_framework,
            target_format=args.target,
            architecture=args.architecture,
            num_classes=args.num_classes,
            image_size=args.image_size
        )
        
        if success:
            logger.info(
                _("Conversion completed successfully: {path}").format(path=args.output)
            )
            return 0
        else:
            logger.error(_("Conversion failed"))
            return 1
            
    except Exception as e:
        logger.error(_("Conversion error: {err}").format(err=e), exc_info=args.verbose)
        return 1


def handle_info_model(args):
    """Handle 'mb info model' command."""
    try:
        text = model_info_text(args.path)
    except FileNotFoundError:
        logger.error(_("Model path not found: {path}").format(path=args.path))
        return 1
    except ValueError as e:
        logger.error(str(e))
        return 1
    print(text)
    return 0


def handle_info_dataset(args):
    """Handle 'mb info dataset' command."""
    try:
        text = dataset_info_text(args.data_dir)
    except FileNotFoundError:
        logger.error(_("Data directory not found: {path}").format(path=args.data_dir))
        return 1
    except ValueError as e:
        logger.error(str(e))
        return 1
    print(text)
    return 0


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Optional list of command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    setup_logging(script_name="mb", log_level=log_level)

    # Handle commands
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    try:
        if parsed_args.command == "data":
            raw_cmd = parsed_args.data_command
            if not raw_cmd:
                logger.error(_("No data subcommand specified"))
                return 1
            try:
                step = ModelBuildStepCommand(raw_cmd)
            except ValueError:
                logger.error(_("Unknown data subcommand: {cmd}").format(cmd=raw_cmd))
                return 1
            _data_handlers = {
                ModelBuildStepCommand.GATHER: handle_data_gather,
                ModelBuildStepCommand.CONVERT: handle_data_convert,
                ModelBuildStepCommand.DEDUPLICATE: handle_data_deduplicate,
                ModelBuildStepCommand.UPSCALE: handle_data_upscale,
                ModelBuildStepCommand.CREATE_DATASET: handle_data_create_dataset,
                ModelBuildStepCommand.ESTIMATE_SPACE: handle_data_estimate_space,
                ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH: handle_data_fix_jpeg_extension_mismatch,
            }
            return _data_handlers[step](parsed_args)
        
        elif parsed_args.command == "train":
            return handle_train(parsed_args)
        
        elif parsed_args.command == "convert":
            return handle_convert(parsed_args)
        
        elif parsed_args.command == "info":
            raw_info = parsed_args.info_command
            if not raw_info:
                logger.error(_("No info subcommand specified"))
                return 1
            try:
                info_sub = InfoSubcommand(raw_info)
            except ValueError:
                logger.error(_("Unknown info subcommand: {cmd}").format(cmd=raw_info))
                return 1
            if info_sub == InfoSubcommand.MODEL:
                return handle_info_model(parsed_args)
            return handle_info_dataset(parsed_args)
        
        else:
            logger.error(_("Unknown command: {cmd}").format(cmd=parsed_args.command))
            return 1
    
    except KeyboardInterrupt:
        logger.info(_("Interrupted by user"))
        return 130
    except Exception as e:
        logger.error(_("Error: {err}").format(err=e), exc_info=parsed_args.verbose)
        return 1


def run_data_subcommand_cli(subcommand: str, argv: Optional[List[str]] = None) -> int:
    """
    Run ``mb data <subcommand>`` for ``python -m mb.data.<module>`` entry points.

    *subcommand* is a :class:`~mb.models.types.ModelBuildStepCommand` value string
    (``gather``, ``convert``, ``deduplicate``, ``upscale``, ``create-dataset``,
    ``fix-jpeg-extension-mismatch``, ``estimate-space``, …). *argv* defaults to ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]
    return main(["data", subcommand, *argv])


if __name__ == "__main__":
    sys.exit(main())
