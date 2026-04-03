"""
Command-line interface for Model Builder.

This module provides the main CLI entry point and subcommand structure.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from mb import __version__
from mb.pipeline_config import get_pipeline_config, reload_pipeline_config
from mb.utils.logging_setup import setup_logging

# Import data processing modules
from mb.data.gather import ImageGatherer
from mb.data.convert import ImageConverter
from mb.data.deduplicate import ImageDeduplicator
from mb.data.upscale import ImageUpscaler
from mb.data.dataset import DatasetCreator

# Import training modules
from mb.training.run_args import TrainingRunArgs, load_training_run_args_json
from mb.models.types import ModelType

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="mb",
        description="Model Builder - A unified CLI for building ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Data subcommands
    data_parser = subparsers.add_parser(
        "data",
        help="Data processing operations",
        description="Data processing operations for preparing image datasets"
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        help="Data subcommands",
        metavar="SUBCOMMAND"
    )
    
    # mb data gather
    gather_parser = data_subparsers.add_parser(
        "gather",
        help="Gather images from source directories",
        description="Gather images from source directories into a target directory, "
                    "with deduplication and optional weighting by subdirectory."
    )
    gather_parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Source directory containing images"
    )
    gather_parser.add_argument(
        "--subdirs",
        nargs="+",
        required=True,
        help="Subdirectories to process"
    )
    gather_parser.add_argument(
        "--target-count",
        type=int,
        default=16000,
        help="Target number of images to gather (default: 16000)"
    )
    gather_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("raw_data/coherent"),
        help="Target directory for gathered images (default: raw_data/coherent)"
    )
    gather_parser.add_argument(
        "--rejected-dir",
        type=Path,
        help="Rejected directory for manually rejected images"
    )
    gather_parser.add_argument(
        "--subdir-weights",
        type=str,
        help='Relative weights for subdirectories in format "subdir1:weight1,subdir2:weight2"'
    )
    gather_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help="Root directory for raw data (default: raw_data)"
    )
    
    # mb data convert
    convert_parser = data_subparsers.add_parser(
        "convert",
        help="Convert images to specified format",
        description="Convert images in the raw data directory to a specified format (e.g., JPEG). "
                    "Large images are automatically resized to prevent memory issues."
    )
    convert_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help="Raw data directory (default: raw_data)"
    )
    convert_parser.add_argument(
        "--format",
        choices=["jpeg", "jpg"],
        default="jpeg",
        help="Target format (default: jpeg)"
    )
    
    # mb data deduplicate
    dedup_parser = data_subparsers.add_parser(
        "deduplicate",
        help="Remove duplicate images",
        description="Remove duplicate images within and across class directories. "
                    "Uses perceptual hashing to identify duplicates and moves them to a review directory."
    )
    dedup_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help="Raw data directory (default: raw_data)"
    )
    
    # mb data upscale
    upscale_parser = data_subparsers.add_parser(
        "upscale",
        help="Upscale small images",
        description="Upscale images that are smaller than a minimum dimension threshold. "
                    "Small images are moved to a review directory for manual inspection before upscaling."
    )
    upscale_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help="Raw data directory (default: raw_data)"
    )
    upscale_parser.add_argument(
        "--review-dir",
        type=Path,
        help="Review directory containing small images (default: raw_data/small_images_review)"
    )
    
    # mb data create-dataset
    dataset_parser = data_subparsers.add_parser(
        "create-dataset",
        help="Create train/test dataset splits",
        description="Create training and test dataset splits from raw data. "
                    "Validates images, removes corrupted files, filters by size, "
                    "and creates balanced train/test splits with hash-based filenames."
    )
    dataset_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("raw_data"),
        help="Raw data directory (default: raw_data)"
    )
    dataset_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output data directory (default: data)"
    )
    dataset_parser.add_argument(
        "--test-per-class",
        type=int,
        default=1000,
        help="Number of test images per class (default: 1000)"
    )
    dataset_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    dataset_parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID of unified snapshot to update (auto-detects latest if not provided)"
    )
    dataset_parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Balance training set to smallest class size"
    )
    dataset_parser.add_argument(
        "--max-train-per-class",
        type=int,
        help="Maximum number of training images per class"
    )
    dataset_parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help="Allow running on external/removable storage (not recommended)"
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        description="Train a machine learning model using the specified framework and architecture. "
                    "Supports transfer learning with frozen/unfrozen training phases."
    )
    train_parser.add_argument(
        "--model-type",
        choices=["image_classification"],
        default="image_classification",
        help="Model type (default: image_classification)"
    )
    train_parser.add_argument(
        "--framework",
        choices=["pytorch", "keras"],
        help="Framework to use (default: from config)"
    )
    train_parser.add_argument(
        "--architecture",
        help="Model architecture (e.g., resnet34)"
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: from config)"
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for models (default: from config)"
    )
    train_parser.add_argument(
        "--frozen-epochs",
        type=int,
        help="Number of frozen training epochs (default: from config)"
    )
    train_parser.add_argument(
        "--unfrozen-epochs",
        type=int,
        help="Number of unfrozen training epochs (default: from config)"
    )
    train_parser.add_argument(
        "--frozen-lr",
        type=float,
        help="Learning rate for frozen phase (default: from config)"
    )
    train_parser.add_argument(
        "--unfrozen-lr-max",
        type=float,
        help="Maximum learning rate for unfrozen phase (default: from config)"
    )
    train_parser.add_argument(
        "--unfrozen-lr-min",
        type=float,
        help="Minimum learning rate for unfrozen phase (default: from config)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default: from config or auto-detect)"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        help="Image size (default: 224)"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers (default: from config)"
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to checkpoint to resume training from"
    )
    train_parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID of unified snapshot to update (auto-detects latest if not provided)"
    )
    train_parser.add_argument(
        "--skip-snapshot-update",
        action="store_true",
        help="Skip updating the unified snapshot with training data"
    )
    train_parser.add_argument(
        "--train-args-json",
        type=Path,
        metavar="PATH",
        help="Load TrainingRunArgs from JSON (see mb.training.run_args); other train flags are ignored",
    )
    
    # Convert command
    convert_model_parser = subparsers.add_parser(
        "convert",
        help="Convert model between formats",
        description="Convert a trained model between different formats. "
                    "Supports PyTorch -> ONNX, PyTorch -> SafeTensors, and Keras -> ONNX."
    )
    convert_model_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input model file"
    )
    convert_model_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output model file"
    )
    convert_model_parser.add_argument(
        "--framework",
        choices=["pytorch", "keras"],
        help="Source framework (auto-detected if not specified)"
    )
    convert_model_parser.add_argument(
        "--target",
        choices=["onnx", "safetensors"],
        required=True,
        help="Target format (onnx or safetensors)"
    )
    convert_model_parser.add_argument(
        "--architecture",
        help="Model architecture (required for PyTorch -> ONNX conversion, e.g., 'resnet34')"
    )
    convert_model_parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of output classes (required for PyTorch -> ONNX conversion)"
    )
    convert_model_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (default: 224, used for ONNX conversion)"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about models or datasets",
        description="Display information about trained models or datasets, including metadata, "
                    "architecture details, and dataset statistics."
    )
    info_subparsers = info_parser.add_subparsers(
        dest="info_command",
        help="Info subcommands",
        metavar="SUBCOMMAND"
    )
    
    # mb info model
    info_model_parser = info_subparsers.add_parser(
        "model",
        help="Show model information",
        description="Display detailed information about a trained model, including architecture, "
                    "framework, number of parameters, and training metadata."
    )
    info_model_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to model file"
    )
    
    # mb info dataset
    info_dataset_parser = info_subparsers.add_parser(
        "dataset",
        help="Show dataset information",
        description="Display statistics about a dataset, including class distributions, "
                    "image counts, and data directory structure."
    )
    info_dataset_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to data directory"
    )
    
    return parser


def handle_data_gather(args):
    """Handle 'mb data gather' command."""
    try:
        # Validate subdirectories
        if not args.subdirs:
            logger.error("Please specify valid subdirectories using --subdirs argument")
            return 1
        
        # Parse subdirectory weights if provided
        subdir_weights = {}
        if hasattr(args, 'subdir_weights') and args.subdir_weights:
            for pair in args.subdir_weights.split(','):
                if ':' not in pair:
                    logger.error(f"Invalid weight format: {pair}. Expected format: subdir:weight")
                    return 1
                subdir, weight_str = pair.split(':', 1)
                subdir = subdir.strip()
                weight = float(weight_str.strip())
                if weight < 0:
                    logger.error(f"Weight must be non-negative: {subdir}={weight}")
                    return 1
                subdir_weights[subdir] = weight
            
            # Validate that weighted subdirectories exist in subdirs
            invalid_weights = set(subdir_weights.keys()) - set(args.subdirs)
            if invalid_weights:
                logger.error(f"Subdirectories in weights not found in --subdirs: {invalid_weights}")
                return 1
        
        # Create and run gatherer
        gatherer = ImageGatherer(
            source_dir=str(args.source_dir),
            valid_subdirs=args.subdirs,
            target_dir=args.target_dir,
            target_count=args.target_count,
            rejected_dir=args.rejected_dir if hasattr(args, 'rejected_dir') and args.rejected_dir else None,
            subdir_weights=subdir_weights if subdir_weights else None
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
    try:
        converter = ImageConverter(raw_data_dir=args.raw_data_dir)
        success = converter.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data convert: {e}", exc_info=True)
        return 1


def handle_data_deduplicate(args):
    """Handle 'mb data deduplicate' command."""
    try:
        deduplicator = ImageDeduplicator(raw_data_dir=args.raw_data_dir)
        success = deduplicator.run()
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
        from mb.data.dataset import confirm_user_action
        
        # Storage checks
        if check_target_external_storage(logger, args.data_dir, override=getattr(args, 'allow_external_storage', False)):
            logger.error("Process terminated due to external storage detection.")
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
            test_images_per_class=args.test_per_class,
            balance_train=getattr(args, 'balance_train', False),
            max_train_per_class=getattr(args, 'max_train_per_class', None),
            run_id=getattr(args, 'run_id', None)
        )
        
        success = creator.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error in data create-dataset: {e}", exc_info=True)
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
            framework = run_args.framework.lower()
            if framework not in ("pytorch", "keras"):
                logger.error(f"Unsupported framework in JSON: {framework}")
                return 1
            model_type_str = pipeline.get("model.default_type", "image_classification")
            if model_type_str != "image_classification":
                logger.error(f"Unsupported model type from config: {model_type_str}")
                return 1
            model_type = ModelType.IMAGE_CLASSIFICATION
            data_dir = run_args.data_dir
            if not data_dir.exists():
                logger.error(f"Data directory does not exist: {data_dir}")
                return 1
            output_dir = run_args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            trainer = ModelTrainer(
                framework=framework,
                model_type=model_type,
                pipeline_config=pipeline,
            )
            supported_archs = trainer.get_supported_architectures()
            if run_args.architecture not in supported_archs:
                logger.error(
                    f"Architecture '{run_args.architecture}' not supported for framework '{framework}'"
                )
                logger.info(f"Supported architectures: {supported_archs}")
                return 1
            logger.info(f"Starting training from JSON ({framework}/{run_args.architecture})")
            model_path = trainer.train(run_args)
            logger.info(f"Training completed successfully. Model saved to: {model_path}")
            return 0
        
        # Determine framework
        framework = args.framework or pipeline.get('model.default_framework', 'pytorch')
        if framework not in ['pytorch', 'keras']:
            logger.error(f"Unsupported framework: {framework}")
            return 1
        
        # Determine model type
        model_type_str = args.model_type or pipeline.get('model.default_type', 'image_classification')
        if model_type_str == 'image_classification':
            model_type = ModelType.IMAGE_CLASSIFICATION
        else:
            logger.error(f"Unsupported model type: {model_type_str}")
            return 1
        
        # Determine architecture
        architecture = args.architecture or pipeline.get('model.default_architecture', 'resnet34')
        
        # Determine data directory
        data_dir = args.data_dir or Path(pipeline.get('data.data_dir', 'data'))
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
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
            framework=framework,
            model_type=model_type,
            pipeline_config=pipeline,
        )
        
        # Check if architecture is supported
        supported_archs = trainer.get_supported_architectures()
        if architecture not in supported_archs:
            logger.error(f"Architecture '{architecture}' not supported for framework '{framework}'")
            logger.info(f"Supported architectures: {supported_archs}")
            return 1
        
        # Train model
        logger.info(f"Starting training with {framework}/{architecture}")
        run_args = TrainingRunArgs(
            framework=framework,
            architecture=architecture,
            data_dir=data_dir,
            output_dir=output_dir,
            resume_from=args.resume_from,
            run_id=getattr(args, "run_id", None),
            update_snapshot=not getattr(args, "skip_snapshot_update", False),
            cli_hyperparams=dict(cli_hyperparams),
        )
        model_path = trainer.train(run_args)
        
        logger.info(f"Training completed successfully. Model saved to: {model_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=args.verbose)
        return 1


def handle_convert(args):
    """Handle 'mb convert' command."""
    try:
        from mb.conversion.converters import convert_model
        
        # Validate arguments
        if not args.input.exists():
            logger.error(f"Input model file not found: {args.input}")
            return 1
        
        # Check if architecture/num_classes are needed
        source_framework = args.framework
        if source_framework is None:
            from mb.conversion.converters import detect_model_framework
            source_framework = detect_model_framework(args.input)
            if source_framework is None:
                logger.error("Could not detect source framework. Please specify --framework")
                return 1
        
        if source_framework == 'pytorch' and args.target == 'onnx':
            if args.architecture is None or args.num_classes is None:
                logger.error(
                    "PyTorch -> ONNX conversion requires --architecture and --num-classes"
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
            logger.info(f"Conversion completed successfully: {args.output}")
            return 0
        else:
            logger.error("Conversion failed")
            return 1
            
    except Exception as e:
        logger.error(f"Conversion error: {e}", exc_info=args.verbose)
        return 1


def handle_info_model(args):
    """Handle 'mb info model' command."""
    logger.info("Info model command - not yet implemented")
    logger.info(f"Model path: {args.path}")
    # TODO: Implement
    return 0


def handle_info_dataset(args):
    """Handle 'mb info dataset' command."""
    logger.info("Info dataset command - not yet implemented")
    logger.info(f"Data dir: {args.data_dir}")
    # TODO: Implement
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
            if parsed_args.data_command == "gather":
                return handle_data_gather(parsed_args)
            elif parsed_args.data_command == "convert":
                return handle_data_convert(parsed_args)
            elif parsed_args.data_command == "deduplicate":
                return handle_data_deduplicate(parsed_args)
            elif parsed_args.data_command == "upscale":
                return handle_data_upscale(parsed_args)
            elif parsed_args.data_command == "create-dataset":
                return handle_data_create_dataset(parsed_args)
            else:
                logger.error("No data subcommand specified")
                return 1
        
        elif parsed_args.command == "train":
            return handle_train(parsed_args)
        
        elif parsed_args.command == "convert":
            return handle_convert(parsed_args)
        
        elif parsed_args.command == "info":
            if parsed_args.info_command == "model":
                return handle_info_model(parsed_args)
            elif parsed_args.info_command == "dataset":
                return handle_info_dataset(parsed_args)
            else:
                logger.error("No info subcommand specified")
                return 1
        
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=parsed_args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
