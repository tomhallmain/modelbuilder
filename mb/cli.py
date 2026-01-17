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
from mb.config import get_config
from mb.utils.logging import setup_logging

# Import data processing modules
from mb.data.gather import ImageGatherer
from mb.data.convert import ImageConverter
from mb.data.deduplicate import ImageDeduplicator
from mb.data.upscale import ImageUpscaler
from mb.data.dataset import DatasetCreator

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
        help="Data processing operations"
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        help="Data subcommands"
    )
    
    # mb data gather
    gather_parser = data_subparsers.add_parser(
        "gather",
        help="Gather images from source directories"
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
        help="Convert images to specified format"
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
        help="Remove duplicate images"
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
        help="Upscale small images"
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
        help="Create train/test dataset splits"
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
        help="Train a model"
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
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    # Convert command
    convert_model_parser = subparsers.add_parser(
        "convert",
        help="Convert model between formats"
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
        help="Source framework"
    )
    convert_model_parser.add_argument(
        "--target",
        choices=["pytorch", "keras", "h5", "onnx"],
        help="Target format"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about models or datasets"
    )
    info_subparsers = info_parser.add_subparsers(
        dest="info_command",
        help="Info subcommands"
    )
    
    # mb info model
    info_model_parser = info_subparsers.add_parser(
        "model",
        help="Show model information"
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
        help="Show dataset information"
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
    logger.info("Train command - not yet implemented")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Framework: {args.framework}")
    logger.info(f"Architecture: {args.architecture}")
    # TODO: Implement in Phase 3-4
    return 0


def handle_convert(args):
    """Handle 'mb convert' command."""
    logger.info("Convert command - not yet implemented")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    # TODO: Implement in Phase 4
    return 0


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
    
    # Load configuration if provided
    config = get_config(parsed_args.config)
    
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
