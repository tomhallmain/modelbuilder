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
    get_pipeline_config,
    reload_pipeline_config,
    resolve_create_dataset_cli,
)
from mb.utils.constants import ModelBuilderTaskType
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
    EvaluateSubcommand,
    ExportSubcommand,
    FrameworkType,
    InfoSubcommand,
    ModelBuildStepCommand,
    ModelType,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    from mb.cli.parsing import register_subparsers

    parser = argparse.ArgumentParser(
        prog="mb",
        description=_("Model Builder - A unified CLI for building ML models"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
    subparsers = parser.add_subparsers(
        dest="command",
        help=_("Available commands"),
        metavar="COMMAND",
    )
    register_subparsers(subparsers)
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
        from mb.data.dataset import confirm_user_action

        reload_pipeline_config(getattr(args, "config", None), force=True)
        ds_opts = resolve_create_dataset_cli(args)

        # Storage checks
        if check_target_external_storage(logger, args.data_dir, override=getattr(args, 'allow_external_storage', False)):
            logger.error(_("Process terminated due to external storage detection."))
            return 1
        
        # User confirmation for same drive case
        if check_same_drive(args.raw_data_dir, args.data_dir):
            if not confirm_user_action(logger, args):
                return 1
        
        # Set random seed from CLI or pipeline (resolve_create_dataset_cli)
        if ds_opts["seed"] is not None:
            import random

            random.seed(ds_opts["seed"])
            logger.info("Using random seed: %s", ds_opts["seed"])

        creator = DatasetCreator(
            raw_data_dir=args.raw_data_dir,
            data_dir=args.data_dir,
            test_per_class=ds_opts["test_per_class"],
            balance_train=getattr(args, 'balance_train', False),
            max_train_per_class=getattr(args, 'max_train_per_class', None),
            run_id=getattr(args, 'run_id', None),
            skip_space_check=getattr(args, "skip_space_check", False),
            test_split_mode=ds_opts["test_split_mode"],
            test_small_class_threshold=ds_opts["test_small_class_threshold"],
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
        if args.operation == ModelBuildStepCommand.CONVERT.value:
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

        # TODO: Read an RNG seed from pipeline config (e.g. a dedicated ``training.seed`` or
        # ``model.seed``; align whether ``data.seed`` should apply here too) and set
        # ``random`` / ``numpy`` / ``torch`` (and CUDA deterministic flags as needed) before
        # ``ModelTrainer.train`` so runs are reproducible. Wire the same hook for the
        # ``train_args_json`` branch below.

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


def handle_export_bundle(args) -> int:
    """Handle 'mb export bundle' command."""
    try:
        from mb.export.bundle import export_bundle

        if not args.input.exists():
            logger.error(_("Input model file not found: {path}").format(path=args.input))
            return 1

        reload_pipeline_config(getattr(args, "config", None), force=True)
        pipeline = get_pipeline_config()

        result = export_bundle(
            input_model=args.input,
            output_dir=args.output_dir,
            architecture=args.architecture,
            num_classes=args.num_classes,
            class_names=list(args.class_names) if args.class_names else None,
            data_dir=args.data_dir,
            image_size=args.image_size,
            include_architecture_py=not bool(args.no_architecture_py),
            pipeline_config=pipeline.to_dict(),
            snapshot_path=args.snapshot,
            run_id=args.run_id,
        )
        logger.info(_("Bundle export completed: {path}").format(path=args.output_dir))
        logger.info(_("Weights: {path}").format(path=result["weights_path"]))
        logger.info(_("Manifest: {path}").format(path=result["manifest_path"]))
        if result.get("architecture_path"):
            logger.info(_("Architecture stub: {path}").format(path=result["architecture_path"]))
        return 0
    except Exception as e:
        logger.error(_("Export failed: {err}").format(err=e), exc_info=args.verbose)
        return 1


def handle_evaluate_metrics(args) -> int:
    """Handle ``mb evaluate metrics``."""
    reload_pipeline_config(getattr(args, "config", None), force=True)
    from mb.evaluate.metrics import run_evaluate_metrics_cli

    return run_evaluate_metrics_cli(args)


def handle_evaluate_misclassified(args) -> int:
    """Handle ``mb evaluate misclassified``."""
    reload_pipeline_config(getattr(args, "config", None), force=True)
    from mb.evaluate.misclassified import run_evaluate_misclassified_cli

    return run_evaluate_misclassified_cli(args)


def handle_evaluate_compare(args) -> int:
    """Handle ``mb evaluate compare``."""
    reload_pipeline_config(getattr(args, "config", None), force=True)
    from mb.evaluate.compare import run_evaluate_compare_cli

    return run_evaluate_compare_cli(args)


_EVALUATE_HANDLERS = {
    EvaluateSubcommand.METRICS: handle_evaluate_metrics,
    EvaluateSubcommand.MISCLASSIFIED: handle_evaluate_misclassified,
    EvaluateSubcommand.COMPARE: handle_evaluate_compare,
}


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
        if parsed_args.command == ModelBuilderTaskType.DATA.value:
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
        
        elif parsed_args.command == ModelBuilderTaskType.TRAIN.value:
            return handle_train(parsed_args)
        
        elif parsed_args.command == ModelBuilderTaskType.CONVERT.value:
            return handle_convert(parsed_args)
        
        elif parsed_args.command == ModelBuilderTaskType.INFO.value:
            raw_info = parsed_args.info_command
            if not raw_info:
                logger.error(_("No info subcommand specified"))
                return 1
            info_sub = InfoSubcommand.try_from(raw_info)
            if info_sub is None:
                logger.error(_("Unknown info subcommand: {cmd}").format(cmd=raw_info))
                return 1
            if info_sub == InfoSubcommand.MODEL:
                return handle_info_model(parsed_args)
            return handle_info_dataset(parsed_args)

        elif parsed_args.command == ModelBuilderTaskType.EXPORT.value:
            raw_export = parsed_args.export_command
            if not raw_export:
                logger.error(_("No export subcommand specified"))
                return 1
            sub = ExportSubcommand.try_from(raw_export)
            if sub is None:
                logger.error(_("Unknown export subcommand: {cmd}").format(cmd=raw_export))
                return 1
            if sub == ExportSubcommand.BUNDLE:
                return handle_export_bundle(parsed_args)
            logger.error(_("Unhandled export subcommand: {cmd}").format(cmd=sub.value))
            return 1

        elif parsed_args.command == ModelBuilderTaskType.EVALUATE.value:
            raw_ev = parsed_args.evaluate_command
            if not raw_ev:
                logger.error(_("No evaluate subcommand specified"))
                return 1
            ev_sub = EvaluateSubcommand.try_from(raw_ev)
            if ev_sub is None:
                logger.error(_("Unknown evaluate subcommand: {cmd}").format(cmd=raw_ev))
                return 1
            return _EVALUATE_HANDLERS[ev_sub](parsed_args)

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

    *subcommand* must match a :class:`~mb.models.types.ModelBuildStepCommand` ``.value``
    (for example :attr:`~mb.models.types.ModelBuildStepCommand.GATHER`). *argv* defaults to ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]
    return main([ModelBuilderTaskType.DATA.value, subcommand, *argv])


if __name__ == "__main__":
    sys.exit(main())
