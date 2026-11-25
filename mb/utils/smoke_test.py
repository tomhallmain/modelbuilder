"""``--smoke-test [--expect-oqs]``: checks a built executable works, without a window.

Both entry scripts (``app_gui.py``, ``app_cli.py``) handle the option before any
other app import, because loggers, the app data dir, the cache and the key store
resolve their paths from the environment as modules load. ``isolate_environment()``
points every per-user location at a scratch directory and gives the run its own
key store identity with a passphrase from the environment, so the run touches
neither the user's data, the OS keyring, nor an external drive; a separate
identity also keeps the key store migration from moving (and deleting) keychain
items of the real install. Imports nothing from the app at load time:
build_exe.py runs this file on its own to build the same environment.

The checks' log lines are build diagnostics, read by whoever builds, and stay in
English.
"""

#: Key store identity for the run; never one a real install uses.
SMOKE_SERVICE_NAME = "ModelBuilderSmokeTest"
SMOKE_APP_IDENTIFIER = "smoke_test"
#: Set to keep the scratch directory at a known place (build_exe.py does).
SCRATCH_DIR_ENV = "MODELBUILDER_SMOKE_DIR"
#: Variables the test suite sets that would change what the run checks.
_TEST_SUITE_VARIABLES = ("MODELBUILDER_TEST_APP_DATA", "MODELBUILDER_TEST_CACHE_PATH")


def passphrase_env_var(service_name: str) -> str:
    """The variable the encryptor reads a service's passphrase from."""
    return f"{service_name.upper()}_PASSPHRASE"


def smoke_test_environment(scratch: str) -> dict:
    """Every variable the run sets, all pointing into *scratch*. Assigned over the
    inherited values, so a variable exported in the developer's shell cannot reach
    the real locations."""
    import os
    import secrets

    home = os.path.join(scratch, "home")
    return {
        # App data dir (get_app_data_dir), the key store root (user_data_dir) and
        # the Hugging Face / torch caches all derive from these.
        "HOME": home,
        "USERPROFILE": home,
        "APPDATA": os.path.join(scratch, "appdata"),
        "LOCALAPPDATA": os.path.join(scratch, "localappdata"),
        "XDG_DATA_HOME": os.path.join(scratch, "data"),
        "XDG_CACHE_HOME": os.path.join(scratch, "cache"),
        # Documents comes from the Windows shell API, not the environment.
        "MODELBUILDER_WORK_DIR": os.path.join(scratch, "work"),
        "MODELBUILDER_KEY_STORE_DIR": os.path.join(scratch, "keys"),
        "MODELBUILDER_KEY_BACKUP_DIR": os.path.join(scratch, "key_backup"),
        # The real AppInfoCache, not the test suite's stand-in.
        "MODELBUILDER_TEST_CACHE": "0",
        # GUI settings in an INI file instead of the registry / plist.
        "MODELBUILDER_SETTINGS_INI": os.path.join(scratch, "gui_settings.ini"),
        passphrase_env_var(SMOKE_SERVICE_NAME): secrets.token_hex(16),
        "QT_QPA_PLATFORM": "offscreen",
        SCRATCH_DIR_ENV: scratch,
    }


def isolate_environment() -> str:
    """Apply smoke_test_environment() to this process and switch AppInfo to the
    smoke identity; returns the scratch directory. SCRATCH_DIR_ENV names it,
    otherwise a new temp directory is used."""
    import os
    import tempfile

    scratch = os.environ.get(SCRATCH_DIR_ENV) or tempfile.mkdtemp(prefix="modelbuilder-smoke-")
    environment = smoke_test_environment(scratch)
    for name in _TEST_SUITE_VARIABLES:
        os.environ.pop(name, None)
    os.environ.update(environment)
    for name in ("HOME", "APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                 "MODELBUILDER_WORK_DIR", "MODELBUILDER_KEY_STORE_DIR", "MODELBUILDER_KEY_BACKUP_DIR"):
        os.makedirs(environment[name], exist_ok=True)

    # Only after the environment is in place: this import loads the logging setup.
    from mb.utils.constants import AppInfo

    AppInfo.SERVICE_NAME = SMOKE_SERVICE_NAME
    AppInfo.APP_IDENTIFIER = SMOKE_APP_IDENTIFIER
    return scratch


def smoke_test_options(argv) -> tuple:
    """(run the smoke test, expect OQS) from a command line."""
    return "--smoke-test" in argv, "--expect-oqs" in argv


class SmokeTest:
    """The checks, each logged on one line and run under guarded(), so one failure
    still lets the others report."""

    def __init__(self, scratch: str):
        from mb.utils.logging_setup import get_logger

        self.scratch = scratch
        self.failures = []
        self._app = None
        self.logger = get_logger("smoke_test")

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.logger.info(f"smoke test: {name}: {'ok' if ok else 'FAILED'} {detail}".rstrip())
        if not ok:
            self.failures.append(name)

    def note(self, name: str, detail: str) -> None:
        """A line that reports without passing or failing."""
        self.logger.info(f"smoke test: {name}: {detail}")

    def guarded(self, name: str, func) -> None:
        """Run the check *func*; anything it raises, SystemExit included (a bundled
        package without metadata exits the process on a version probe), fails
        *name* with its full traceback."""
        import traceback

        try:
            func()
        except BaseException:
            self.logger.error(f"smoke test: {name}: FAILED\n{traceback.format_exc()}")
            self.failures.append(name)

    def result(self) -> int:
        """Exit code: 0 if every check passed."""
        if self.failures:
            self.logger.error(f"smoke test failed: {', '.join(self.failures)}")
            return 1
        self.logger.info("smoke test passed")
        return 0

    def _work_dir(self, name: str) -> str:
        import os

        path = os.path.join(self.scratch, name)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Qt-free checks (both entry points)
    # ------------------------------------------------------------------

    def check_bundled_files(self) -> None:
        import glob
        import os

        from mb.utils.app_paths import IS_BUILD, resource_path
        from mb.utils.translations import I18N
        from mb.utils.version import VERSION_FILE, app_version, build_info

        catalogs = glob.glob(os.path.join(I18N.localedir, "*", "LC_MESSAGES", "base.mo"))
        self.check("locale", bool(catalogs), f"{len(catalogs)} catalog(s) in {I18N.localedir}")
        for parts in (("mb", "config", "default_pipeline.yaml"),
                      ("mb", "config", "application.example.yaml"),
                      ("mb", "config", "text_classification.example.yaml"),
                      ("mb", "config", "image_classification_convnext.example.yaml"),
                      ("mb", "config", "image_classification_efficientnet.example.yaml"),
                      ("mb", "config", "image_classification_focalnet.example.yaml"),
                      ("mb", "config", "image_classification_mobilenet.example.yaml"),
                      ("mb", "config", "image_classification_resnet.example.yaml"),
                      ("mb", "config", "image_classification_siglip2_base.example.yaml"),
                      ("mb", "config", "image_classification_siglip2_so400m.example.yaml"),
                      ("mb", "config", "image_classification_vit.example.yaml")):
            self.check("/".join(parts), resource_path(*parts).is_file())
        self.check("VERSION", resource_path(VERSION_FILE).is_file(), app_version())
        if IS_BUILD:
            recorded = build_info().get("version")
            self.check("build info", recorded == app_version(), f"records version {recorded}")
        else:
            self.note("build info", "not checked in a source checkout")

    def check_translations(self) -> None:
        import gettext

        from mb.utils.translations import I18N

        german = gettext.translation("base", I18N.localedir, languages=["de"])
        text = german.gettext("Model Builder")
        self.check("translations (de)", text != "Model Builder", text)

    def check_default_configs(self) -> None:
        from mb.pipeline_config import default_pipeline_yaml_dict, get_pipeline_config
        from utils.config import get_application_config

        defaults = default_pipeline_yaml_dict()
        self.check("pipeline defaults", "training" in defaults and "data" in defaults,
                   ", ".join(sorted(defaults)))
        pipeline = get_pipeline_config()
        self.check("pipeline config", pipeline.get("training.num_workers") is not None,
                   f"base dir {pipeline.base_dir}")
        app_config = get_application_config()
        self.check("application config", app_config.gui is not None)

    def check_keyring_backend(self) -> None:
        import keyring
        import keyring.backends.fail

        backend = keyring.get_keyring()
        self.check(
            "keyring backend",
            not isinstance(backend, keyring.backends.fail.Keyring),
            f"{type(backend).__module__}.{type(backend).__name__}",
        )

    def check_encrypted_cache(self) -> None:
        """Keys generated and the cache stored and read back, under the smoke identity."""
        from mb.utils.constants import AppInfo
        from utils.app_info_cache import AppInfoCache
        from utils.encryptor import get_encryptor, read_key_store

        service, app_id = AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER
        if service != SMOKE_SERVICE_NAME:
            raise RuntimeError(f"Refusing to create keys outside the smoke identity: {service}:{app_id}")
        # The encryptor counts an environment passphrase as keys having existed, so
        # it refuses to generate them for an empty key store; force_new is its
        # deliberate way to start one. Only with no store, so nothing is purged.
        if read_key_store(service, app_id) is None:
            get_encryptor(service, app_id).generate_and_store_keys(service, app_id, force_new=True)

        cache = AppInfoCache()
        cache.set("smoke_test", "round trip")
        encrypted = cache.store()
        reloaded = AppInfoCache()
        self.check(
            "encrypted cache",
            encrypted and reloaded.get("smoke_test") == "round trip",
            cache.cache_loc if encrypted else "stored as plain JSON, encryption failed",
        )

    def check_oqs(self, expect_oqs: bool) -> None:
        from utils import encryptor

        available = encryptor.KeyEncapsulation is not None
        if expect_oqs:
            self.check("OQS key encapsulation", available)
        else:
            self.note("OQS key encapsulation", "available" if available else "not in this build")

    def check_fault_log(self) -> None:
        import os

        from mb.utils.crash_log import enable_fault_log

        path = enable_fault_log()
        self.check("fault log", os.path.isfile(path), path)

    def check_version_line(self) -> None:
        from mb.utils.version import describe

        self.note("version", describe())

    def check_modules(self) -> None:
        """Modules mb imports only on first use, so startup never loads them."""
        import importlib.util

        for name in ("torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "safetensors",
                     "transformers", "sklearn", "scipy", "imageio_ffmpeg"):
            self.check(f"module {name}", importlib.util.find_spec(name) is not None)

    def check_transformers(self) -> None:
        """transformers probes optional packages' versions as it loads; a bundled one
        without metadata ends the process there (caught by guarded())."""
        import transformers

        config_class = transformers.AutoConfig
        self.check("transformers", config_class is not None, transformers.__version__)

    def check_text_backends(self) -> None:
        from mb.models.text_backends import get_backend_class
        from mb.models.types import TextBackendType

        for backend in TextBackendType:
            cls = get_backend_class(backend)
            self.check(f"text backend {backend.value}", cls.name == backend)

    def check_architectures(self) -> None:
        import mb.models.frameworks.pytorch  # noqa: F401  (registers the architectures)
        from mb.models.frameworks.registry import list_architectures
        from mb.models.types import FrameworkType

        names = list_architectures(FrameworkType.PYTORCH).get(FrameworkType.PYTORCH.value, [])
        siglip = [n for n in names if str(n).startswith("siglip2")]
        self.check("pytorch architectures", bool(names), f"{len(names)} registered, {len(siglip)} SigLIP2")
        self.check("SigLIP2 architectures", bool(siglip))

    def check_torch_onnx(self) -> None:
        """A small model run forward, exported with mb's ONNX exporter, and run again
        with onnxruntime: the exporter reads bytecode or traces, which a build may not
        support."""
        import os

        import numpy as np
        import onnxruntime
        import torch

        from mb.conversion.converters import export_torch_module_to_onnx

        self.note("torch", f"{torch.__version__}, CUDA {torch.version.cuda or 'none'}, "
                           f"GPU {'available' if torch.cuda.is_available() else 'not available'}")
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(4, 2),
        ).eval()
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            expected = model(image).numpy()
        self.check("torch forward", expected.shape == (1, 2))

        path = os.path.join(self._work_dir("onnx"), "smoke.onnx")
        export_torch_module_to_onnx(model, path, image_size=32)
        session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: image.numpy()})[0]
        self.check("ONNX export and inference", np.allclose(actual, expected, atol=1e-4),
                   f"max difference {float(np.abs(actual - expected).max()):.2e}")

    def check_data_loader_worker(self) -> None:
        """A DataLoader with one worker process: on Windows multiprocessing starts it
        from sys.executable, in a build the executable itself."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(torch.arange(8, dtype=torch.float32)),
                            batch_size=4, num_workers=1, timeout=120)
        total = sum(float(batch[0].sum()) for batch in loader)
        self.check("DataLoader worker", total == 28.0, f"sum {total}")

    def check_text_classifier(self) -> None:
        """char_ngram_linear trained on a few rows, saved, loaded and scored, and its
        run's predict.py generated."""
        from pathlib import Path

        import numpy as np

        from mb.evaluate.classification.text_predict_script import predict_script_source
        from mb.models.text_backends.base import TextFitContext, TextFitData
        from mb.models.text_backends.char_ngram_linear import CharNgramLinearBackend

        positive = ["great product", "really great", "great value", "great service",
                    "simply great", "great quality"]
        negative = ["awful product", "really awful", "awful value", "awful service",
                    "simply awful", "awful quality"]
        texts = positive + negative
        data = TextFitData(texts=texts, labels=np.array([1] * 6 + [0] * 6),
                           weights=np.ones(len(texts), dtype=np.float64))
        run_dir = self._work_dir("text_run")
        ctx = TextFitContext(
            model_id=None, seed=0, max_length=64, lr=None, epochs=None, batch_size=None,
            warmup_ratio=0.0, weight_decay=0.0, patience=0,
            options=CharNgramLinearBackend.resolve_options({}), device="cpu",
            run_dir=Path(run_dir), selection_metric=lambda probs: float(probs.mean()),
        )
        backend = CharNgramLinearBackend()
        backend.fit(data, data, ctx)
        probs = backend.predict_proba(texts)
        self.check("text classifier fit", float(probs[:6].mean()) > float(probs[6:].mean()),
                   f"mean positive {float(probs[:6].mean()):.3f}, negative {float(probs[6:].mean()):.3f}")

        model_dir = Path(self._work_dir("text_model"))
        backend.save_weights(model_dir)
        reloaded = CharNgramLinearBackend.load_weights(model_dir, backend.metadata(), "cpu")
        self.check("text classifier reload", np.allclose(reloaded.predict_proba(texts), probs))

        source = predict_script_source(CharNgramLinearBackend)
        compile(source, "predict.py", "exec")
        self.check("predict.py", True, f"{len(source)} characters")

    def check_ffmpeg(self) -> None:
        """The ffmpeg binary imageio-ffmpeg ships, started as mb data convert starts it."""
        import subprocess
        import sys

        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        result = subprocess.run([exe, "-version"], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=60, creationflags=flags)
        first_line = (result.stdout.splitlines() or [""])[0]
        self.check("ffmpeg", result.returncode == 0, f"{exe}: {first_line}")

    # ------------------------------------------------------------------
    # Entry-point specific checks
    # ------------------------------------------------------------------

    def _qt_app(self):
        """The QApplication, created on first use and kept for the rest of the run:
        one nothing references can be destroyed, and a later event loop then runs
        without it."""
        from PySide6.QtWidgets import QApplication

        if self._app is None:
            self._app = QApplication.instance() or QApplication([])
        return self._app

    def check_qt(self, main_window_class) -> None:
        """Qt's messages reaching the log, and the main window created and deleted
        offscreen. *main_window_class* comes from the entry point: in a build the
        entry script is the compiled __main__ and may not be importable by name."""
        from PySide6.QtCore import QCoreApplication, QEvent

        from ui.qt_message_log import qt_message_reaches_logging

        app = self._qt_app()
        self.check("Qt messages logged", qt_message_reaches_logging())
        window = main_window_class()
        self.check("main window", True, window.windowTitle())
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

    def check_cli(self) -> None:
        from mb.cli import main

        try:
            code = main(["--version"])
        except SystemExit as e:  # argparse's version action exits
            code = e.code
        self.check("mb --version", code in (0, None))

    def check_no_qt(self) -> None:
        """The headless build excludes PySide6; an import reachable from its code
        would fail there only when that path runs."""
        import sys

        loaded = sorted(name for name in sys.modules if name.split(".")[0] == "PySide6")
        self.check("no Qt loaded", not loaded, ", ".join(loaded[:5]))


def run_smoke_test(scratch: str, expect_oqs: bool = False, main_window_class=None) -> int:
    """Run every check; returns the exit code. With *main_window_class* (the GUI
    entry point) the Qt checks run, without it the headless ones."""
    smoke = SmokeTest(scratch)
    smoke.note("scratch directory", scratch)
    smoke.guarded("version", smoke.check_version_line)
    smoke.guarded("bundled files", smoke.check_bundled_files)
    smoke.guarded("translations", smoke.check_translations)
    smoke.guarded("default configs", smoke.check_default_configs)
    smoke.guarded("keyring backend", smoke.check_keyring_backend)
    smoke.guarded("encrypted cache", smoke.check_encrypted_cache)
    smoke.guarded("OQS key encapsulation", lambda: smoke.check_oqs(expect_oqs))
    smoke.guarded("fault log", smoke.check_fault_log)
    smoke.guarded("modules", smoke.check_modules)
    smoke.guarded("transformers", smoke.check_transformers)
    smoke.guarded("text backends", smoke.check_text_backends)
    smoke.guarded("pytorch architectures", smoke.check_architectures)
    smoke.guarded("torch and ONNX", smoke.check_torch_onnx)
    smoke.guarded("DataLoader worker", smoke.check_data_loader_worker)
    smoke.guarded("text classifier", smoke.check_text_classifier)
    smoke.guarded("ffmpeg", smoke.check_ffmpeg)
    if main_window_class is not None:
        smoke.guarded("Qt", lambda: smoke.check_qt(main_window_class))
    else:
        smoke.guarded("mb --version", smoke.check_cli)
        smoke.guarded("no Qt loaded", smoke.check_no_qt)
    return smoke.result()
