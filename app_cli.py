"""Entry script for the compiled CLI (``ModelBuilderHeadless``); Nuitka needs a
file to compile. Same as ``python -m mb.cli`` / ``mb``, plus ``--smoke-test
[--expect-oqs]`` (handled before the CLI parses its arguments, so required ones do
not block it) and a fault log for native crashes. The headless build excludes
PySide6, so nothing reached from here may import Qt."""

import multiprocessing
import sys


def _smoke_test(expect_oqs: bool) -> int:
    # Before any other app import: modules resolve their paths as they load.
    from mb.utils.smoke_test import isolate_environment, run_smoke_test

    scratch = isolate_environment()

    from mb.utils.crash_log import enable_fault_log

    enable_fault_log()
    return run_smoke_test(scratch, expect_oqs)


def main() -> int:
    from mb.utils.smoke_test import smoke_test_options

    argv = sys.argv[1:]
    smoke_test, expect_oqs = smoke_test_options(argv)
    if smoke_test:
        return _smoke_test(expect_oqs)
    from mb.utils.crash_log import enable_fault_log

    enable_fault_log()
    from mb.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    # A DataLoader worker started from the executable runs the worker, not the CLI.
    multiprocessing.freeze_support()
    sys.exit(main())
