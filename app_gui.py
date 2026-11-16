"""Entry script for the compiled GUI (``ModelBuilder``); Nuitka needs a file to
compile. Same as ``python -m ui`` / ``mb-gui``, plus ``--smoke-test [--expect-oqs]``
and a fault log for native crashes."""

import multiprocessing
import sys


def _smoke_test(expect_oqs: bool) -> int:
    # Before any other app import: modules resolve their paths as they load.
    from mb.utils.smoke_test import isolate_environment, run_smoke_test

    scratch = isolate_environment()

    from mb.utils.crash_log import enable_fault_log
    from ui.qt_message_log import install_qt_message_handler

    enable_fault_log()
    install_qt_message_handler()
    from ui.main_window import MainWindow

    return run_smoke_test(scratch, expect_oqs, main_window_class=MainWindow)


def main() -> int:
    from mb.utils.smoke_test import smoke_test_options

    argv = sys.argv[1:]
    smoke_test, expect_oqs = smoke_test_options(argv)
    if smoke_test:
        return _smoke_test(expect_oqs)
    from mb.utils.crash_log import enable_fault_log

    enable_fault_log()
    from ui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    # A DataLoader worker started from the executable runs the worker, not the app.
    multiprocessing.freeze_support()
    sys.exit(main())
