"""
End-to-end ``char_ngram_linear`` run on a tiny synthetic dataset (needs scikit-learn).

Covers the run-directory contract: every output file, a reproducible ``metrics.json``,
line-preserving scoring, and no conflict-group or cross-split rows in predictions.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mb.training.text_config import TEXT_CLASSIFICATION_DEFAULTS, parse_text_run_config

from tests.fixtures.text_dataset import write_text_dataset

RUN_FILES = (
    "config.yaml",
    "environment.json",
    "model/backend.json",
    "calibrator.json",
    "thresholds.json",
    "metrics.json",
    "predictions_test.tsv",
    "review_queue.tsv",
    "cut_rescore.tsv",
    "extra_reports.json",
    "MODEL_CARD.md",
    "predict.py",
)


def _config(data_dir: Path, runs_dir: Path):
    raw = copy.deepcopy(TEXT_CLASSIFICATION_DEFAULTS)
    raw.update(
        {
            "data_dir": str(data_dir),
            "runs_dir": str(runs_dir),
            "backend_options": {"min_df": 1, "ngram_max": 4},
            "model_card_notes": "Synthetic test data.",
        }
    )
    return parse_text_run_config(raw)


@pytest.fixture
def char_ngram_runs(tmp_path: Path):
    pytest.importorskip("sklearn", reason="char_ngram_linear needs scikit-learn")
    pytest.importorskip("scipy", reason="char_ngram_linear needs scipy")
    from mb.training.text_trainer import train_text_classifier

    data_dir = write_text_dataset(tmp_path / "ds")
    config = _config(data_dir, tmp_path / "runs")
    first = train_text_classifier(config)
    second = train_text_classifier(config)
    return data_dir, first, second


@pytest.mark.slow
def test_run_directory_has_every_output(char_ngram_runs) -> None:
    _data, run, _ = char_ngram_runs
    for name in RUN_FILES:
        assert (run / name).is_file(), name


@pytest.mark.slow
def test_second_run_reproduces_metrics_exactly(char_ngram_runs) -> None:
    _data, first, second = char_ngram_runs
    assert first != second
    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()


@pytest.mark.slow
def test_reevaluation_reproduces_metrics(char_ngram_runs) -> None:
    from mb.evaluate.classification.text_evaluation import evaluate_text_run

    _data, run, _ = char_ngram_runs
    before = (run / "metrics.json").read_bytes()
    evaluate_text_run(run)
    assert (run / "metrics.json").read_bytes() == before


@pytest.mark.slow
def test_predictions_hold_only_test_rows_without_conflicts(char_ngram_runs) -> None:
    from mb.data.text_dataset import load_text_dataset

    data_dir, run, _ = char_ngram_runs
    ds = load_text_dataset(data_dir)
    test_texts = {ds.texts[i] for i in ds.split_indices("test", exclude_conflicts=True)}
    lines = (run / "predictions_test.tsv").read_text(encoding="utf-8").split("\n")[1:-1]
    predicted = [line.split("\t")[0] for line in lines]
    assert set(predicted) == test_texts
    assert "Afstab" not in predicted and "A_F_S_T_A_B" not in predicted


@pytest.mark.slow
def test_score_keeps_line_count_and_order(char_ngram_runs, tmp_path: Path) -> None:
    from mb.evaluate.classification.text_evaluation import score_text_file

    _data, run, _ = char_ngram_runs
    lines = ["zo blood", "", "line\u2028sep", "NA", "kalo"]
    src = tmp_path / "lines.txt"
    src.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    out = tmp_path / "scores.tsv"
    result = score_text_file(run, src, out)
    rows = out.read_text(encoding="utf-8").split("\n")[:-1]
    assert result.n_lines == len(lines)
    assert rows[0] == "term\tscore"
    assert [r.rsplit("\t", 1)[0] for r in rows[1:]] == lines


@pytest.mark.slow
def test_predict_script_matches_score_without_mb(char_ngram_runs, tmp_path: Path) -> None:
    from mb.evaluate.classification.text_evaluation import score_text_file

    _data, run, _ = char_ngram_runs
    lines = ["zo blood", "", "line\u2028sep", "NA", "kalo"]
    src = tmp_path / "lines.txt"
    src.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    expected_out = tmp_path / "scores.tsv"
    score_text_file(run, src, expected_out)
    expected = [r.rsplit("\t", 1)[1] for r in expected_out.read_text(encoding="utf-8").split("\n")[1:-1]]

    code = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import predict\n"
        "clf = predict.load()\n"
        "lines = predict._read_lines(sys.argv[2])\n"
        "print(json.dumps({'scores': [f'{s:.6f}' for s in clf.score(lines)],"
        " 'flags': [bool(f) for f in clf.flag(lines)], 'threshold': clf.threshold,"
        " 'mb_imported': any(m == 'mb' or m.startswith('mb.') for m in sys.modules)}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(run), str(src)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    got = json.loads(proc.stdout)
    assert got["scores"] == expected
    assert got["flags"] == [float(s) >= got["threshold"] for s in got["scores"]]
    assert got["mb_imported"] is False
