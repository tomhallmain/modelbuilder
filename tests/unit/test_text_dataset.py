"""Tests for ``mb.data.text_dataset`` (reading, verification, selection, weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mb.data.text_dataset import (
    TextDatasetError,
    compute_sample_weights,
    gold_mask,
    load_text_dataset,
    read_text_lines,
    select_training_indices,
    verify_text_dataset,
)
from mb.utils.translations import _

from tests.fixtures.text_dataset import AWKWARD_TEXTS, split_counts, write_text_dataset


@pytest.fixture
def text_data_dir(tmp_path: Path) -> Path:
    return write_text_dataset(tmp_path / "ds")


def test_load_keeps_awkward_texts_verbatim(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    for t in AWKWARD_TEXTS:
        assert t in ds.texts
    train, val, test = split_counts(text_data_dir)
    assert len(ds) == train + val + test


def test_conflict_rows_marked(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    marked = {ds.texts[i] for i in np.flatnonzero(ds.conflict)}
    assert marked == {"Afstab", "A_F_S_T_A_B"}


def test_verify_passes_on_intact_dataset(text_data_dir: Path) -> None:
    report = verify_text_dataset(text_data_dir)
    assert report.ok, report.format()
    assert "dataset.tsv" in report.file_sha256


def test_verify_fails_after_one_byte_changes(text_data_dir: Path) -> None:
    path = text_data_dir / "dataset.tsv"
    data = bytearray(path.read_bytes())
    i = data.index(b"\n") + 1
    data[i] = ord("Z") if data[i] != ord("Z") else ord("Y")
    path.write_bytes(bytes(data))
    report = verify_text_dataset(text_data_dir)
    assert not report.ok
    assert any(c.name == "sha256 dataset.tsv" and not c.ok for c in report.checks)


def test_verify_without_manifest_checks_uniqueness_and_groups(tmp_path: Path) -> None:
    d = write_text_dataset(tmp_path / "ds", manifest=False)
    lines = (d / "dataset.tsv").read_text(encoding="utf-8").split("\n")
    fields = lines[1].split("\t")
    other_split = "val" if fields[-1] != "val" else "test"
    lines.append("\t".join(fields[:-1] + [other_split]))
    (d / "dataset.tsv").write_text(
        "\n".join(l for l in lines if l) + "\n", encoding="utf-8", newline="\n"
    )
    report = verify_text_dataset(d)
    failed = {c.name for c in report.checks if not c.ok}
    assert failed == {_("text values unique"), _("each group in one split")}


def test_load_rejects_bad_label(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "dataset.tsv").write_text("text\tlabel\tsplit\nfoo\t2\ttrain\n", encoding="utf-8")
    with pytest.raises(TextDatasetError):
        load_text_dataset(d)


def test_read_text_lines_splits_on_lf_only(tmp_path: Path) -> None:
    p = tmp_path / "in.txt"
    p.write_bytes("a\u2028b\n\nc\n".encode("utf-8"))
    assert read_text_lines(p) == ["a\u2028b", "", "c"]


def test_crlf_line_endings_read_like_lf(tmp_path: Path) -> None:
    lines_path = tmp_path / "in.txt"
    lines_path.write_bytes(b"a\r\n\r\nb\r\n")
    assert read_text_lines(lines_path) == ["a", "", "b"]
    d = tmp_path / "ds"
    d.mkdir()
    (d / "dataset.tsv").write_bytes(b"text\tlabel\tsplit\r\nfoo\t1\ttrain\r\nbar\t0\tval\r\n")
    ds = load_text_dataset(d)
    assert ds.texts == ["foo", "bar"]
    assert ds.split.value(1) == "val"


def test_training_selection_excludes_conflicts(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    idx, info = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=None, seed=1)
    assert info is None
    assert not ds.conflict[idx].any()
    assert set(ds.split.value(i) for i in idx) == {"train"}


def test_subsample_keeps_positives_and_is_seeded(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    a, info = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=0.5, seed=3)
    b, _ = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=0.5, seed=3)
    full, _ = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=None, seed=3)
    assert np.array_equal(a, b)
    assert ds.labels[a].sum() == ds.labels[full].sum()
    assert info is not None and info.kept == min(info.available, round(0.5 * int(ds.labels[full].sum())))


def test_balanced_weights_equalize_classes(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    idx, _ = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=None, seed=0)
    w, _cw = compute_sample_weights(ds, idx, class_weight="balanced", tier_weight={})
    labels = ds.labels[idx]
    assert w[labels == 1].sum() == pytest.approx(w[labels == 0].sum(), rel=1e-5)


def test_tier_weight_multiplies_and_rejects_unknown_tiers(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    idx, _ = select_training_indices(ds, exclude_conflicts=True, subsample_keep_unreviewed=None, seed=0)
    w, _ = compute_sample_weights(ds, idx, class_weight="none", tier_weight={"keep_unreviewed": 0.5})
    tiers = np.array([ds.tier_of(i) for i in idx])
    assert set(w[tiers == "keep_unreviewed"].tolist()) == {0.5}
    assert set(w[tiers != "keep_unreviewed"].tolist()) == {1.0}
    with pytest.raises(TextDatasetError):
        compute_sample_weights(ds, idx, class_weight="none", tier_weight={"keep_unreviwed": 0.5})


def test_gold_mask_uses_tiers(text_data_dir: Path) -> None:
    ds = load_text_dataset(text_data_dir)
    idx = ds.split_indices("test", exclude_conflicts=True)
    gold = gold_mask(ds, idx, ("reject_reviewed", "keep_reviewed"))
    assert {ds.tier_of(i) for i in idx[gold]} <= {"reject_reviewed", "keep_reviewed"}
