"""
Multi-label dataset contract: schema parsing, manifest reading, and label resolution.

The property that matters most is index stability. Output index order has to survive adding
a label later, because a shipped model's weights are tied to the order it was trained with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.data.label_schema import (
    FALLBACK_THRESHOLD,
    LABEL_MANIFEST_FILENAME,
    LABEL_SCHEMA_FILENAME,
    LabelSchemaError,
    find_label_schema,
    label_positive_counts,
    labels_for_sample,
    load_label_manifest,
    load_label_schema,
    parse_label_schema,
    write_label_schema,
)

_SCHEMA = {
    "schema_version": 1,
    "labels": ["neutral", "sexy", "gore"],
    "axes": {
        "violence": {"ordered": False, "labels": ["gore"]},
        "sexual": {"ordered": True, "labels": ["neutral", "sexy"]},
    },
    "default_thresholds": {"gore": 0.35},
    "default_threshold": 0.5,
}


def _write(tmp_path: Path, schema: dict | None = None, manifest_lines: list[str] | None = None) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / LABEL_SCHEMA_FILENAME).write_text(
        json.dumps(schema if schema is not None else _SCHEMA), encoding="utf-8"
    )
    if manifest_lines is not None:
        (data_dir / LABEL_MANIFEST_FILENAME).write_text(
            "\n".join(manifest_lines), encoding="utf-8"
        )
    return data_dir


def test_parse_basic_schema() -> None:
    schema = parse_label_schema(_SCHEMA)
    assert schema.labels == ["neutral", "sexy", "gore"]
    assert schema.num_labels == 3
    assert schema.index_of("gore") == 2
    assert set(schema.axes) == {"violence", "sexual"}
    assert schema.axes["sexual"].ordered is True


def test_thresholds_fall_back_to_the_default() -> None:
    schema = parse_label_schema(_SCHEMA)
    assert schema.threshold_for("gore") == 0.35
    assert schema.threshold_for("neutral") == 0.5
    assert schema.thresholds() == [0.5, 0.5, 0.35]


def test_default_threshold_when_absent() -> None:
    schema = parse_label_schema({"labels": ["a"]})
    assert schema.default_threshold == FALLBACK_THRESHOLD


def test_appending_a_label_does_not_move_existing_indices() -> None:
    """
    The reason label order comes from the schema and not from sorted directory names.

    ``sorted()`` would place a new label alphabetically and silently permute every index
    after it, invalidating an already-trained model's output mapping.
    """
    before = parse_label_schema(_SCHEMA)
    extended = dict(_SCHEMA)
    extended["labels"] = [*_SCHEMA["labels"], "bdsm"]
    after = parse_label_schema(extended)

    for name in before.labels:
        assert after.index_of(name) == before.index_of(name)
    assert after.index_of("bdsm") == 3


def test_multi_hot_encoding() -> None:
    schema = parse_label_schema(_SCHEMA)
    assert schema.multi_hot(["neutral"]) == [1.0, 0.0, 0.0]
    assert schema.multi_hot(["gore", "sexy"]) == [0.0, 1.0, 1.0]
    assert schema.multi_hot([]) == [0.0, 0.0, 0.0]


def test_unknown_label_is_rejected() -> None:
    schema = parse_label_schema(_SCHEMA)
    with pytest.raises(LabelSchemaError):
        schema.index_of("nope")


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"labels": []},
        {"labels": ["a", "a"]},
        {"labels": ["a", ""]},
        {"labels": ["a"], "axes": {"x": {"labels": ["missing"]}}},
        {"labels": ["a"], "axes": {"x": {"labels": []}}},
        {"labels": ["a"], "default_thresholds": {"missing": 0.5}},
    ],
)
def test_invalid_schemas_are_rejected(bad: dict) -> None:
    with pytest.raises(LabelSchemaError):
        parse_label_schema(bad)


def test_load_and_find_schema(tmp_path: Path) -> None:
    data_dir = _write(tmp_path)
    assert load_label_schema(data_dir).num_labels == 3
    assert load_label_schema(data_dir / LABEL_SCHEMA_FILENAME).num_labels == 3
    assert find_label_schema(data_dir) is not None
    assert find_label_schema(tmp_path / "empty") is None


def test_missing_schema_raises_when_loaded_directly(tmp_path: Path) -> None:
    with pytest.raises(LabelSchemaError):
        load_label_schema(tmp_path / "nothing")


def test_missing_manifest_is_not_an_error(tmp_path: Path) -> None:
    """An existing single-label dataset is a valid multi-label dataset with no manifest."""
    data_dir = _write(tmp_path)
    schema = load_label_schema(data_dir)
    assert load_label_manifest(data_dir, schema) == {}


def test_manifest_round_trip(tmp_path: Path) -> None:
    data_dir = _write(
        tmp_path,
        manifest_lines=[
            json.dumps({"path": "train/gore/a.jpg", "labels": ["gore", "sexy"]}),
            "",
            json.dumps({"path": "train/neutral/b.jpg", "labels": []}),
        ],
    )
    schema = load_label_schema(data_dir)
    manifest = load_label_manifest(data_dir, schema)

    assert manifest["train/gore/a.jpg"] == ["gore", "sexy"]
    assert manifest["train/neutral/b.jpg"] == []


def test_manifest_normalizes_windows_separators(tmp_path: Path) -> None:
    data_dir = _write(
        tmp_path,
        manifest_lines=[json.dumps({"path": "train\\gore\\a.jpg", "labels": ["sexy"]})],
    )
    schema = load_label_schema(data_dir)
    assert "train/gore/a.jpg" in load_label_manifest(data_dir, schema)


def test_manifest_rejects_unknown_labels_when_strict(tmp_path: Path) -> None:
    data_dir = _write(
        tmp_path,
        manifest_lines=[json.dumps({"path": "train/a/b.jpg", "labels": ["not_a_label"]})],
    )
    schema = load_label_schema(data_dir)
    with pytest.raises(LabelSchemaError):
        load_label_manifest(data_dir, schema)

    dropped = load_label_manifest(data_dir, schema, strict=False)
    assert dropped["train/a/b.jpg"] == []


def test_manifest_rejects_malformed_lines(tmp_path: Path) -> None:
    for lines in (["{not json"], [json.dumps({"labels": ["gore"]})], [json.dumps([1, 2])]):
        data_dir = _write(tmp_path, manifest_lines=lines)
        schema = load_label_schema(data_dir)
        with pytest.raises(LabelSchemaError):
            load_label_manifest(data_dir, schema)


def test_labels_for_sample_always_includes_the_folder_label() -> None:
    """
    A manifest entry can widen an image's labels but never remove the folder's assertion.

    Otherwise a typo in the manifest could silently strip the primary label from a whole
    class of images.
    """
    manifest = {"train/gore/a.jpg": ["sexy"]}
    assert labels_for_sample("train/gore/a.jpg", "gore", manifest) == ["gore", "sexy"]
    # No entry at all: just the folder label.
    assert labels_for_sample("train/gore/z.jpg", "gore", manifest) == ["gore"]
    # Manifest repeating the primary label does not duplicate it.
    assert labels_for_sample("x", "gore", {"x": ["gore"]}) == ["gore"]


def test_label_positive_counts() -> None:
    vectors = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert label_positive_counts(vectors, 3) == [2, 1, 0]


def test_write_schema_round_trips(tmp_path: Path) -> None:
    schema = parse_label_schema(_SCHEMA)
    out = write_label_schema(tmp_path / "out", schema)
    assert out.name == LABEL_SCHEMA_FILENAME

    reloaded = load_label_schema(out)
    assert reloaded.labels == schema.labels
    assert reloaded.thresholds() == schema.thresholds()
    assert set(reloaded.axes) == set(schema.axes)


def test_write_schema_to_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    """
    A write target need not already exist.

    Deciding file-vs-directory by ``is_dir()`` would misread a not-yet-created dataset
    directory as a filename and write the schema to a file named after the directory.
    """
    target = tmp_path / "brand" / "new" / "data"
    assert not target.exists()

    out = write_label_schema(target, parse_label_schema(_SCHEMA))

    assert out == target / LABEL_SCHEMA_FILENAME
    assert out.is_file()
    assert target.is_dir()


def test_write_schema_to_an_explicit_file_path(tmp_path: Path) -> None:
    out = write_label_schema(tmp_path / "custom.json", parse_label_schema(_SCHEMA))
    assert out.name == "custom.json"
    assert load_label_schema(out).labels == _SCHEMA["labels"]
