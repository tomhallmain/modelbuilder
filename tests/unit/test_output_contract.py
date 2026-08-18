"""
The decode contract recorded with an exported model.

The point of this block is that a consumer never has to guess how to read the output
tensor, so the tests are mostly about fields being present and unambiguous rather than
about computation.
"""

from __future__ import annotations

import json

from mb.models.output_contract import (
    MULTI_LABEL,
    SIGMOID,
    SINGLE_LABEL,
    SOFTMAX,
    OutputContract,
    single_label_contract,
)


def test_single_label_contract_shape() -> None:
    contract = single_label_contract(["a", "b", "gore"])
    assert contract.label_mode == SINGLE_LABEL
    assert contract.activation == SOFTMAX
    assert contract.labels == ["a", "b", "gore"]
    # A softmax argmax has no per-class threshold and no axis grouping.
    assert contract.thresholds is None
    assert contract.axes is None


def test_all_five_fields_are_always_present() -> None:
    """
    Multi-label-only fields are emitted as null rather than omitted.

    A consumer can then branch on the values alone, without having to tell "absent because
    this is an older artifact" from "absent because it does not apply".
    """
    block = single_label_contract(None).to_manifest_dict()
    assert set(block) == {"label_mode", "activation", "labels", "thresholds", "axes"}
    assert block["labels"] is None
    assert block["thresholds"] is None
    assert block["axes"] is None


def test_no_labels_is_null_not_empty_list() -> None:
    """None means unresolved; an empty list would claim a zero-class model."""
    assert single_label_contract(None).labels is None
    assert single_label_contract([]).labels is None


def test_labels_are_stringified_and_order_preserved() -> None:
    contract = single_label_contract(["b", "a", "c"])
    assert contract.labels == ["b", "a", "c"]  # output-index order, not sorted


def test_manifest_dict_is_json_serializable() -> None:
    """The same block goes into a JSON manifest and into ONNX metadata as a JSON string."""
    block = single_label_contract(["x", "y"]).to_manifest_dict()
    assert json.loads(json.dumps(block)) == block


def test_manifest_dict_copies_mutable_fields() -> None:
    labels = ["a", "b"]
    contract = single_label_contract(labels)
    block = contract.to_manifest_dict()
    block["labels"].append("c")
    assert contract.labels == ["a", "b"]


def test_multi_label_values_are_expressible() -> None:
    """Phase 3 populates these; the field names and vocabulary are fixed here."""
    contract = OutputContract(
        label_mode=MULTI_LABEL,
        activation=SIGMOID,
        labels=["gore", "sexy"],
        thresholds={"gore": 0.35, "sexy": 0.5},
        axes={"violence": {"ordered": False, "labels": ["gore"]}},
    )
    block = contract.to_manifest_dict()
    assert block["activation"] == SIGMOID
    assert block["thresholds"]["gore"] == 0.35
    assert block["axes"]["violence"]["labels"] == ["gore"]


def test_activation_values_are_distinct() -> None:
    """Applying the wrong one silently produces plausible numbers, so they must not alias."""
    assert SOFTMAX != SIGMOID
    assert SINGLE_LABEL != MULTI_LABEL
