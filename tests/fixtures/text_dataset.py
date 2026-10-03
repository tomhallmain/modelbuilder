"""
Small synthetic text-classification dataset directories using every optional file and column.

Tiers: ``reviewed`` and ``pattern`` positives, ``reviewed`` and ``unreviewed`` negatives.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_POSITIVE_STEMS = ("zorp", "vex", "glim", "brak", "quon")
_SYLLABLES = ("ka", "lo", "mi", "ne", "ru", "ta", "zo", "be", "fi", "qu", "wa", "xi")

# Rows a text-classification reader must keep verbatim: NA-like tokens, quotes, a Unicode
# line separator (which str.splitlines would split on), and emoji.
AWKWARD_TEXTS = ("NA", "null", '"quoted"', "line\u2028sep", "smile \U0001F600")


def _group(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if ch.isalnum()) or text


def _split(group: str) -> str:
    bucket = int(hashlib.sha256(f"split-v1:{group}".encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "val" if bucket < 20 else "train"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(n: int = 1200, seed: int = 0) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    seen = set()
    i = 0
    while len(rows) < n:
        i += 1
        word = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4)))
        positive = rng.random() < 0.3
        if positive:
            stem = rng.choice(_POSITIVE_STEMS)
            text = f"{word}{stem}" if rng.random() < 0.5 else f"{stem} {word}"
        else:
            text = word.capitalize() if rng.random() < 0.5 else f"{word} {rng.choice(_SYLLABLES)}"
        g = _group(text)
        if text in seen or g in {r["group"] for r in rows[-50:]}:
            continue
        seen.add(text)
        if positive:
            tier = "reviewed" if rng.random() < 0.5 else "pattern"
            source = "manual" if tier == "reviewed" else "rule"
            hint = "topic_a" if rng.random() < 0.7 else ""
            ref = ""
        else:
            tier = "reviewed" if rng.random() < 0.3 else "unreviewed"
            source = "import"
            hint = ""
            ref = f"{rng.random() * 0.5:.4f}"
        rows.append(
            {
                "text": text,
                "label": "1" if positive else "0",
                "tier": tier,
                "source": source,
                "category_hint": hint,
                "reference_score": ref,
                "group": g,
            }
        )
    for text in AWKWARD_TEXTS:
        rows.append(
            {
                "text": text,
                "label": "0",
                "tier": "unreviewed",
                "source": "import",
                "category_hint": "",
                "reference_score": "0.0100",
                "group": _group(text),
            }
        )
    # One conflict group: two spellings, opposite labels.
    rows.append(
        {"text": "Afvex", "label": "1", "tier": "pattern", "source": "rule",
         "category_hint": "topic_a", "reference_score": "", "group": "afvex"}
    )
    rows.append(
        {"text": "A_F_V_E_X", "label": "0", "tier": "unreviewed", "source": "import",
         "category_hint": "", "reference_score": "0.2000", "group": "afvex"}
    )
    # Groups must be unique per split; drop accidental group collisions across splits.
    by_group: Dict[str, str] = {}
    out = []
    for r in rows:
        r["split"] = _split(r["group"])
        if by_group.setdefault(r["group"], r["split"]) == r["split"]:
            out.append(r)
    return out


def write_text_dataset(data_dir: Path, *, n: int = 1200, seed: int = 0, manifest: bool = True) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(n, seed)
    cols = ("text", "label", "tier", "source", "category_hint", "reference_score", "group", "split")
    lines = ["\t".join(cols)] + ["\t".join(r[c] for c in cols) for r in rows]
    (data_dir / "dataset.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    conflict_rows = [r for r in rows if r["group"] == "afvex"]
    (data_dir / "label_conflicts.tsv").write_text(
        "group\tlabel\ttier\ttext\n"
        + "".join(f"{r['group']}\t{r['label']}\t{r['tier']}\t{r['text']}\n" for r in conflict_rows),
        encoding="utf-8",
        newline="\n",
    )
    unlabeled = ["Mystery phrase", "kalo mi", "zo vex"]
    (data_dir / "unlabeled.tsv").write_text(
        "text\treference_score\n" + "".join(f"{t}\t0.6000\n" for t in unlabeled),
        encoding="utf-8",
        newline="\n",
    )
    if manifest:
        write_manifest(data_dir, rows, n_unlabeled=len(unlabeled))
    return data_dir


def write_manifest(data_dir: Path, rows: List[Dict[str, str]], *, n_unlabeled: int) -> None:
    splits: Dict[str, Dict[str, int]] = {}
    for s in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == s]
        n1 = sum(1 for r in sel if r["label"] == "1")
        splits[s] = {"rows": len(sel), "label_1": n1, "label_0": len(sel) - n1}
    files = {}
    for name in ("dataset.tsv", "unlabeled.tsv", "label_conflicts.tsv"):
        p = data_dir / name
        files[name] = {"sha256": _sha(p), "bytes": p.stat().st_size}
    manifest = {
        "dataset_rows": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "tiers": dict(Counter(r["tier"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "category_hints": dict(Counter(r["category_hint"] for r in rows if r["category_hint"])),
        "splits": splits,
        "groups": len({r["group"] for r in rows}),
        "conflict_groups": 1,
        "unlabeled": n_unlabeled,
        "files": files,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def split_counts(data_dir: Path) -> Tuple[int, int, int]:
    lines = (data_dir / "dataset.tsv").read_text(encoding="utf-8").split("\n")[1:-1]
    splits = Counter(line.split("\t")[-1] for line in lines)
    return splits["train"], splits["val"], splits["test"]
