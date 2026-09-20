"""Reproduce the pinned promoter sample from its origin and (re)write the inline block of ``samples.py``.

Steps, all against immutable references:

1. fetch the four gzipped interval lists of ``human_nontata_promoters`` from the Apache-2.0
   ``ML-Bioinfo-CEITEC/genomic_benchmarks`` repository at ``CORPUS_REVISION`` and check their sizes and
   SHA-256 against ``CORPUS_FILES``;
2. draw ``SAMPLE_SPLIT`` records per split with ``SAMPLE_SEED`` (train/validation from the origin train lists,
   test from the origin test lists, equal counts per label);
3. read every interval's bases from the GRCh38 reference through the Ensembl REST batch endpoint (1-based
   inclusive coordinates; ``-`` strand returned reverse-complemented, the origin's convention);
4. cross-check every sequence and label against the benchmark authors' Hub re-upload
   (``katarinagresova/Genomic_Benchmarks_human_nontata_promoters``) - an oracle for the coordinate
   conversion, not a data source; and
5. with ``--write``, replace ``_INLINE_RECORDS`` / ``SAMPLE_DIGEST`` / ``SAMPLE_SPLIT_DIGESTS`` in
   ``samples.py``; without it, assert the reproduced draw equals the inline block byte for byte.

Network access is required (GitHub raw, Ensembl REST, the Hub). Nothing in the package or the notebook calls
this tool: it exists so the inline data can be regenerated and audited.
"""

from __future__ import annotations

# ruff: noqa: E501  -- the inline sample block and contract lines are kept at the fleet width
import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nucleotide_transformer_genomics_pipeline import samples as S  # noqa: E402

ORACLE_URL = (
    "https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_nontata_promoters/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)
FIELDS = ("id", "region", "start", "end", "strand", "label", "split", "sequence")


def _session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "nucleotide-transformer-genomics-pipeline/pin_sample"
    return session


def fetch_intervals(session) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"train": [], "test": []}
    for rel, (expected_bytes, expected_sha) in S.CORPUS_FILES.items():
        blob = session.get(f"{S.CORPUS_URL}/{rel}", timeout=120).content
        digest = hashlib.sha256(blob).hexdigest()
        if (len(blob), digest) != (expected_bytes, expected_sha):
            raise SystemExit(f"{rel}: {len(blob)} bytes / {digest[:12]} != pinned {expected_bytes} / {expected_sha[:12]}")
        split, cls = rel.split("/")
        label = 1 if cls.startswith("positive") else 0
        for row in csv.DictReader(io.StringIO(gzip.decompress(blob).decode("utf-8"))):
            rows[split].append({"id": row["id"], "region": row["region"], "start": int(row["start"]), "end": int(row["end"]), "strand": row["strand"], "label": label})
    for split, expected in S.CORPUS_ROWS.items():
        if len(rows[split]) != expected:
            raise SystemExit(f"origin {split} list has {len(rows[split])} rows, pinned {expected}")
    return rows


def draw(rows: dict[str, list[dict]]) -> list[dict]:
    rng = random.Random(S.SAMPLE_SEED)
    sample: list[dict] = []
    per_class = {name: n // 2 for name, n in S.SAMPLE_SPLIT.items()}
    for label in (1, 0):
        pool = sorted((r for r in rows["train"] if r["label"] == label), key=lambda r: r["id"])
        rng.shuffle(pool)
        n_train, n_val = per_class["train"], per_class["validation"]
        sample += [{**r, "split": "train"} for r in pool[:n_train]]
        sample += [{**r, "split": "validation"} for r in pool[n_train : n_train + n_val]]
    for label in (1, 0):
        pool = sorted((r for r in rows["test"] if r["label"] == label), key=lambda r: r["id"])
        rng.shuffle(pool)
        sample += [{**r, "split": "test"} for r in pool[: per_class["test"]]]
    if len({r["id"] for r in sample}) != len(sample):
        raise SystemExit("duplicate ids in the draw")
    return sample


def _region(r: dict) -> str:
    return f"{r['region'].removeprefix('chr')}:{r['start'] + 1}..{r['end']}:{1 if r['strand'] == '+' else -1}"


def fetch_sequences(session, sample: list[dict]) -> None:
    for start in range(0, len(sample), 50):
        batch = sample[start : start + 50]
        for attempt in range(6):
            try:
                response = session.post(
                    S.ENSEMBL_REST,
                    params={"coord_system_version": "GRCh38"},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json={"regions": [_region(r) for r in batch]},
                    timeout=120,
                )
                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After", 5)))
                    continue
                response.raise_for_status()
                break
            except OSError:
                time.sleep(3 * (attempt + 1))
        else:
            raise SystemExit("Ensembl REST unreachable")
        by_query = {item["query"]: item["seq"] for item in response.json()}
        for r in batch:
            r["sequence"] = by_query[_region(r)].upper()
            if len(r["sequence"]) != r["end"] - r["start"]:
                raise SystemExit(f"{r['id']}: {len(r['sequence'])} bases for a {r['end'] - r['start']}-base interval")
        time.sleep(0.2)


def oracle_check(session, sample: list[dict]) -> dict[str, int]:
    import pyarrow.parquet as pq

    oracle: dict[str, int] = {}
    for split in ("train", "test"):
        table = pq.read_table(io.BytesIO(session.get(ORACLE_URL.format(split=split), timeout=120).content)).to_pydict()
        for seq, label in zip(table["seq"], table["label"], strict=True):
            oracle[str(seq).upper()] = int(label)
    matches = sum(1 for r in sample if oracle.get(r["sequence"]) == r["label"])
    absent = sum(1 for r in sample if r["sequence"] not in oracle)
    return {"matches": matches, "absent": absent, "label_disagreements": len(sample) - matches - absent}


def render_block(sample: list[dict]) -> str:
    return "\n".join("\t".join(str(r[f]) for f in FIELDS) for r in sample)


def write_samples(sample: list[dict]) -> None:
    path = ROOT / "src" / "nucleotide_transformer_genomics_pipeline" / "samples.py"
    text = path.read_text(encoding="utf-8")
    digest_all = S.dataset_digest(sample)
    per_split = {name: S.dataset_digest([r for r in sample if r["split"] == name]) for name in S.SAMPLE_SPLIT}
    text, n1 = re.subn(r'^SAMPLE_DIGEST = "[^"]*"', f'SAMPLE_DIGEST = "{digest_all}"', text, flags=re.M)
    text, n2 = re.subn(r"^SAMPLE_SPLIT_DIGESTS = .*$", f"SAMPLE_SPLIT_DIGESTS = {json.dumps(per_split)}", text, flags=re.M)
    text, n3 = re.subn(r'_INLINE_RECORDS = """\n.*?\n"""', lambda _m: '_INLINE_RECORDS = """\n' + render_block(sample) + '\n"""', text, flags=re.S)
    if (n1, n2, n3) != (1, 1, 1):
        raise SystemExit(f"samples.py markers not found exactly once: {(n1, n2, n3)}")
    path.write_text(text, encoding="utf-8", newline="\n")
    print({"wrote": str(path), "records": len(sample), "digest": digest_all, "block_bytes": len(render_block(sample))})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="rewrite the inline block in samples.py (default: check it)")
    parser.add_argument("--skip-oracle", action="store_true", help="skip the Hub re-upload cross-check")
    args = parser.parse_args(argv)
    session = _session()
    started = time.perf_counter()
    rows = fetch_intervals(session)
    sample = draw(rows)
    fetch_sequences(session, sample)
    report = {"records": len(sample), "splits": {name: sum(r["split"] == name for r in sample) for name in S.SAMPLE_SPLIT}, "seconds": round(time.perf_counter() - started, 1)}
    if not args.skip_oracle:
        report["oracle"] = oracle_check(session, sample)
        if report["oracle"]["matches"] != len(sample):
            raise SystemExit(f"oracle disagreement: {report['oracle']}")
    print(report)
    if args.write:
        write_samples(sample)
        return 0
    if render_block(sample) != S._INLINE_RECORDS.strip("\n"):
        raise SystemExit("inline block differs from the reproduced draw; run with --write to update it")
    if S.dataset_digest(sample) != S.SAMPLE_DIGEST:
        raise SystemExit("SAMPLE_DIGEST does not match the reproduced draw")
    print("inline sample reproduced exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
