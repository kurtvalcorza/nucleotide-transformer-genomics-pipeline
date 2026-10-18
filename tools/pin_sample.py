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
5. replace every validation window that overlaps a training window on the genome with an unused origin-train
   window of the same label at least ``LOCUS_MARGIN`` bases from every train and validation window, drawn with
   ``VALIDATION_REDRAW_SEED`` (train and test unchanged; T58-6, option A);
6. draw the fresh test with ``FRESH_TEST_SEED``: ``FRESH_TEST_SIZE`` balanced windows from origin-test intervals
   not in the sample, each at least ``LOCUS_MARGIN`` bases from every sample window and from each other, then
   fetch, oracle-check and assert locus-disjointness and no shared exact / reverse-complement sequence; and
7. with ``--write``, replace ``_INLINE_RECORDS`` / ``SAMPLE_DIGEST`` / ``SAMPLE_SPLIT_DIGESTS`` /
   ``_FRESH_TEST_RECORDS`` / ``FRESH_TEST_DIGEST`` in ``samples.py``; without it, assert the reproduced draw
   equals the inline blocks byte for byte.

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


def _near(record: dict, others: list[dict], margin: int) -> bool:
    return bool(S.locus_overlaps([record], others, margin=margin))


def redraw_validation(rows: dict[str, list[dict]], sample: list[dict]) -> tuple[list[dict], dict]:
    """Replace the validation windows that overlap a training window (T58-6, option A); train is unchanged."""
    train = [r for r in sample if r["split"] == "train"]
    validation = [r for r in sample if r["split"] == "validation"]
    overlapping = {rid for rid, _o, _s in S.locus_overlaps(validation, train)}
    kept = [r for r in validation if r["id"] not in overlapping]
    used = {r["id"] for r in sample}
    rng = random.Random(S.VALIDATION_REDRAW_SEED)
    chosen: list[dict] = []
    for label in (1, 0):
        need = sum(1 for r in validation if r["id"] in overlapping and r["label"] == label)
        pool = sorted((r for r in rows["train"] if r["label"] == label and r["id"] not in used), key=lambda r: r["id"])
        rng.shuffle(pool)
        picked = 0
        for candidate in pool:
            if picked == need:
                break
            if _near(candidate, train + kept + chosen, S.LOCUS_MARGIN):
                continue
            chosen.append({**candidate, "split": "validation"})
            picked += 1
        if picked != need:
            raise SystemExit(f"only {picked} of {need} replacement validation windows with label {label}")
    replacements = iter(sorted(chosen, key=lambda r: (-r["label"], r["id"])))
    by_label: dict[int, list[dict]] = {1: [], 0: []}
    for r in replacements:
        by_label[r["label"]].append(r)
    new_sample = []
    for r in sample:
        new_sample.append(by_label[r["label"]].pop(0) if r["id"] in overlapping else r)
    report = {"validation_windows_replaced": len(overlapping), "by_label": {str(k): sum(1 for r in validation if r["id"] in overlapping and r["label"] == k) for k in (0, 1)}}
    return new_sample, report


def draw_fresh_test(rows: dict[str, list[dict]], sample: list[dict]) -> list[dict]:
    """FRESH_TEST_SIZE balanced never-scored origin-test windows, >= LOCUS_MARGIN from every sample window and each other."""
    used = {r["id"] for r in sample}
    rng = random.Random(S.FRESH_TEST_SEED)
    fresh: list[dict] = []
    for label in (1, 0):
        pool = sorted((r for r in rows["test"] if r["label"] == label and r["id"] not in used), key=lambda r: r["id"])
        rng.shuffle(pool)
        picked = 0
        for candidate in pool:
            if picked == S.FRESH_TEST_SIZE // 2:
                break
            if _near(candidate, sample + fresh, S.LOCUS_MARGIN):
                continue
            fresh.append({**candidate, "split": "fresh_test"})
            picked += 1
        if picked != S.FRESH_TEST_SIZE // 2:
            raise SystemExit(f"only {picked} fresh-test windows with label {label}")
    return fresh


def assert_disjoint(sample: list[dict], fresh: list[dict]) -> dict:
    splits = {name: [r for r in sample if r["split"] == name] for name in S.SAMPLE_SPLIT}
    splits["fresh_test"] = fresh
    report = S.check_locus_disjoint(splits, [("validation", "train"), ("fresh_test", "train"), ("fresh_test", "validation"), ("fresh_test", "test")])
    report.update({f"{k}_margin": 0 for k in S.check_locus_disjoint(splits, [("fresh_test", "train"), ("fresh_test", "validation"), ("fresh_test", "test")], margin=S.LOCUS_MARGIN)})
    report["fresh_test_internal_margin"] = len(S.locus_overlaps(fresh, fresh, margin=S.LOCUS_MARGIN))
    if report["fresh_test_internal_margin"]:
        raise SystemExit("fresh-test windows closer than LOCUS_MARGIN to each other")
    S.check_split_disjoint(splits, reverse_complement_too=True)
    report["exact_or_reverse_complement_shared"] = 0
    return report


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


def write_samples(sample: list[dict], fresh: list[dict]) -> None:
    path = ROOT / "src" / "nucleotide_transformer_genomics_pipeline" / "samples.py"
    text = path.read_text(encoding="utf-8")
    digest_all = S.dataset_digest(sample)
    per_split = {name: S.dataset_digest([r for r in sample if r["split"] == name]) for name in S.SAMPLE_SPLIT}
    text, n1 = re.subn(r'^SAMPLE_DIGEST = "[^"]*"', f'SAMPLE_DIGEST = "{digest_all}"', text, flags=re.M)
    text, n2 = re.subn(r"^SAMPLE_SPLIT_DIGESTS = .*$", f"SAMPLE_SPLIT_DIGESTS = {json.dumps(per_split)}", text, flags=re.M)
    text, n3 = re.subn(r'_INLINE_RECORDS = """\n.*?\n"""', lambda _m: '_INLINE_RECORDS = """\n' + render_block(sample) + '\n"""', text, flags=re.S)
    text, n4 = re.subn(r'^FRESH_TEST_DIGEST = "[^"]*"', f'FRESH_TEST_DIGEST = "{S.dataset_digest(fresh)}"', text, flags=re.M)
    text, n5 = re.subn(r'_FRESH_TEST_RECORDS = """\n.*?"""', lambda _m: '_FRESH_TEST_RECORDS = """\n' + render_block(fresh) + '\n"""', text, flags=re.S)
    if (n1, n2, n3, n4, n5) != (1, 1, 1, 1, 1):
        raise SystemExit(f"samples.py markers not found exactly once: {(n1, n2, n3, n4, n5)}")
    path.write_text(text, encoding="utf-8", newline="\n")
    print({"wrote": str(path), "records": len(sample), "digest": digest_all, "block_bytes": len(render_block(sample))})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="rewrite the inline block in samples.py (default: check it)")
    parser.add_argument("--skip-oracle", action="store_true", help="skip the Hub re-upload cross-check")
    parser.add_argument("--reuse-inline", action="store_true", help="take the bases of windows already in the inline blocks from them (fetched and oracle-checked when they were pinned) and fetch only new windows")
    args = parser.parse_args(argv)
    session = _session()
    started = time.perf_counter()
    rows = fetch_intervals(session)
    sample = draw(rows)
    sample, redraw = redraw_validation(rows, sample)
    fresh = draw_fresh_test(rows, sample)
    known = {r["id"]: r["sequence"] for r in S._parse_inline()} if args.reuse_inline else {}
    if args.reuse_inline and S._FRESH_TEST_RECORDS.strip():
        known.update({r["id"]: r["sequence"] for r in S.fresh_test_dataset(verify=False)})
    for r in sample + fresh:
        if r["id"] in known:
            r["sequence"] = known[r["id"]]
    fetch_sequences(session, [r for r in sample + fresh if "sequence" not in r])
    report = {"records": len(sample), "splits": {name: sum(r["split"] == name for r in sample) for name in S.SAMPLE_SPLIT}, "redraw": redraw, "fresh_test": len(fresh), "disjoint": assert_disjoint(sample, fresh), "seconds": round(time.perf_counter() - started, 1)}
    if not args.skip_oracle:
        report["oracle"] = oracle_check(session, sample + fresh)
        if report["oracle"]["matches"] != len(sample) + len(fresh):
            raise SystemExit(f"oracle disagreement: {report['oracle']}")
    print(json.dumps(report))
    if args.write:
        write_samples(sample, fresh)
        return 0
    if render_block(sample) != S._INLINE_RECORDS.strip("\n") or render_block(fresh) != S._FRESH_TEST_RECORDS.strip("\n"):
        raise SystemExit("inline blocks differ from the reproduced draw; run with --write to update them")
    if S.dataset_digest(sample) != S.SAMPLE_DIGEST or S.dataset_digest(fresh) != S.FRESH_TEST_DIGEST:
        raise SystemExit("SAMPLE_DIGEST / FRESH_TEST_DIGEST do not match the reproduced draw")
    print("inline sample and fresh test reproduced exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
