#!/usr/bin/env python3
"""Source hygiene checks for this repository's model card and tutorial notebook.

This is deliberately **not** the fleet's `validate_release_assets.py`: this row ships no DIMER
pipeline package while it is on licence HOLD, so there is no package identity, no generator parity
surface and no release-asset contract to enforce yet. What is checked here is what exists: the
notebook parses, its code compiles and carries no stored outputs, the pinned identity is consistent
across every document, the model card has its MODEL_CARD_SPEC 1.1 front matter and section list,
and the non-commercial licence notice is actually present where the documents promise it is.

A PASS here is not clean-runtime execution evidence; see docs/release-verification.md.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "nucleotide_transformer_colab.ipynb"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
MODEL_LICENSE = "cc-by-nc-sa-4.0"
MODEL_KEY = "nt-v2-50m-multi-species"
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|is release-grade|now release-grade"
    r"|approved for DIMER|live (?:in|on) DIMER|research use only|open[- ]source licen[sc]e)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Statements the licence gate depends on; if any of these stops being true the documents are wrong.
# The mandated licence notice quotes the licence name, which is spelled the American way; the
# surrounding prose uses British spelling. Accept either rather than forcing one into a quotation.
CARD_MARKERS = (
    re.compile(r"CC BY-NC-SA 4\.0"),
    re.compile(r"Commercial use is not permitted under this licen[sc]e"),
    re.compile(r"ShareAlike"),
    re.compile(r"DIMER status"),
    re.compile(r"trust_remote_code"),
)
NOTEBOOK_MARKERS = (
    re.compile(r"Non-commercial model licen[sc]e"),
    re.compile(r"Commercial use is not permitted under this licen[sc]e"),
    re.compile(r"trust_remote_code=True"),
    re.compile(r"[Dd]o not upload confidential or restricted"),
    # the notebook bolds "representation"; match either rendering
    re.compile(r"a \*{0,2}representation\*{0,2}, not a prediction"),
)
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone", re.compile(r"\bgit\b[^\n]*\bclone\b")),
    ("mutable model reference", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    (
        "unsafe deserialization",
        re.compile(r"\bpickle\.load|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)


class CheckError(AssertionError):
    """Raised for any defect; the message names the file and the rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_code_fences(text: str) -> str:
    """Drop fenced code blocks so that a `# comment` inside one is not read as a heading."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def check_manifest() -> dict:
    manifest = json.loads(_read(ROOT / "weights" / MODEL_KEY / "dimer-base-manifest.json"))
    _check(manifest["modelId"] == MODEL_ID, "manifest modelId drifted")
    _check(manifest["revision"] == MODEL_REVISION, "manifest revision drifted")
    paths = {entry["path"] for entry in manifest["files"]}
    _check("model.safetensors" in paths, "manifest must list model.safetensors")
    _check(
        {"modeling_esm.py", "esm_config.py"} <= paths,
        "manifest MUST list the remote-code files the notebook executes, so their digests are checked first",
    )
    for entry in manifest["files"]:
        _check(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None, f"{entry['path']}: bad digest")
    return manifest


def check_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "pipeline_tag:", "base_model:", "date_published:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check('model_card_spec: "1.1"' in front, "MODEL_CARD.md model_card_spec must be 1.1")
    _check(
        f"license: {MODEL_LICENSE}" in front, f"MODEL_CARD.md front-matter license must be {MODEL_LICENSE}"
    )
    _check(f"base_model: {MODEL_ID}" in front, "MODEL_CARD.md base_model must equal the pinned model id")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/licence claim")
    prose = _strip_code_fences(text)
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", prose)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in prose.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")
    missing = [marker.pattern for marker in CARD_MARKERS if not marker.search(text)]
    _check(not missing, f"MODEL_CARD.md missing required statements: {missing}")


def check_identity_consistency() -> None:
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md", "tutorials/README.md"):
        text = _read(ROOT / name)
        _check(MODEL_ID in text, f"{name} must name the upstream model `{MODEL_ID}`")
        _check(MODEL_REVISION in text, f"{name} must cite the immutable revision {MODEL_REVISION}")
        stray = sorted({sha for sha in re.findall(r"\b[0-9a-f]{40}\b", text) if sha != MODEL_REVISION})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/licence claim")


def check_release_status() -> None:
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade|Hold)\b", status)
    _check(match is not None, "STATUS.md must declare a 'Current status: **...**' token")
    token = match.group(1)
    _check(token == "Hold", "this row is on licence HOLD until the maintainer rules on hosting")
    for name in ("README.md", "tutorials/README.md"):
        _check(token in _read(ROOT / name), f"{name} must record the {token} status")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md needs '## Recorded executions'",
    )


def check_notebook() -> None:
    notebook = json.loads(_read(NOTEBOOK))
    _check(notebook.get("nbformat") == 4, "notebook nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer", {})
    _check(
        dimer.get("notebook_profile") == "TASK-INFERENCE",
        "metadata.dimer.notebook_profile must be TASK-INFERENCE",
    )
    _check(dimer.get("notebook_spec") == "2.0", "metadata.dimer.notebook_spec must be 2.0")
    _check(dimer.get("standalone") is True, "metadata.dimer.standalone must be true")
    _check(
        dimer.get("remote_code_executed") is True, "metadata.dimer must record that remote code is executed"
    )
    _check(
        dimer.get("model_license") == MODEL_LICENSE,
        "metadata.dimer.model_license must record the upstream licence",
    )
    cells = notebook.get("cells", [])
    _check(bool(cells) and cells[0]["cell_type"] == "markdown", "first cell must be markdown")
    code_sources, markdown = [], []
    for index, cell in enumerate(cells):
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            markdown.append(source)
            continue
        _check(cell.get("execution_count") is None, f"code cell {index} has an execution_count")
        _check(not cell.get("outputs"), f"code cell {index} persists outputs")
        _check(
            cells[index - 1]["cell_type"] == "markdown", f"code cell {index} lacks a preceding markdown cell"
        )
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise CheckError(f"code cell {index} does not compile: {exc}") from exc
        code_sources.append(source)
    code = "\n".join(code_sources)
    markdown_text = "\n".join(markdown)
    _check(not PLACEHOLDER.search(code + markdown_text), "notebook contains placeholder text")
    present = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(code)]
    _check(not present, f"notebook contains forbidden/insecure source: {present}")
    missing = [marker.pattern for marker in NOTEBOOK_MARKERS if not marker.search(markdown_text)]
    _check(not missing, f"notebook is missing required learner-facing statements: {missing}")
    _check(f"MODEL_REVISION = {MODEL_REVISION!r}" in code, "notebook must pin MODEL_REVISION")
    _check("USE_BYOD = False" in code, "the BYOD gate must default to False")
    _check("trust_remote_code=False" in code, "the notebook must load the tokenizer without remote code")
    _check(
        "verify_snapshot(WEIGHTS_DIR)" in code,
        "the notebook must verify the snapshot before loading anything",
    )
    verify_at = code.index("verify_snapshot(WEIGHTS_DIR)")
    load_at = code.index("trust_remote_code=True")
    _check(verify_at < load_at, "the snapshot MUST be verified before remote code is executed")


def main() -> int:
    check_manifest()
    check_model_card()
    check_identity_consistency()
    check_release_status()
    check_notebook()
    print("asset checks: PASS (manifest, model-card, identity, status, notebook)")
    print("NOTE: source checks only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
