"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there (before this change it printed a dict and carried on). The router now
carries the canonical refusal channel (dat20-router-refusal/1). The in-cell invalidation is kept; a genuine failure
keeps its traceback. Section 4 is executed from the committed notebook with the embedded modules."""
# ruff: noqa: E501

from __future__ import annotations

import json
import sys

import pytest

from test_tutorial_review_2026_10_17 import _cell, _code, _modules, nb  # noqa: F401  (fixture)

REFUSAL_CELL = "nucleotide_transformer-18"
MARKER = "USE_BYOD = False  # @param"


def _section4(nb: dict, path: str, ns: dict | None = None) -> tuple[dict, BaseException | None]:  # noqa: F811
    ns = _modules(nb) if ns is None else ns
    source = _cell(nb, MARKER).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    try:
        exec(compile(source, "<section 4>", "exec"), ns)
    except Exception as exc:
        if not getattr(exc, "dimer_refusal", False):
            raise
        return ns, exc
    return ns, None


def _line(exc: BaseException | None) -> str:
    assert exc is not None, "Section 4 did not raise a refusal"
    assert type(exc).__name__ == "BYODRefusal" and isinstance(exc, ValueError) and exc.__suppress_context__
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(nb, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    ns, exc = _section4(nb, str(hostname))
    line = _line(exc)
    assert line.startswith("Refused: hostname: expected a labelled sequence CSV") and "Run after" in line
    assert "byod_refused" not in capsys.readouterr().out and ns["RUN_STATE"]["refused"].startswith("hostname: ")


def test_a_directory_and_no_upload_dialog_are_refused_by_name(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    assert _line(_section4(nb, str(tmp_path / "etc"))[1]).startswith("Refused: etc: BYOD path ")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert _line(_section4(nb, "")[1]).startswith("Refused: (no file): BYOD is on but its path field is empty")


def test_the_in_cell_invalidation_is_kept(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    ns = _modules(nb)
    ns.update(splits={"stale": []}, train_records=["stale"], adapt_result={"stale": True})
    ns, exc = _section4(nb, str(tmp_path / "missing.csv"), ns)
    assert _line(exc).startswith("Refused: missing.csv: ")
    assert not {"splits", "train_records", "adapt_result"} & set(ns)
    status = json.loads((tmp_path / "outputs" / "nucleotide_transformer_run_status.json").read_text(encoding="utf-8"))
    assert status["status"].startswith("invalidated") and status["reason"].startswith("missing.csv: ")


def test_a_genuine_failure_keeps_its_type(nb, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine.csv").write_text("sequence,label\nACGTACGTACGT,0\n", encoding="utf-8")
    ns = _modules(nb)

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        _section4(nb, str(tmp_path / "mine.csv"), ns)


def test_the_refusal_cell_id_and_the_router_channel(nb):  # noqa: F811
    cell = next(c for c in nb["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _cell(nb, MARKER) and "BYOD_PATH = ''" in source and "raise BYODRefusal(byod_label, " in source
    router = "".join(c for c in _code(nb) if "class IsolatedCellRefusal(Exception):" in c)
    assert 'if getattr(exc, "dimer_refusal", False):' in router and 'elif kind == "refusal":' in router
    assert 'return [f"Refused: {self}"]' in router
