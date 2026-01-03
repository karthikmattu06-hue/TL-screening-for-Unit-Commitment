#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TEXT_EXTS = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}


@dataclass
class FileDigest:
    relpath: str
    n_lines: int
    has_argparse: bool
    model_classes: list[str]
    checkpoint_strings: list[str]
    dataset_strings: list[str]
    key_imports: list[str]


def is_ignored_path(p: Path) -> bool:
    parts = set(p.parts)
    # skip heavy or irrelevant directories
    return any(
        x in parts
        for x in {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            "results",
        }
    )


def tree(root: Path, max_depth: int = 5, only: Iterable[str] | None = None) -> list[str]:
    root = root.resolve()
    lines: list[str] = []

    only_prefixes = None
    if only:
        only_prefixes = [str((root / o).resolve()) for o in only]

    def allow(p: Path) -> bool:
        if only_prefixes is None:
            return True
        ps = str(p.resolve())
        return any(ps.startswith(pref) for pref in only_prefixes)

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)

        if is_ignored_path(dp):
            dirnames[:] = []
            continue

        # depth pruning
        rel = dp.relative_to(root)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue

        # prefix filter
        if not allow(dp):
            dirnames[:] = []
            continue

        indent = "  " * len(rel.parts)
        lines.append(f"{indent}{rel.as_posix() or '.'}/")

        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            fp = dp / fn
            if is_ignored_path(fp):
                continue
            if not allow(fp):
                continue
            lines.append(f"{indent}  {fn}")

    return lines


def read_text_safe(p: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<<ERROR reading {p}: {e}>>"


def digest_python(p: Path, root: Path) -> FileDigest:
    txt = read_text_safe(p)
    lines = txt.splitlines()

    # crude but effective signals
    has_argparse = "argparse" in txt and "add_argument" in txt

    model_classes = re.findall(
        r"^class\s+([A-Za-z0-9_]+)\s*\(", txt, flags=re.M)
    # include dataclasses too
    model_classes += re.findall(
        r"^class\s+([A-Za-z0-9_]+)\s*:", txt, flags=re.M)

    # checkpoint/dataset patterns
    checkpoint_strings = []
    for pat in [
        r"models[/\\][^\"']+",
        r"best\.pt",
        r"metrics\.json",
        r"torch\.load\(",
        r"torch\.save\(",
    ]:
        if re.search(pat, txt):
            checkpoint_strings.append(pat)

    dataset_strings = []
    for pat in [
        r"datasets[/\\][^\"']+",
        r"dataset_windows\.npz",
        r"\.npz",
        r"parquet",
        r"demands\.parquet",
        r"load_npz",
    ]:
        if re.search(pat, txt):
            dataset_strings.append(pat)

    # key imports (top 50 lines)
    key_imports = []
    for ln in lines[:80]:
        ln = ln.strip()
        if ln.startswith("import ") or ln.startswith("from "):
            key_imports.append(ln)

    relpath = str(p.relative_to(root))
    return FileDigest(
        relpath=relpath,
        n_lines=len(lines),
        has_argparse=has_argparse,
        model_classes=sorted(set(model_classes)),
        checkpoint_strings=checkpoint_strings,
        dataset_strings=dataset_strings,
        key_imports=key_imports[:25],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root (directory that contains tcuc_screening/).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="tcuc_screening",
        help="Comma-separated subpaths to include in tree/digests (relative to root).",
    )
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument(
        "--write_report",
        action="store_true",
        help="Write a markdown report to tcuc_screening/results/repo_snapshot.md (safe for sharing).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    only = [x.strip() for x in args.only.split(",") if x.strip()]

    print("\n=== TREE (filtered) ===")
    for line in tree(root, max_depth=args.max_depth, only=only):
        print(line)

    # Pick python files that matter
    candidates = []
    for sub in only:
        base = (root / sub).resolve()
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if is_ignored_path(p):
                continue
            candidates.append(p)

    candidates.sort()

    print("\n=== PYTHON DIGESTS (signals) ===")
    digests: list[FileDigest] = []
    for p in candidates:
        d = digest_python(p, root)
        # Print only interesting files (argparse / predictors / uc_eval / scripts / models)
        if (
            d.has_argparse
            or "/scripts/" in d.relpath
            or "/models/" in d.relpath
            or "/uc_eval/" in d.relpath
            or "predict" in d.relpath
            or "train" in d.relpath
        ):
            digests.append(d)
            print(f"\n--- {d.relpath} ({d.n_lines} lines) ---")
            if d.has_argparse:
                print("argparse: yes")
            if d.model_classes:
                print("classes:", ", ".join(
                    d.model_classes[:15]) + (" ..." if len(d.model_classes) > 15 else ""))
            if d.checkpoint_strings:
                print("checkpoint signals:", ", ".join(d.checkpoint_strings))
            if d.dataset_strings:
                print("dataset signals:", ", ".join(d.dataset_strings))
            if d.key_imports:
                print("imports (top):")
                for imp in d.key_imports:
                    print("  " + imp)

    if args.write_report:
        out = root / "tcuc_screening" / "results" / "repo_snapshot.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        md = []
        md.append("# Repo Snapshot\n")
        md.append("## Tree (filtered)\n")
        md.append("```\n" + "\n".join(tree(root,
                  max_depth=args.max_depth, only=only)) + "\n```\n")
        md.append("## File Digests\n")
        for d in digests:
            md.append(f"### `{d.relpath}`\n")
            md.append(f"- Lines: {d.n_lines}\n")
            md.append(f"- argparse: {d.has_argparse}\n")
            if d.model_classes:
                md.append(f"- Classes: {', '.join(d.model_classes)}\n")
            if d.checkpoint_strings:
                md.append(
                    f"- Checkpoint signals: {', '.join(d.checkpoint_strings)}\n")
            if d.dataset_strings:
                md.append(
                    f"- Dataset signals: {', '.join(d.dataset_strings)}\n")
            if d.key_imports:
                md.append("**Top imports:**\n")
                md.append("```")
                md.extend(d.key_imports)
                md.append("```\n")
        out.write_text("\n".join(md), encoding="utf-8")
        print(f"\nWROTE REPORT: {out}")


if __name__ == "__main__":
    main()
