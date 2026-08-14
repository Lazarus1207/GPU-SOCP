#!/usr/bin/env python3
"""patch_pdhcg.py v3 - Apply/strip ablation patches via *environment variables*.

The previous v2 tried to inject --no_halpern/--no_reflection into cli.c's
getopt_long options, but the actual cli.c layout differs from the assumed
anchors. v3 abandons CLI flag injection entirely and instead lets the
run-time defaults in utils.cu read ``getenv`` so that the four configs are
selected purely via shell environment variables at run time.

Configs
-------
baseline      : strip ablation residue, no defaults emitted at all
                 -> halpern and reflection are unconditionally ON
                    (i.e. params fields simply do not exist).
no_halpern    : emit ``params->use_halpern = false;``
                 ``params->use_reflection = true;``
                 (env vars still respected if both are set, but here only
                  one defaults toggle is needed.)
no_reflection : emit ``params->use_halpern = true;``
                 ``params->use_reflection = false;``
both_off      : emit ``params->use_halpern = false;``
                 ``params->use_reflection = false;``

In addition, a *single* env-var override block is inserted once into utils.cu
that lets the user override defaults at run time::

    PDHCG_USE_HALPERN    = 0|1
    PDHCG_USE_REFLECTION = 0|1

Usage
-----
    python3 patch_pdhcg.py --config <cfg> --src-dir <path> [--no-compile]

The CLI does NOT take a config flag for no_halpern/no_reflection any more.
Just ``export PDHCG_USE_HALPERN=0`` before running pdhcg.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Lines matching any of these patterns are stripped before each config's
# specific insertion, so the script is idempotent across runs.
STRIP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\buse_halpern\b"),
    re.compile(r"\buse_reflection\b"),
    re.compile(r"\bPDHCG_USE_HALPERN\b"),
    re.compile(r"\bPDHCG_USE_REFLECTION\b"),
    re.compile(r"\bAblation\b"),
    re.compile(r"if\s*\(.*!.*->use_halpern.*\)\s*return\s*;\s*$"),
    re.compile(r"if\s*\(.*!.*->use_reflection.*\)\s*return\s*;\s*$"),
    re.compile(r"\b__ablation_defaults__\b"),
]


# Env-var override block to be inserted immediately before the default
# values block in utils.cu. This block reads shell env vars and overrides
# whichever defaults were emitted by the chosen config.
ENV_OVERRIDE_BLOCK: List[str] = [
    "    /* __ablation_defaults__ env-var override (start) */",
    "    {",
    "        const char *_env_h = getenv(\"PDHCG_USE_HALPERN\");",
    "        if (_env_h != NULL) params->use_halpern = (atoi(_env_h) != 0);",
    "        const char *_env_r = getenv(\"PDHCG_USE_REFLECTION\");",
    "        if (_env_r != NULL) params->use_reflection = (atoi(_env_r) != 0);",
    "    }",
    "    /* __ablation_defaults__ env-var override (end) */",
]


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_lines(path: Path) -> Tuple[List[str], str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    eol = detect_eol(text)
    return text.splitlines(), eol, text


def write_lines(path: Path, lines: List[str], eol: str, original_text: str) -> None:
    out = eol.join(lines)
    if original_text.endswith(eol):
        out += eol
    tmp = path.parent / (path.name + ".patch_tmp")
    tmp.write_text(out, encoding="utf-8")
    os.replace(tmp, path)


def find_matching_lines(lines: List[str], pattern: re.Pattern) -> List[int]:
    return [i for i, line in enumerate(lines) if pattern.search(line)]


# ----------------------------------------------------------------------
# Mutators
# ----------------------------------------------------------------------

def strip_lines(path: Path, patterns: List[re.Pattern]) -> int:
    lines, eol, text = read_lines(path)
    kept = [line for line in lines if not any(p.search(line) for p in patterns)]
    removed = len(lines) - len(kept)
    if removed > 0:
        write_lines(path, kept, eol, text)
    return removed


def insert_after_all(
    path: Path,
    anchor_pattern: re.Pattern,
    new_lines: List[str],
    expected_count: Optional[int] = None,
) -> List[int]:
    lines, eol, text = read_lines(path)
    anchor_indices = find_matching_lines(lines, anchor_pattern)
    if expected_count is not None and len(anchor_indices) != expected_count:
        die(
            f"{path}: expected {expected_count} anchor match(es) for "
            f"{anchor_pattern.pattern!r}, found {len(anchor_indices)}: "
            f"{[i + 1 for i in anchor_indices]}"
        )
    if not anchor_indices:
        die(f"{path}: no anchor match for {anchor_pattern.pattern!r}")

    for anchor_idx in reversed(anchor_indices):
        for j, new_line in enumerate(new_lines):
            lines.insert(anchor_idx + 1 + j, new_line)

    write_lines(path, lines, eol, text)
    return [i + 1 for i in anchor_indices]


def insert_before_all(
    path: Path,
    anchor_pattern: re.Pattern,
    new_lines: List[str],
    expected_count: Optional[int] = None,
) -> List[int]:
    lines, eol, text = read_lines(path)
    anchor_indices = find_matching_lines(lines, anchor_pattern)
    if expected_count is not None and len(anchor_indices) != expected_count:
        die(
            f"{path}: expected {expected_count} anchor match(es) for "
            f"{anchor_pattern.pattern!r}, found {len(anchor_indices)}: "
            f"{[i + 1 for i in anchor_indices]}"
        )
    if not anchor_indices:
        die(f"{path}: no anchor match for {anchor_pattern.pattern!r}")

    for anchor_idx in reversed(anchor_indices):
        for j, new_line in enumerate(new_lines):
            lines.insert(anchor_idx + j, new_line)

    write_lines(path, lines, eol, text)
    return [i + 1 for i in anchor_indices]


def insert_into_function_body(
    path: Path,
    func_def_re: re.Pattern,
    guard_line: str,
    expected_count: int = 1,
) -> int:
    lines, eol, text = read_lines(path)
    indices = find_matching_lines(lines, func_def_re)
    if len(indices) != expected_count:
        die(
            f"{path}: expected exactly {expected_count} match(es) for "
            f"{func_def_re.pattern!r}, found {len(indices)}: "
            f"{[i + 1 for i in indices]}"
        )
    func_idx = indices[0]
    if func_idx + 1 >= len(lines):
        die(f"{path}: func def at line {func_idx + 1} has no next line")
    next_line = lines[func_idx + 1]
    if "{" not in next_line:
        die(
            f"{path}: func def at line {func_idx + 1}: "
            f"next line is not `{{`: {next_line!r}"
        )

    new_lines = list(lines)
    new_lines.insert(func_idx + 1, guard_line)
    write_lines(path, new_lines, eol, text)
    return func_idx + 2


def backup_file(path: Path, backup_dir: Optional[Path]) -> None:
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{path.name}.orig.{ts}"
    shutil.copy2(path, dest)
    info(f"Backed up {path} -> {dest}")


# ----------------------------------------------------------------------
# Per-file patchers
# ----------------------------------------------------------------------

def patch_types_h(src_dir: Path, config: str) -> None:
    """Insert struct fields use_halpern / use_reflection around reflection_coefficient."""
    path = src_dir / "include" / "pdhcg_types.h"
    if not path.exists():
        die(f"File not found: {path}")
    info(f"Patching {path}")

    if config == "baseline":
        info("  baseline: nothing to add")
        return

    refl_field_re = re.compile(r"^\s*double\s+reflection_coefficient\s*;\s*$")
    insert_before_all(path, refl_field_re, ["bool use_halpern;"], expected_count=1)
    insert_after_all(path, refl_field_re, ["bool use_reflection;"], expected_count=1)
    info("  pdhcg_types.h patched OK")


def patch_utils_cu(src_dir: Path, config: str) -> None:
    """Insert default values + env-var override block in utils.cu.

    Anchor: ``params->reflection_coefficient = 1.0;`` (one match expected).
    Insert before it:
        1. Two ``params->use_X = <default>;`` lines based on config.
        2. The env-var override block that lets run-time env vars win.
    """
    path = src_dir / "src" / "utils.cu"
    if not path.exists():
        die(f"File not found: {path}")
    info(f"Patching {path}")

    if config == "baseline":
        info("  baseline: nothing to add")
        return

    defaults = {
        "no_halpern":    ("false", "true"),
        "no_reflection": ("true",  "false"),
        "both_off":      ("false", "false"),
    }
    if config not in defaults:
        die(f"Invalid config: {config}")
    use_h_default, use_r_default = defaults[config]

    refl_assign_re = re.compile(
        r"^\s*params->reflection_coefficient\s*=\s*1\.0\s*;\s*$"
    )

    block = [
        f"params->use_halpern = {use_h_default};",
        f"params->use_reflection = {use_r_default};",
    ] + list(ENV_OVERRIDE_BLOCK)

    insert_before_all(path, refl_assign_re, block, expected_count=1)
    info("  utils.cu patched OK")


def patch_pdhg_core_op_cu(src_dir: Path, config: str) -> None:
    """Insert runtime guards into halpern_update and/or recompute_reflected_at_cones."""
    path = src_dir / "src" / "pdhg_core_op.cu"
    if not path.exists():
        die(f"File not found: {path}")
    info(f"Patching {path}")

    if config == "baseline":
        info("  baseline: nothing to add")
        return

    halpern_re = re.compile(r"^\s*void\s+halpern_update\s*\(")
    refl_re = re.compile(r"^\s*static\s+void\s+recompute_reflected_at_cones\s*\(")

    halpern_guard = "    if (state->params != NULL && !state->params->use_halpern) return;"
    refl_guard = "    if (state->params != NULL && !state->params->use_reflection) return;"

    if config in ("no_halpern", "both_off"):
        insert_into_function_body(path, halpern_re, halpern_guard, expected_count=1)
        info("  inserted halpern guard")
    if config in ("no_reflection", "both_off"):
        insert_into_function_body(path, refl_re, refl_guard, expected_count=1)
        info("  inserted reflection guard")

    info("  pdhg_core_op.cu patched OK")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply/strip ablation patches on PDHCG source code (v3, env-var driven).",
    )
    ap.add_argument(
        "--config",
        required=True,
        choices=["baseline", "no_halpern", "no_reflection", "both_off"],
        help="Ablation configuration to apply. Note: env vars PDHCG_USE_HALPERN / "
             "PDHCG_USE_REFLECTION can still override at run time.",
    )
    ap.add_argument(
        "--src-dir",
        required=True,
        type=Path,
        help="Path to the PDHCG source root directory.",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        default=True,
        help="Skip compilation step (default: skip).",
    )
    ap.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="If set, backup originals here with .orig.<timestamp> suffix.",
    )
    args = ap.parse_args()

    src = args.src_dir.resolve()
    if not src.is_dir():
        die(f"--src-dir does not exist or is not a directory: {src}")

    info(f"Config:    {args.config}")
    info(f"Source:    {src}")
    info(f"Backup:    {args.backup_dir if args.backup_dir else '(none)'}")

    targets: List[Path] = [
        src / "include" / "pdhcg_types.h",
        src / "src" / "utils.cu",
        src / "src" / "pdhg_core_op.cu",
    ]
    for t in targets:
        if not t.exists():
            die(f"Missing source file: {t}")

    # 1) Backup originals before any modification.
    for t in targets:
        backup_file(t, args.backup_dir)

    # 2) Global strip pass for idempotency.
    info("Pre-strip pass (idempotency):")
    for t in targets:
        n = strip_lines(t, STRIP_PATTERNS)
        info(f"  {t}: removed {n} residual line(s)")

    # 3) Apply config-specific additions on top of stripped baseline.
    if args.config != "baseline":
        patch_types_h(src, args.config)
        patch_utils_cu(src, args.config)
        patch_pdhg_core_op_cu(src, args.config)
    else:
        info("Baseline config: no additions applied.")

    info(f"Done. config={args.config}")
    if args.no_compile:
        info("--no-compile set: compilation skipped. Build externally if needed.")


if __name__ == "__main__":
    main()
