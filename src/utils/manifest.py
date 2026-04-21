"""
Provenance manifest for plotting / analysis script outputs.

Writes two sibling files in a target directory:
    manifest.json — machine-readable structured metadata
    MANIFEST.md   — human-readable companion

Manifest captures: timestamp, hostname, git SHA + dirty flag, script path
and argv, SHA256 + mtime + size of every input file, labelled list of
outputs, free-form config dict, warnings, and per-entry extras.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileRef:
    label: str
    path: str  # absolute path at record time
    sha256: str | None = None
    mtime_iso: str | None = None
    size_bytes: int | None = None
    exists: bool = True


def _repo_root_from(start: Path) -> Path:
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


def git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return bool(out.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def git_branch(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_ref(path: str | Path, label: str, *, compute_sha: bool = True) -> FileRef:
    p = Path(path)
    if not p.exists():
        return FileRef(label=label, path=str(p.resolve() if p.is_absolute() else p), exists=False)
    stat = p.stat()
    return FileRef(
        label=label,
        path=str(p.resolve()),
        sha256=file_sha256(p) if compute_sha else None,
        mtime_iso=_dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc).isoformat(),
        size_bytes=stat.st_size,
        exists=True,
    )


@dataclass
class Manifest:
    title: str
    description: str = ""
    timestamp_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    repo_root: str | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    script_path: str | None = None
    argv: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[FileRef] = field(default_factory=list)
    outputs: list[FileRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def build_manifest(
    *,
    title: str,
    description: str = "",
    script_path: str | Path | None = None,
    argv: list[str] | None = None,
    config: dict[str, Any] | None = None,
    inputs: list[FileRef] | None = None,
    outputs: list[FileRef] | None = None,
    warnings: list[str] | None = None,
    extras: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> Manifest:
    if repo_root is None:
        repo_root = _repo_root_from(Path(script_path or __file__))
    return Manifest(
        title=title,
        description=description,
        repo_root=str(repo_root),
        git_sha=git_sha(repo_root),
        git_branch=git_branch(repo_root),
        git_dirty=git_dirty(repo_root),
        script_path=str(Path(script_path).resolve()) if script_path else None,
        argv=list(argv) if argv else [],
        config=dict(config) if config else {},
        inputs=list(inputs) if inputs else [],
        outputs=list(outputs) if outputs else [],
        warnings=list(warnings) if warnings else [],
        extras=dict(extras) if extras else {},
    )


def _json_default(o: Any) -> Any:
    if hasattr(o, "__fspath__"):
        return os.fspath(o)
    return str(o)


def write_manifest(
    out_dir: str | Path,
    manifest: Manifest,
    *,
    json_name: str = "manifest.json",
    md_name: str = "MANIFEST.md",
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / json_name
    md_path = out / md_name

    with open(json_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, default=_json_default, sort_keys=False)

    with open(md_path, "w") as f:
        f.write(_render_markdown(manifest))

    return json_path, md_path


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    size = float(n)
    for u in units:
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {u}"
    return f"{size:.1f} PiB"


def _render_markdown(m: Manifest) -> str:
    lines: list[str] = []
    lines.append(f"# {m.title}\n")
    if m.description:
        lines.append(f"{m.description}\n")
    lines.append("## Run metadata\n")
    lines.append(f"- Timestamp (UTC): `{m.timestamp_utc}`")
    lines.append(f"- Hostname: `{m.hostname}`")
    lines.append(f"- Platform: `{m.platform}`")
    lines.append(f"- Python: `{m.python_version}`")
    if m.repo_root:
        lines.append(f"- Repo root: `{m.repo_root}`")
    if m.git_sha:
        dirty = " (dirty)" if m.git_dirty else ""
        branch = f" on `{m.git_branch}`" if m.git_branch else ""
        lines.append(f"- Git: `{m.git_sha[:12]}`{branch}{dirty}")
    if m.script_path:
        lines.append(f"- Script: `{m.script_path}`")
    if m.argv:
        argv_str = " ".join(m.argv)
        lines.append(f"- Argv: `{argv_str}`")
    lines.append("")

    if m.config:
        lines.append("## Config\n")
        for k, v in m.config.items():
            if isinstance(v, (dict, list)):
                lines.append(f"- **{k}**:")
                lines.append("  ```json")
                lines.append("  " + json.dumps(v, indent=2, default=_json_default).replace("\n", "\n  "))
                lines.append("  ```")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    if m.inputs:
        lines.append(f"## Inputs ({len(m.inputs)})\n")
        lines.append("| Label | Path | Size | Modified (UTC) | SHA-256 |")
        lines.append("|---|---|---|---|---|")
        for i in m.inputs:
            exists = "" if i.exists else " (missing)"
            size = _fmt_bytes(i.size_bytes) if i.exists else "—"
            mtime = i.mtime_iso or "—"
            sha = f"`{i.sha256[:12]}`" if i.sha256 else "—"
            lines.append(f"| {i.label}{exists} | `{i.path}` | {size} | {mtime} | {sha} |")
        lines.append("")

    if m.outputs:
        lines.append(f"## Outputs ({len(m.outputs)})\n")
        lines.append("| Label | Path | Size |")
        lines.append("|---|---|---|")
        for o in m.outputs:
            exists = "" if o.exists else " (missing)"
            size = _fmt_bytes(o.size_bytes) if o.exists else "—"
            lines.append(f"| {o.label}{exists} | `{o.path}` | {size} |")
        lines.append("")

    if m.warnings:
        lines.append(f"## Warnings ({len(m.warnings)})\n")
        for w in m.warnings:
            lines.append(f"- {w}")
        lines.append("")

    if m.extras:
        lines.append("## Extras\n")
        lines.append("```json")
        lines.append(json.dumps(m.extras, indent=2, default=_json_default))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
