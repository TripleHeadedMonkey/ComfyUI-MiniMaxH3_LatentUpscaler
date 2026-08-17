"""Package-local LQ preview and HQ upscale media helpers."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .stash import output_root, resolve_package_dir
from .stash_preview import PREVIEW_FILE, find_ffmpeg

LOG = logging.getLogger("h3_lq_stash.media")

UPSCALED_RE = re.compile(r"^(.+)_upscaled_(\d{3})\.mp4$", re.IGNORECASE)
DEFAULT_FPS = 24.0


def upscaled_filename(package_name: str, index: int) -> str:
    return f"{package_name}_upscaled_{int(index):03d}.mp4"


def parse_upscaled_index(filename: str, package_name: str) -> int | None:
    match = UPSCALED_RE.fullmatch(Path(filename).name)
    if not match:
        return None
    if match.group(1) != package_name:
        return None
    return int(match.group(2))


def is_allowed_media_filename(package_dir: Path, filename: str) -> bool:
    name = Path(filename).name
    if name != filename or not name or ".." in name:
        return False
    if name == PREVIEW_FILE:
        return True
    return parse_upscaled_index(name, Path(package_dir).name) is not None


def next_upscaled_path(package_dir: Path) -> Path:
    """First unused <package>_upscaled_NNN.mp4 under package_dir."""
    package_dir = Path(package_dir)
    name = package_dir.name
    index = 1
    while index <= 9999:
        candidate = package_dir / upscaled_filename(name, index)
        if not candidate.exists():
            return candidate
        index += 1
    raise RuntimeError(f"Too many upscale iterations in {package_dir}")


def list_upscaled_variants(package_dir: Path) -> list[dict[str, Any]]:
    """Return sorted HQ iterations found in a package folder."""
    package_dir = Path(package_dir)
    if not package_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for child in package_dir.iterdir():
        if not child.is_file():
            continue
        index = parse_upscaled_index(child.name, package_dir.name)
        if index is None:
            continue
        stat = child.stat()
        found.append(
            {
                "index": index,
                "name": child.name,
                "path": child,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )
    found.sort(key=lambda item: item["index"])
    return found


def resolve_package_media(package: str, filename: str | None = None) -> Path:
    """Resolve a package-local mp4. filename None/empty means preview.mp4."""
    package_dir = resolve_package_dir(package)
    media_name = str(filename or PREVIEW_FILE).replace("\\", "/")
    if Path(media_name).name != media_name or not media_name:
        raise ValueError(f"Refusing package media filename {filename!r}")
    if not is_allowed_media_filename(package_dir, media_name):
        raise ValueError(f"Refusing package media filename {filename!r}")
    path = (package_dir / media_name).resolve()
    try:
        path.relative_to(package_dir.resolve())
        path.relative_to(output_root())
    except ValueError as exc:
        raise ValueError(f"Media path escapes package directory: {filename}") from exc
    return path


def probe_media(path: Path, *, fps: float = DEFAULT_FPS) -> dict[str, Any]:
    """Probe duration, dimensions, and audio from an mp4."""
    path = Path(path)
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    info = {
        "frame_count": 1,
        "width": 1280,
        "height": 720,
        "has_audio": False,
        "fps": fps,
        "duration": 0.0,
    }
    if not path.is_file():
        return info

    ffmpeg = find_ffmpeg()
    probe = None
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(
            "ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"
        )
        if sibling.is_file():
            probe = sibling
        else:
            found = shutil.which("ffprobe")
            probe = Path(found) if found else None
    if probe is None or not Path(probe).is_file():
        return info

    proc = subprocess.run(
        [
            str(probe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=codec_type,width,height,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return info

    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    info["duration"] = duration

    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "audio":
            info["has_audio"] = True
            continue
        if stream.get("codec_type") != "video":
            continue
        try:
            info["width"] = int(stream.get("width") or info["width"])
            info["height"] = int(stream.get("height") or info["height"])
        except (TypeError, ValueError):
            pass
        nb_frames = stream.get("nb_frames")
        try:
            if nb_frames not in (None, "N/A", ""):
                info["frame_count"] = max(1, int(nb_frames))
        except (TypeError, ValueError):
            pass
        rate = str(stream.get("avg_frame_rate") or "")
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                if float(den):
                    info["fps"] = float(num) / float(den)
            except (TypeError, ValueError):
                pass

    if info["frame_count"] <= 1 and duration > 0:
        info["frame_count"] = max(1, int(round(duration * fps)))
    return info


def variant_payload(
    package: str,
    package_dir: Path,
    *,
    fps: float = DEFAULT_FPS,
    probe: bool = True,
) -> list[dict[str, Any]]:
    """JSON-safe HQ variant list for the gallery / Collect UI."""
    from urllib.parse import quote

    out: list[dict[str, Any]] = []
    for item in list_upscaled_variants(package_dir):
        probed = probe_media(item["path"], fps=fps) if probe else {}
        out.append(
            {
                "index": item["index"],
                "name": item["name"],
                "mtime": item["mtime"],
                "size": item["size"],
                "frame_count": probed.get("frame_count"),
                "width": probed.get("width"),
                "height": probed.get("height"),
                "has_audio": probed.get("has_audio"),
                "url": (
                    f"/h3_lq_stash/media?package={quote(package, safe='')}"
                    f"&file={quote(item['name'], safe='')}"
                ),
            }
        )
    return out
