"""Export package timeline edits as combined MP4 or FCP7 XML."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree.ElementTree import Element, SubElement, tostring

from .stash import META_FILE, output_root, resolve_package_dir, stash_root
from .stash_media import probe_media, resolve_package_media
from .stash_preview import PREVIEW_FILE, find_ffmpeg

LOG = logging.getLogger("h3_lq_stash.edit")

EDITS_DIR_NAME = "_edits"
DEFAULT_FPS = 24.0


def edits_dir() -> Path:
    path = stash_root() / EDITS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_edit_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (value or "edit").strip())
    cleaned = cleaned.strip("._") or "edit"
    return cleaned[:96]


def _safe_under_output(path: Path) -> Path:
    out = output_root()
    resolved = path.resolve()
    try:
        resolved.relative_to(out)
    except ValueError as exc:
        raise ValueError(f"Path escapes output directory: {path}") from exc
    return resolved


def _read_meta(package_dir: Path) -> dict[str, Any]:
    meta_path = package_dir / META_FILE
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _probe_duration_frames(preview: Path, fps: float) -> int:
    """Fallback frame count from ffprobe / file duration when meta lacks frame_count."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return max(1, int(round(fps * 1.0)))
    # Prefer ffprobe next to ffmpeg
    probe = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe")
    if not probe.is_file():
        probe_exe = shutil.which("ffprobe")
        probe = Path(probe_exe) if probe_exe else None
    if probe is None or not Path(probe).is_file():
        return max(1, int(round(fps * 1.0)))
    proc = subprocess.run(
        [
            str(probe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(preview),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        duration = float((proc.stdout or "").strip() or "0")
    except ValueError:
        duration = 0.0
    if duration <= 0:
        return max(1, int(round(fps * 1.0)))
    return max(1, int(round(duration * fps)))


def resolve_edit_clips(
    clips: list[dict[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
) -> list[dict[str, Any]]:
    """Resolve package refs to media files and clamp in/out frames."""
    if not clips:
        raise ValueError("No clips in edit")
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []

    for raw in clips:
        package = str((raw or {}).get("package") or "").strip().replace("\\", "/")
        if not package or ".." in Path(package).parts:
            raise ValueError(f"Invalid package path: {package!r}")
        media_name = str((raw or {}).get("media") or "").strip().replace("\\", "/")
        if not media_name:
            media_name = PREVIEW_FILE
        try:
            preview = resolve_package_media(package, media_name)
        except (ValueError, FileNotFoundError):
            missing.append(f"{package}/{media_name}")
            continue
        if not preview.is_file():
            missing.append(f"{package}/{media_name}")
            continue

        package_dir = _safe_under_output(resolve_package_dir(package))
        meta = _read_meta(package_dir)
        probed = probe_media(preview, fps=float(meta.get("fps") or fps) or fps)
        meta_fps = float(probed.get("fps") or meta.get("fps") or fps) or fps
        frame_count = max(1, int(probed.get("frame_count") or meta.get("frame_count") or 1))

        in_frame = max(0, int((raw or {}).get("in_frame") or 0))
        out_frame = int(
            (raw or {}).get("out_frame")
            if (raw or {}).get("out_frame") is not None
            else frame_count
        )
        out_frame = min(frame_count, max(in_frame + 1, out_frame))

        width = int(probed.get("width") or meta.get("width") or 0) or 1280
        height = int(probed.get("height") or meta.get("height") or 0) or 720

        resolved.append(
            {
                "package": package,
                "name": meta.get("package_name") or Path(package).name,
                "media": media_name,
                "preview": preview,
                "package_dir": package_dir,
                "in_frame": in_frame,
                "out_frame": out_frame,
                "frame_count": frame_count,
                "fps": meta_fps,
                "note": str(meta.get("note") or ""),
                "prompt": str(meta.get("prompt") or ""),
                "width": width,
                "height": height,
                "has_audio": bool(probed.get("has_audio") or meta.get("audio_shape")),
            }
        )

    if missing:
        raise FileNotFoundError("Missing media for: " + ", ".join(missing))
    if not resolved:
        raise ValueError("No resolvable clips with media")
    return resolved


def _run_ffmpeg(cmd: list[str]) -> None:
    LOG.info("ffmpeg edit: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {(proc.stderr or '')[-2000:]}"
        )


def export_mp4(
    clips: list[dict[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
    name: str = "edit",
) -> dict[str, Any]:
    """Trim each clip and concat into one MP4 under _edits/."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (needed for edit export)")

    resolved = resolve_edit_clips(clips, fps=fps)
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"{_sanitize_edit_name(name)}_{stamp}.mp4"
    out_path = edits_dir() / out_name

    with tempfile.TemporaryDirectory(prefix="h3_lq_edit_") as tmp:
        tmp_dir = Path(tmp)
        segment_paths: list[Path] = []
        for i, clip in enumerate(resolved):
            seg = tmp_dir / f"seg_{i:04d}.mp4"
            in_sec = clip["in_frame"] / clip["fps"]
            out_sec = clip["out_frame"] / clip["fps"]
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{in_sec:.6f}",
                "-to",
                f"{out_sec:.6f}",
                "-i",
                str(clip["preview"]),
            ]
            if clip.get("has_audio"):
                cmd += [
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    "-g",
                    "12",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(seg),
                ]
            else:
                # Silent stereo AAC keeps concat demuxer layout uniform.
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    "-g",
                    "12",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(seg),
                ]
            _run_ffmpeg(cmd)
            if not seg.is_file() or seg.stat().st_size < 32:
                raise RuntimeError(f"Empty segment for {clip['package']}")
            segment_paths.append(seg)

        list_file = tmp_dir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segment_paths) + "\n",
            encoding="utf-8",
        )
        tmp_out = tmp_dir / "out.mp4"
        _run_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp_out),
            ]
        )
        if out_path.exists():
            out_path.unlink()
        shutil.copy2(tmp_out, out_path)

    rel = out_path.relative_to(output_root()).as_posix()
    return {"path": str(out_path.resolve()), "rel": rel, "name": out_name}


def _pathurl(path: Path) -> str:
    # file:///C:/... style for Windows + POSIX
    resolved = path.resolve()
    as_posix = resolved.as_posix()
    if re.match(r"^[A-Za-z]:/", as_posix):
        return "file://localhost/" + quote(as_posix)
    return "file://localhost" + quote(as_posix)


def _write_fcp7_xml(
    resolved: list[dict[str, Any]],
    out_path: Path,
    *,
    fps: float,
    name: str,
) -> None:
    """Write xmeml v5 for already-resolved clips. pathurl uses each clip['preview']."""
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    timebase = int(round(fps))
    edit_name = _sanitize_edit_name(name)
    width = max(clip["width"] for clip in resolved) or 1280
    height = max(clip["height"] for clip in resolved) or 720
    total_frames = sum(c["out_frame"] - c["in_frame"] for c in resolved)

    xmeml = Element("xmeml", {"version": "5"})
    sequence = SubElement(xmeml, "sequence", {"id": f"sequence-{edit_name}"})
    SubElement(sequence, "name").text = edit_name
    SubElement(sequence, "duration").text = str(total_frames)
    rate = SubElement(sequence, "rate")
    SubElement(rate, "timebase").text = str(timebase)
    SubElement(rate, "ntsc").text = "FALSE"

    media = SubElement(sequence, "media")
    video = SubElement(media, "video")
    vformat = SubElement(video, "format")
    samplechars = SubElement(vformat, "samplecharacteristics")
    vrate = SubElement(samplechars, "rate")
    SubElement(vrate, "timebase").text = str(timebase)
    SubElement(vrate, "ntsc").text = "FALSE"
    SubElement(samplechars, "width").text = str(width)
    SubElement(samplechars, "height").text = str(height)
    SubElement(samplechars, "pixelaspectratio").text = "square"
    SubElement(samplechars, "fielddominance").text = "none"

    vtrack = SubElement(video, "track")
    audio = SubElement(media, "audio")
    aformat = SubElement(audio, "format")
    asample = SubElement(aformat, "samplecharacteristics")
    SubElement(asample, "depth").text = "16"
    SubElement(asample, "samplerate").text = "48000"
    atrack = SubElement(audio, "track")

    timeline_pos = 0
    for i, clip in enumerate(resolved):
        dur = clip["out_frame"] - clip["in_frame"]
        file_id = f"file-{i}"
        clip_id = f"clipitem-{i}"

        vitem = SubElement(vtrack, "clipitem", {"id": clip_id})
        SubElement(vitem, "name").text = clip["package"]
        SubElement(vitem, "duration").text = str(dur)
        vr = SubElement(vitem, "rate")
        SubElement(vr, "timebase").text = str(timebase)
        SubElement(vr, "ntsc").text = "FALSE"
        SubElement(vitem, "start").text = str(timeline_pos)
        SubElement(vitem, "end").text = str(timeline_pos + dur)
        SubElement(vitem, "in").text = str(clip["in_frame"])
        SubElement(vitem, "out").text = str(clip["out_frame"])

        file_el = SubElement(vitem, "file", {"id": file_id})
        SubElement(file_el, "name").text = Path(clip["preview"]).name
        SubElement(file_el, "pathurl").text = _pathurl(clip["preview"])
        fr = SubElement(file_el, "rate")
        SubElement(fr, "timebase").text = str(timebase)
        SubElement(fr, "ntsc").text = "FALSE"
        SubElement(file_el, "duration").text = str(clip["frame_count"])
        fmedia = SubElement(file_el, "media")
        fv = SubElement(fmedia, "video")
        fvs = SubElement(fv, "samplecharacteristics")
        SubElement(fvs, "width").text = str(clip["width"])
        SubElement(fvs, "height").text = str(clip["height"])
        fa = SubElement(fmedia, "audio")
        fas = SubElement(fa, "samplecharacteristics")
        SubElement(fas, "depth").text = "16"
        SubElement(fas, "samplerate").text = "48000"
        SubElement(fa, "channelcount").text = "2"

        logginginfo = SubElement(vitem, "logginginfo")
        SubElement(logginginfo, "description").text = clip["note"] or clip["prompt"] or ""
        SubElement(logginginfo, "scene").text = clip["package"]
        st = SubElement(vitem, "sourcetrack")
        SubElement(st, "mediatype").text = "video"
        SubElement(st, "trackindex").text = "1"

        aitem = SubElement(atrack, "clipitem", {"id": f"{clip_id}-audio"})
        SubElement(aitem, "name").text = clip["package"]
        SubElement(aitem, "duration").text = str(dur)
        ar = SubElement(aitem, "rate")
        SubElement(ar, "timebase").text = str(timebase)
        SubElement(ar, "ntsc").text = "FALSE"
        SubElement(aitem, "start").text = str(timeline_pos)
        SubElement(aitem, "end").text = str(timeline_pos + dur)
        SubElement(aitem, "in").text = str(clip["in_frame"])
        SubElement(aitem, "out").text = str(clip["out_frame"])
        SubElement(aitem, "file", {"id": file_id})
        ast = SubElement(aitem, "sourcetrack")
        SubElement(ast, "mediatype").text = "audio"
        SubElement(ast, "trackindex").text = "1"
        link = SubElement(aitem, "link")
        SubElement(link, "linkclipref").text = clip_id
        SubElement(link, "mediatype").text = "video"
        SubElement(link, "trackindex").text = "1"
        SubElement(link, "clipindex").text = "1"

        timeline_pos += dur

    out_path.write_bytes(tostring(xmeml, encoding="utf-8", xml_declaration=True))


def build_fcp7_xml(
    clips: list[dict[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
    name: str = "edit",
) -> dict[str, Any]:
    """Write Final Cut Pro 7 / Premiere / Resolve xmeml v5 XML."""
    resolved = resolve_edit_clips(clips, fps=fps)
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    edit_name = _sanitize_edit_name(name)
    out_name = f"{edit_name}_{stamp}.xml"
    out_path = edits_dir() / out_name
    _write_fcp7_xml(resolved, out_path, fps=fps, name=edit_name)
    rel = out_path.relative_to(output_root()).as_posix()
    return {"path": str(out_path.resolve()), "rel": rel, "name": out_name}


def _unique_copy_name(original: str, used: set[str]) -> str:
    name = Path(original).name or "clip.mp4"
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix or ".mp4"
    index = 2
    while True:
        candidate = f"{stem}_{index:02d}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def export_bundle(
    clips: list[dict[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
    name: str = "edit",
) -> dict[str, Any]:
    """Copy chosen clip files into a new folder with FCP7 XML + edit JSON."""
    resolved = resolve_edit_clips(clips, fps=fps)
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    edit_name = _sanitize_edit_name(name)
    folder = edits_dir() / f"{edit_name}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)

    used_names: set[str] = set()
    copied: dict[str, Path] = {}
    bundled: list[dict[str, Any]] = []
    json_clips: list[dict[str, Any]] = []

    try:
        for clip in resolved:
            source = Path(clip["preview"]).resolve()
            key = str(source)
            if key not in copied:
                dest_name = _unique_copy_name(source.name, used_names)
                dest = folder / dest_name
                shutil.copy2(source, dest)
                copied[key] = dest
            dest = copied[key]
            bundled_clip = dict(clip)
            bundled_clip["preview"] = dest
            bundled.append(bundled_clip)
            json_clip = {
                "package": clip["package"],
                "in_frame": clip["in_frame"],
                "out_frame": clip["out_frame"],
                "media": dest.name,
                "source_media": clip.get("media") or source.name,
            }
            json_clips.append(json_clip)

        xml_name = f"{edit_name}.xml"
        json_name = f"{edit_name}.json"
        _write_fcp7_xml(bundled, folder / xml_name, fps=fps, name=edit_name)
        (folder / json_name).write_text(
            json.dumps(
                {
                    "version": 1,
                    "fps": fps,
                    "root": "h3_lq_stash",
                    "name": edit_name,
                    "clips": json_clips,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    rel = folder.relative_to(output_root()).as_posix()
    return {
        "path": str(folder.resolve()),
        "rel": rel,
        "name": folder.name,
        "xml": xml_name,
        "json": json_name,
        "files": [path.name for path in copied.values()],
    }
