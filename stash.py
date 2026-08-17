"""Save / load MiniMax H3 packages for deferred HQ upscale.

Packages live under ComfyUI/output/h3_lq_stash/[project/]<name>/ with:
  latent.safetensors, conditioning.pt, meta.json,
  and optionally preview.mp4 + thumb.jpg for the gallery browser.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import folder_paths
import torch
from safetensors.torch import load_file, save_file

from .stash_preview import PREVIEW_FILE, THUMB_FILE, write_preview_from_images
from .utils import extract_tensor, wrap_tensor

PACKAGE_FORMAT = "h3_lq_package_v1"
PACKAGE_PIPE_TYPE = "H3_PACKAGE_PIPE"
STASH_ROOT_NAME = "h3_lq_stash"
LATENT_FILE = "latent.safetensors"
COND_FILE = "conditioning.pt"
META_FILE = "meta.json"
NO_PACKAGE = "(no packages)"
SELECTION_VERSION = 1
# MiniMax H3 always generates at 24 fps.
H3_FPS = 24.0
LOAD_SOURCE_SELECTION = "selection list"
LOAD_SOURCE_TIMELINE = "timeline order"
LOAD_SOURCE_TIMELINE_CROPS = "timeline order and crops"
LOAD_SOURCES = [
    LOAD_SOURCE_SELECTION,
    LOAD_SOURCE_TIMELINE,
    LOAD_SOURCE_TIMELINE_CROPS,
]
CHUNK_MODE_DIRECT = "direct load"
CHUNK_MODE_SCENES = "scene aware chunks (experimental)"
CHUNK_MODE_SCENES_LEGACY = "scene aware chunks"
CHUNK_MODES = [CHUNK_MODE_DIRECT, CHUNK_MODE_SCENES]


def output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


def stash_root() -> Path:
    root = output_root() / STASH_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _sanitize_segment(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned[:96]


def _sanitize_project(value: str) -> str:
    """Allow nested project path with / separators; sanitize each segment."""
    raw = (value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [_sanitize_segment(p) for p in raw.split("/") if p and p not in (".", "..")]
    parts = [p for p in parts if p]
    return "/".join(parts)


def _sanitize_name(value: str) -> str:
    return _sanitize_segment(value) or "h3_lq"


def _unique_package_dir(prefix: str, *, project: str = "") -> Path:
    project_rel = _sanitize_project(project)
    parent = stash_root() / project_rel if project_rel else stash_root()
    parent.mkdir(parents=True, exist_ok=True)
    base = _sanitize_name(prefix)
    candidate = parent / base
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    counter = 1
    while True:
        candidate = parent / f"{base}_{counter:05d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        counter += 1


def package_rel_path(package_dir: Path) -> str:
    """Relative path from stash root, forward-slash."""
    root = stash_root()
    resolved = package_dir.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return package_dir.name
    return rel.as_posix()


def package_output_rel(package_dir: Path) -> str:
    """Path relative to ComfyUI's output folder, forward-slash."""
    resolved = Path(package_dir).resolve()
    try:
        return resolved.relative_to(output_root()).as_posix()
    except ValueError:
        rel = package_rel_path(resolved)
        return f"{STASH_ROOT_NAME}/{rel}" if rel else STASH_ROOT_NAME


def canonical_package_name(
    package_dir: Path,
    *,
    scene_index: int | None = None,
    scene_count: int = 1,
) -> str:
    """Human-facing output-relative package id; scene labels are not filesystem paths."""
    base = package_output_rel(package_dir)
    if scene_count > 1 and scene_index is not None:
        return f"{base}#scene{int(scene_index):02d}"
    return base


def build_package_pipe(
    package_dir: Path,
    meta: dict[str, Any] | None = None,
    *,
    scene_index: int = 0,
    scene_count: int = 1,
) -> dict[str, Any]:
    """Provenance payload for Upscale Collect. Filesystem fields never include #scene."""
    resolved = Path(package_dir).resolve()
    meta = meta or {}
    return {
        "type": "h3_package_pipe_v1",
        "package_rel": str(meta.get("package_rel") or package_rel_path(resolved)),
        "output_rel": package_output_rel(resolved),
        "package_name": resolved.name,
        "package_dir": str(resolved),
        "scene_index": int(scene_index),
        "scene_count": int(scene_count),
        "scene_range": meta.get("scene_range"),
        "timeline_crop": meta.get("timeline_crop"),
    }


def parse_package_pipe(raw: Any) -> dict[str, Any]:
    """Normalize a Collect pipe input to a dict with resolvable package identity."""
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        data = parsed if isinstance(parsed, dict) else {"package_rel": text, "output_rel": text}
    else:
        raise ValueError("package_pipe is missing; wire Load Package → Upscale Collect")

    package_rel = str(data.get("package_rel") or "").strip().replace("\\", "/")
    output_rel = str(data.get("output_rel") or "").strip().replace("\\", "/")
    identity = package_rel or output_rel
    if not identity:
        raise ValueError("package_pipe has no package_rel / output_rel")
    # Scene labels are display-only and must not be resolved as folders.
    identity = identity.split("#scene", 1)[0]
    package_rel = package_rel.split("#scene", 1)[0]
    output_rel = output_rel.split("#scene", 1)[0]
    return {
        "type": str(data.get("type") or "h3_package_pipe_v1"),
        "package_rel": package_rel or identity,
        "output_rel": output_rel or identity,
        "package_name": str(data.get("package_name") or Path(identity).name),
        "package_dir": str(data.get("package_dir") or ""),
        "scene_index": int(data.get("scene_index") or 0),
        "scene_count": int(data.get("scene_count") or 1),
        "scene_range": data.get("scene_range"),
        "timeline_crop": data.get("timeline_crop"),
    }


def resolve_package_dir(rel_or_name: str) -> Path:
    """Resolve a package relative path under stash root, else under output."""
    raw = (rel_or_name or "").strip().replace("\\", "/")
    if not raw or raw == NO_PACKAGE:
        raise FileNotFoundError("No package selected")
    if ".." in Path(raw).parts:
        raise ValueError(f"Invalid package path: {raw}")

    stash_candidate = (stash_root() / raw).resolve()
    try:
        stash_candidate.relative_to(stash_root())
    except ValueError as exc:
        raise ValueError(f"Package path escapes stash root: {raw}") from exc
    if is_package_dir(stash_candidate):
        return stash_candidate

    out_candidate = (output_root() / raw).resolve()
    try:
        out_candidate.relative_to(output_root())
    except ValueError as exc:
        raise ValueError(f"Package path escapes output directory: {raw}") from exc
    if is_package_dir(out_candidate):
        return out_candidate

    # Prefer stash path for error messages (legacy flat names).
    return stash_candidate


def is_package_dir(path: Path) -> bool:
    return path.is_dir() and (path / META_FILE).is_file() and (path / LATENT_FILE).is_file()


def list_packages() -> list[str]:
    """Flat legacy listing (immediate children only). Prefer selection_json for gallery."""
    root = stash_root()
    names: list[str] = []
    for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if is_package_dir(path):
            names.append(path.name)
    return names


def _noise_mask_members(latent: dict) -> tuple[list[torch.Tensor] | None, bool]:
    mask = latent.get("noise_mask")
    if mask is None:
        return None, False
    return extract_tensor(mask)


def parse_selection_json(raw: str) -> tuple[str, list[str]]:
    """Return (root, packages[]) from selection_json. packages are rel paths from stash root."""
    text = (raw or "").strip()
    if not text:
        return STASH_ROOT_NAME, []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection_json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("selection_json must be an object")
    packages = data.get("packages") or []
    if not isinstance(packages, list):
        raise ValueError("selection_json.packages must be a list")
    cleaned: list[str] = []
    for item in packages:
        rel = str(item or "").strip().replace("\\", "/")
        if not rel or ".." in Path(rel).parts:
            continue
        cleaned.append(rel)
    root = str(data.get("root") or STASH_ROOT_NAME).strip() or STASH_ROOT_NAME
    return root, cleaned


def parse_edit_json(raw: str) -> list[dict[str, int | str]]:
    """Return valid ordered timeline clips, preserving duplicate packages."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid edit_json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("edit_json must be an object")
    clips = data.get("clips") or []
    if not isinstance(clips, list):
        raise ValueError("edit_json.clips must be a list")

    cleaned: list[dict[str, int | str]] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("package") or "").strip().replace("\\", "/")
        if not rel or ".." in Path(rel).parts:
            continue
        frame_start = max(0, int(clip.get("in_frame") or 0))
        frame_end = max(frame_start + 1, int(clip.get("out_frame") or frame_start + 1))
        entry: dict[str, int | str] = {
            "package": rel,
            "in_frame": frame_start,
            "out_frame": frame_end,
        }
        media = str(clip.get("media") or "").strip().replace("\\", "/")
        if media and "/" not in media and ".." not in media:
            entry["media"] = media
        cleaned.append(entry)
    return cleaned


def load_entries(
    selection_json: str,
    edit_json: str,
    load_source: str,
) -> list[dict[str, int | str | None]]:
    """Build the ordered queue entries for the selected load source."""
    if load_source == LOAD_SOURCE_SELECTION:
        _, packages = parse_selection_json(selection_json)
        return [
            {"package": rel, "in_frame": None, "out_frame": None}
            for rel in packages
        ]
    if load_source in (LOAD_SOURCE_TIMELINE, LOAD_SOURCE_TIMELINE_CROPS):
        return parse_edit_json(edit_json)
    raise ValueError(f"Unknown load_source: {load_source!r}")


def crop_lq_package(
    latent: dict,
    positive: list,
    negative: list | None,
    meta: dict[str, Any],
    frame_start: int,
    frame_end: int,
) -> tuple[dict, list, list | None, dict[str, Any]]:
    """Crop a package to a timeline range, snapping outward to H3 token boundaries."""
    from .chunked import (
        pixel_frames_before_token,
        slice_av_latent,
        token_range_covering_pixel_frames,
        window_minimax_conditioning,
    )

    members, _ = extract_tensor(latent["samples"])
    video = members[0]
    if video.ndim != 5:
        raise ValueError(f"Expected video latent BxCxTxHxW, got {tuple(video.shape)}")
    latent_t = int(video.shape[2])
    total_frames = pixel_frames_before_token(latent_t)
    requested_start = max(0, min(total_frames - 1, int(frame_start)))
    requested_end = min(total_frames, max(requested_start + 1, int(frame_end)))
    t0, t1 = token_range_covering_pixel_frames(requested_start, requested_end, latent_t)
    if t1 <= t0:
        raise ValueError(
            f"Timeline crop [{frame_start}, {frame_end}) produced an empty latent range"
        )

    cropped_latent = slice_av_latent(latent, t0, t1)
    cropped_positive = window_minimax_conditioning(positive, t0, t1)
    cropped_negative = window_minimax_conditioning(negative, t0, t1)
    effective_start = pixel_frames_before_token(t0)
    effective_end = pixel_frames_before_token(t1)

    cropped_meta = dict(meta)
    cropped_members, _ = extract_tensor(cropped_latent["samples"])
    cropped_meta["video_shape"] = list(cropped_members[0].shape)
    cropped_meta["audio_shape"] = (
        list(cropped_members[1].shape) if len(cropped_members) > 1 else None
    )
    cropped_meta["frame_count"] = effective_end - effective_start
    cropped_meta["timeline_crop"] = {
        "fps": int(H3_FPS),
        "requested_in_frame": int(frame_start),
        "requested_out_frame": int(frame_end),
        "clamped_in_frame": requested_start,
        "clamped_out_frame": requested_end,
        "effective_in_frame": effective_start,
        "effective_out_frame": effective_end,
        "latent_t0": t0,
        "latent_t1": t1,
        "snapped_to_token_grid": (
            effective_start != requested_start or effective_end != requested_end
        ),
    }
    return cropped_latent, cropped_positive or [], cropped_negative, cropped_meta


def _latent_pixel_frame_count(latent: dict) -> int:
    from .chunked import pixel_frames_before_token

    members, _ = extract_tensor(latent["samples"])
    video = members[0]
    if video.ndim != 5:
        raise ValueError(f"Expected video latent BxCxTxHxW, got {tuple(video.shape)}")
    return pixel_frames_before_token(int(video.shape[2]))


def expand_load_chunks(
    selection_json: str,
    edit_json: str,
    load_source: str,
    index: int,
    *,
    chunk_mode: str = CHUNK_MODE_DIRECT,
    scene_threshold: float = 27.0,
    min_scene_seconds: float = 1.0,
    max_scene_seconds: float = 5.0,
) -> tuple[list[dict], list, list, list[dict[str, Any]], list[str], list[dict[str, Any]], int]:
    """Resolve one load entry and expand it into one or more cropped chunks."""
    from .stash_scenes import build_scene_chunk_ranges, require_preview_mp4

    entries = load_entries(selection_json, edit_json, load_source)
    if not entries:
        source_hint = (
            "Add clips to the Timeline tab."
            if load_source != LOAD_SOURCE_SELECTION
            else "Open Browse Saved Packages and Apply a selection."
        )
        raise FileNotFoundError(
            f"No packages available for {load_source}. {source_hint}"
        )

    n = len(entries)
    i = int(index) % n
    entry = entries[i]
    rel = str(entry["package"])
    latent, positive, negative, meta = load_lq_package(rel)
    if negative is None:
        negative = []

    total_frames = _latent_pixel_frame_count(latent)
    if load_source == LOAD_SOURCE_TIMELINE_CROPS:
        window_start = max(0, int(entry["in_frame"]))
        window_end = max(window_start + 1, int(entry["out_frame"]))
        window_end = min(total_frames, window_end)
        window_start = min(window_start, max(0, window_end - 1))
    else:
        window_start = 0
        window_end = total_frames

    package_dir = resolve_package_dir(rel)
    package_name = canonical_package_name(package_dir)
    base_meta = dict(meta)
    base_meta["load_source"] = load_source
    base_meta["load_index"] = i
    base_meta["load_count"] = n
    base_meta["chunk_mode"] = chunk_mode

    chunk_ranges: list[tuple[int, int]]
    used_soft_split = False
    if chunk_mode == CHUNK_MODE_DIRECT:
        chunk_ranges = [(window_start, window_end)]
    elif chunk_mode in (CHUNK_MODE_SCENES, CHUNK_MODE_SCENES_LEGACY):
        preview = require_preview_mp4(package_dir, package_name)
        chunk_ranges, used_soft_split = build_scene_chunk_ranges(
            preview,
            threshold=float(scene_threshold),
            min_scene_seconds=float(min_scene_seconds),
            max_scene_seconds=float(max_scene_seconds),
            fps=H3_FPS,
            window_start=window_start,
            window_end=window_end,
        )
    else:
        raise ValueError(f"Unknown chunk_mode: {chunk_mode!r}")

    if not chunk_ranges:
        chunk_ranges = [(window_start, max(window_start + 1, window_end))]

    scene_count = len(chunk_ranges)
    latents: list[dict] = []
    positives: list = []
    negatives: list = []
    metas: list[dict[str, Any]] = []
    names: list[str] = []
    pipes: list[dict[str, Any]] = []

    scenes_mode = chunk_mode in (CHUNK_MODE_SCENES, CHUNK_MODE_SCENES_LEGACY)
    for scene_index, (frame_start, frame_end) in enumerate(chunk_ranges):
        warnings: list[str] = []
        if scenes_mode:
            warnings.append(
                "Text prompt conditioning stays global; only keyframes/guides "
                "outside this scene window are dropped."
            )
            warnings.append("Cuts snap outward to the H3 temporal token grid.")
            if used_soft_split:
                warnings.append(
                    "At least one long scene was soft-split by max_scene_seconds "
                    "(no overlap lock between soft pieces)."
                )

        # Direct selection/timeline-order with a full window: skip a no-op crop.
        needs_crop = not (
            chunk_mode == CHUNK_MODE_DIRECT
            and frame_start == 0
            and frame_end >= total_frames
        )
        if needs_crop:
            c_latent, c_pos, c_neg, c_meta = crop_lq_package(
                latent,
                positive,
                negative,
                base_meta,
                frame_start,
                frame_end,
            )
            if c_neg is None:
                c_neg = []
            crop_info = c_meta.get("timeline_crop") or {}
            if crop_info.get("snapped_to_token_grid"):
                warnings.append(
                    "Requested frame range snapped to H3 token boundaries "
                    f"(effective {crop_info.get('effective_in_frame')}–"
                    f"{crop_info.get('effective_out_frame')}f)."
                )
        else:
            c_latent, c_pos, c_neg, c_meta = latent, positive, negative, dict(base_meta)

        c_meta = dict(c_meta)
        c_meta["chunk_mode"] = chunk_mode
        c_meta["scene_index"] = scene_index
        c_meta["scene_count"] = scene_count
        c_meta["scene_range"] = {
            "in_frame": int(frame_start),
            "out_frame": int(frame_end),
            "soft_split_applied": bool(used_soft_split),
        }
        if warnings:
            c_meta["warnings"] = warnings
        chunk_name = canonical_package_name(
            package_dir,
            scene_index=scene_index,
            scene_count=scene_count,
        )
        latents.append(c_latent)
        positives.append(c_pos)
        negatives.append(c_neg)
        metas.append(c_meta)
        names.append(chunk_name)
        pipes.append(
            build_package_pipe(
                package_dir,
                c_meta,
                scene_index=scene_index,
                scene_count=scene_count,
            )
        )

    return latents, positives, negatives, metas, names, pipes, n


def save_lq_package(
    latent: dict,
    positive: list,
    *,
    negative: list | None = None,
    filename_prefix: str = "h3_lq",
    project: str = "",
    note: str = "",
    seed: int | None = None,
    prompt: str = "",
    images=None,
    audio=None,
    fps: float = H3_FPS,
) -> tuple[Path, dict[str, Any]]:
    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')
    if positive is None:
        raise ValueError("positive CONDITIONING is required to save a package")

    members, was_nested = extract_tensor(latent["samples"])
    if len(members) < 1:
        raise ValueError("LATENT samples are empty")

    tensors: dict[str, torch.Tensor] = {
        "video": members[0].detach().cpu().contiguous(),
    }
    if len(members) > 1:
        tensors["audio"] = members[1].detach().cpu().contiguous()
    for index, extra in enumerate(members[2:]):
        tensors[f"extra_{index}"] = extra.detach().cpu().contiguous()

    mask_members, mask_nested = _noise_mask_members(latent)
    if mask_members:
        tensors["noise_mask_video"] = mask_members[0].detach().cpu().contiguous()
        if len(mask_members) > 1:
            tensors["noise_mask_audio"] = mask_members[1].detach().cpu().contiguous()

    package_dir = _unique_package_dir(filename_prefix, project=project)
    rel = package_rel_path(package_dir)
    metadata = {
        "format": PACKAGE_FORMAT,
        "was_nested": str(bool(was_nested)).lower(),
        "member_count": str(len(members)),
        "has_audio": str("audio" in tensors).lower(),
        "has_noise_mask": str(mask_members is not None).lower(),
        "noise_mask_nested": str(bool(mask_nested)).lower(),
    }
    save_file(tensors, str(package_dir / LATENT_FILE), metadata=metadata)

    torch.save(
        {
            "positive": positive,
            "negative": negative,
        },
        package_dir / COND_FILE,
    )

    project_rel = _sanitize_project(project)
    meta: dict[str, Any] = {
        "format": PACKAGE_FORMAT,
        "package_name": package_dir.name,
        "package_rel": rel,
        "project": project_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "was_nested": bool(was_nested),
        "member_count": len(members),
        "video_shape": list(tensors["video"].shape),
        "audio_shape": list(tensors["audio"].shape) if "audio" in tensors else None,
        "has_noise_mask": mask_members is not None,
        "note": note or "",
        "prompt": prompt or "",
        "has_preview": False,
        "has_thumb": False,
        "thumb": None,
        "preview": None,
    }
    if seed is not None:
        meta["seed"] = int(seed)

    if images is not None:
        try:
            preview_meta = write_preview_from_images(
                package_dir,
                images,
                audio=audio,
                fps=fps,
            )
            meta.update(preview_meta)
        except Exception as exc:
            print(f"[H3 Packages] preview encode failed: {exc}")

    (package_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return package_dir, meta


def load_lq_package(package_rel: str) -> tuple[dict, list, list | None, dict[str, Any]]:
    package_dir = resolve_package_dir(package_rel)
    meta_path = package_dir / META_FILE
    latent_path = package_dir / LATENT_FILE
    cond_path = package_dir / COND_FILE
    if not meta_path.is_file() or not latent_path.is_file():
        raise FileNotFoundError(f"Incomplete package: {package_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != PACKAGE_FORMAT:
        raise ValueError(
            f"Unsupported package format {meta.get('format')!r}; "
            f"expected {PACKAGE_FORMAT}"
        )

    tensors = load_file(str(latent_path), device="cpu")
    if "video" not in tensors:
        raise KeyError(f"{latent_path} is missing the video tensor")
    members = [tensors["video"]]
    if "audio" in tensors:
        members.append(tensors["audio"])
    extras = sorted(
        (key for key in tensors if key.startswith("extra_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    for key in extras:
        members.append(tensors[key])

    was_nested = bool(meta.get("was_nested", len(members) > 1))
    latent: dict[str, Any] = {
        "samples": wrap_tensor(members, was_nested=was_nested),
    }

    if "noise_mask_video" in tensors:
        mask_members = [tensors["noise_mask_video"]]
        if "noise_mask_audio" in tensors:
            mask_members.append(tensors["noise_mask_audio"])
        latent["noise_mask"] = wrap_tensor(
            mask_members,
            was_nested=bool(meta.get("has_noise_mask")) and len(mask_members) > 1,
        )

    if not cond_path.is_file():
        raise FileNotFoundError(f"Missing conditioning file: {cond_path}")
    cond_blob = torch.load(cond_path, map_location="cpu", weights_only=False)
    positive = cond_blob.get("positive")
    negative = cond_blob.get("negative")
    if positive is None:
        raise ValueError(f"Package {package_rel} has no positive conditioning")
    meta.setdefault("package_rel", package_rel_path(package_dir))
    return latent, positive, negative, meta


class MiniMaxH3LQPackageSave:
    """Persist Sampler-1 latent + conditioning for deferred HQ upscale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "positive": ("CONDITIONING",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "h3_lq",
                        "tooltip": (
                            "Package folder name under output/h3_lq_stash/[project]/. "
                            "A numeric suffix is added if the name already exists."
                        ),
                    },
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "project": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional project subfolder under the packages root "
                            "(e.g. my_film or my_film/act1)."
                        ),
                    },
                ),
                "note": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Freeform note stored in meta.json only.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional prompt text for later browsing (meta only).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        # Plain integer field: opt out of Comfy's seed control widget.
                        "control_after_generate": False,
                        "tooltip": "Optional seed recorded in meta.json. -1 = omit.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional VAE-decoded LQ frames. When connected, writes "
                            "preview.mp4 + thumb.jpg into the package for the gallery."
                        ),
                    },
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional audio muxed into preview.mp4 when images are set.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("latent", "positive", "negative", "package_path")
    FUNCTION = "save"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Save the clean Sampler-1 NestedTensor latent plus positive/negative "
        "conditioning under output/h3_lq_stash/[project]/<name>/ for overnight HQ "
        "Combined + Sampler-2. Optionally wire IMAGE (+ AUDIO) for gallery preview."
    )

    def save(
        self,
        samples,
        positive,
        filename_prefix="h3_lq",
        negative=None,
        project="",
        note="",
        prompt="",
        seed=-1,
        images=None,
        audio=None,
    ):
        seed_value = None if int(seed) < 0 else int(seed)
        package_dir, meta = save_lq_package(
            samples,
            positive,
            negative=negative,
            filename_prefix=filename_prefix,
            project=project,
            note=note,
            seed=seed_value,
            prompt=prompt,
            images=images,
            audio=audio,
            fps=H3_FPS,
        )
        preview_note = ""
        if meta.get("has_preview"):
            preview_note = f", preview={meta.get('preview')}"
        elif images is not None:
            preview_note = ", preview=failed/thumb-only"
        print(
            f"Saved H3 package {meta.get('package_rel', package_dir.name)} "
            f"(video={tuple(meta['video_shape'])}, "
            f"audio={tuple(meta['audio_shape']) if meta['audio_shape'] else None}"
            f"{preview_note})"
        )
        return (samples, positive, negative, str(package_dir.resolve()))


class MiniMaxH3LQPackageLoad:
    """Load a package from a multi-select selection_json + queue index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Filled by the Browse Saved Packages modal. "
                            '{"version":1,"root":"h3_lq_stash","packages":["rel/..."]}'
                        ),
                    },
                ),
                "load_source": (
                    LOAD_SOURCES,
                    {
                        "default": LOAD_SOURCE_SELECTION,
                        "tooltip": (
                            "selection list: use the Browse Selected list. "
                            "timeline order: use only timeline clips in edit order, "
                            "ignoring trims. timeline order and crops: also crop each "
                            "loaded latent and conditioning to its timeline IN/OUT range."
                        ),
                    },
                ),
                "chunk_mode": (
                    CHUNK_MODES,
                    {
                        "default": CHUNK_MODE_DIRECT,
                        "tooltip": (
                            "direct load: one latent per queued entry (returned as a "
                            "1-element list). scene aware chunks (experimental): detect cuts on "
                            "preview.mp4 and emit one list item per scene so Combined/"
                            "Sampler fan out in a single queue job. Requires preview.mp4."
                        ),
                    },
                ),
                "scene_threshold": (
                    "FLOAT",
                    {
                        "default": 27.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": (
                            "PySceneDetect ContentDetector threshold. Lower = more cuts."
                        ),
                    },
                ),
                "min_scene_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "Minimum scene length; shorter detections are suppressed.",
                    },
                ),
                "max_scene_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.5,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": (
                            "Soft-split fallback: scenes longer than this are cut into "
                            "equal pieces with no overlap lock."
                        ),
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "control_after_generate": False,
                        "tooltip": (
                            "Which entry from the active load source to load. Always "
                            "resolved as index % selected_count for safe queue batching."
                        ),
                    },
                ),
                "index_mode": (
                    ["increment", "fixed"],
                    {
                        "default": "increment",
                        "tooltip": (
                            "increment: index advances by 1 (wrapping) after each "
                            "queued job. fixed: index stays put."
                        ),
                    },
                ),
                "edit_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Filled by the Timeline tab. Drives timeline load modes."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "LATENT",
        "CONDITIONING",
        "CONDITIONING",
        "STRING",
        "STRING",
        "INT",
        PACKAGE_PIPE_TYPE,
    )
    RETURN_NAMES = (
        "latent",
        "positive",
        "negative",
        "meta",
        "package_name",
        "selected_count",
        "package_pipe",
    )
    # Class-level: direct mode still returns length-1 lists.
    OUTPUT_IS_LIST = (True, True, True, True, True, False, True)
    FUNCTION = "load"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Load a saved Sampler-1 package from the gallery selection or timeline. "
        "Timeline crops trim the latent and conditioning to each clip's IN/OUT range. "
        "Scene-aware chunks split on preview.mp4 cuts and fan out as ComfyUI lists. "
        "package_pipe carries output-relative origin for Upscale Collect."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        selection_json,
        load_source=LOAD_SOURCE_SELECTION,
        chunk_mode=CHUNK_MODE_DIRECT,
        scene_threshold=27.0,
        min_scene_seconds=1.0,
        max_scene_seconds=5.0,
        index=0,
        index_mode="increment",
        edit_json="",
    ):
        try:
            entries = load_entries(selection_json, edit_json, load_source)
        except Exception:
            return float("nan")
        if not entries:
            return float("nan")
        n = len(entries)
        i = int(index) % n
        entry = entries[i]
        rel = str(entry["package"])
        try:
            package_dir = resolve_package_dir(rel)
        except Exception:
            return float("nan")
        meta_path = package_dir / META_FILE
        if not meta_path.is_file():
            return float("nan")
        crop_key = ""
        if load_source == LOAD_SOURCE_TIMELINE_CROPS:
            crop_key = f":{entry['in_frame']}:{entry['out_frame']}"
        latent_path = package_dir / LATENT_FILE
        cond_path = package_dir / COND_FILE
        preview_path = package_dir / PREVIEW_FILE
        mtimes = (
            meta_path.stat().st_mtime_ns,
            latent_path.stat().st_mtime_ns if latent_path.is_file() else 0,
            cond_path.stat().st_mtime_ns if cond_path.is_file() else 0,
            preview_path.stat().st_mtime_ns if preview_path.is_file() else 0,
        )
        scene_key = ""
        if chunk_mode in (CHUNK_MODE_SCENES, CHUNK_MODE_SCENES_LEGACY):
            scene_key = (
                f":{float(scene_threshold):.3f}:{float(min_scene_seconds):.3f}"
                f":{float(max_scene_seconds):.3f}"
            )
        return (
            f"{load_source}:{chunk_mode}:{rel}:{mtimes}:{i}:{n}:{index_mode}"
            f"{crop_key}{scene_key}"
        )

    def load(
        self,
        selection_json,
        load_source=LOAD_SOURCE_SELECTION,
        chunk_mode=CHUNK_MODE_DIRECT,
        scene_threshold=27.0,
        min_scene_seconds=1.0,
        max_scene_seconds=5.0,
        index=0,
        index_mode="increment",
        edit_json="",
    ):
        _ = index_mode
        latents, positives, negatives, metas, names, pipes, n = expand_load_chunks(
            selection_json,
            edit_json,
            load_source,
            index,
            chunk_mode=chunk_mode,
            scene_threshold=scene_threshold,
            min_scene_seconds=min_scene_seconds,
            max_scene_seconds=max_scene_seconds,
        )
        scene_count = len(latents)
        package = names[0].split("#scene")[0] if names else "?"
        print(
            f"Package Load source={load_source} chunk_mode={chunk_mode} "
            f"index={int(index) % max(1, n)}/{n} scenes={scene_count} -> {package}"
        )
        return (
            latents,
            positives,
            negatives,
            [json.dumps(m, indent=2) for m in metas],
            names,
            int(n),
            pipes,
        )
