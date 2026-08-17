"""Scene detection helpers for package scene-aware chunk loads."""

from __future__ import annotations

from pathlib import Path

from .stash_preview import PREVIEW_FILE

# Defaults match the Load node widgets.
DEFAULT_SCENE_THRESHOLD = 27.0
DEFAULT_MIN_SCENE_SECONDS = 1.0
DEFAULT_MAX_SCENE_SECONDS = 5.0


def require_preview_mp4(package_dir: Path, package_rel: str) -> Path:
    """Return preview.mp4 or raise a clear FileNotFoundError."""
    preview = Path(package_dir) / PREVIEW_FILE
    if preview.is_file():
        return preview.resolve()
    raise FileNotFoundError(
        f"Scene-aware chunks require preview.mp4 for package {package_rel!r}, "
        f"but it was not found at {preview}. Re-save the package with IMAGE "
        f"(and optional AUDIO) wired into MiniMax H3 Save Package."
    )


def detect_scene_frame_ranges(
    preview_path: Path,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_frames: int = 24,
) -> list[tuple[int, int]]:
    """Detect [in_frame, out_frame) ranges on a preview mp4 via ContentDetector.

    Frame indices are 0-based at the container fps (H3 previews are 24fps).
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise ImportError(
            "Scene-aware chunks need the scenedetect package "
            "(PySceneDetect). Install it into the ComfyUI Python env, e.g. "
            "`python -m pip install scenedetect`."
        ) from exc

    path = Path(preview_path)
    if not path.is_file():
        raise FileNotFoundError(f"Preview video not found: {path}")

    min_len = max(1, int(min_scene_frames))
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=float(threshold), min_scene_len=min_len)
    )
    manager.detect_scenes(video)
    scenes = manager.get_scene_list(start_in_scene=True)
    if not scenes:
        # Whole file as one scene when no cuts fire.
        total = int(video.duration.get_frames())
        if total < 1:
            raise ValueError(f"Preview has no frames: {path}")
        return [(0, total)]

    ranges: list[tuple[int, int]] = []
    for start_tc, end_tc in scenes:
        start = int(start_tc.get_frames())
        end = int(end_tc.get_frames())
        if end > start:
            ranges.append((start, end))
    if not ranges:
        total = int(video.duration.get_frames())
        return [(0, max(1, total))]
    return ranges


def split_ranges_by_max_frames(
    ranges: list[tuple[int, int]],
    max_frames: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Split any range longer than max_frames into equal-ish pieces.

    Returns (ranges, used_soft_split). Soft splits have no overlap.
    """
    max_frames = max(1, int(max_frames))
    out: list[tuple[int, int]] = []
    used_soft_split = False
    for start, end in ranges:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        length = end - start
        if length <= max_frames:
            out.append((start, end))
            continue
        used_soft_split = True
        n_parts = (length + max_frames - 1) // max_frames
        # Prefer nearly equal chunk lengths rather than max_frames + leftover.
        base = length // n_parts
        rem = length % n_parts
        cursor = start
        for i in range(n_parts):
            piece = base + (1 if i < rem else 0)
            nxt = cursor + max(1, piece)
            if i == n_parts - 1:
                nxt = end
            out.append((cursor, nxt))
            cursor = nxt
    return out, used_soft_split


def intersect_ranges(
    ranges: list[tuple[int, int]],
    window_start: int,
    window_end: int,
) -> list[tuple[int, int]]:
    """Intersect scene ranges with an active [window_start, window_end) crop."""
    w0 = int(window_start)
    w1 = int(window_end)
    if w1 <= w0:
        return []
    out: list[tuple[int, int]] = []
    for start, end in ranges:
        a = max(w0, int(start))
        b = min(w1, int(end))
        if b > a:
            out.append((a, b))
    return out


def build_scene_chunk_ranges(
    preview_path: Path,
    *,
    threshold: float,
    min_scene_seconds: float,
    max_scene_seconds: float,
    fps: float,
    window_start: int | None = None,
    window_end: int | None = None,
) -> tuple[list[tuple[int, int]], bool]:
    """Detect, window-intersect, and max-duration-split scene ranges."""
    rate = max(1.0, float(fps))
    min_frames = max(1, int(round(float(min_scene_seconds) * rate)))
    max_frames = max(1, int(round(float(max_scene_seconds) * rate)))
    ranges = detect_scene_frame_ranges(
        preview_path,
        threshold=threshold,
        min_scene_frames=min_frames,
    )
    if window_start is not None and window_end is not None:
        ranges = intersect_ranges(ranges, window_start, window_end)
        if not ranges:
            # Window empty vs detections: keep the window as a single chunk.
            ranges = [(int(window_start), int(window_end))]
    return split_ranges_by_max_frames(ranges, max_frames)
