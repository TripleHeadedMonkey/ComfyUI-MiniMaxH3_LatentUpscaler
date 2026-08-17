"""Encode Comfy IMAGE (+ optional AUDIO) into gallery preview.mp4 / thumb.jpg."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

LOG = logging.getLogger("h3_lq_stash.preview")

PREVIEW_FILE = "preview.mp4"
THUMB_FILE = "thumb.jpg"


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Comfy IMAGE [N,H,W,C] float 0-1 -> uint8 NHWC RGB."""
    arr = images.detach().cpu().float().numpy()
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected IMAGE [N,H,W,C], got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _write_thumb(frames: np.ndarray, thumb_path: Path, *, max_width: int = 480) -> None:
    first = frames[0]
    img = Image.fromarray(first, mode="RGB")
    if img.width > max_width:
        height = max(1, round(img.height * max_width / img.width))
        img = img.resize((max_width, height), Image.Resampling.LANCZOS)
    img.save(thumb_path, format="JPEG", quality=85)


def _audio_to_wav(audio: dict[str, Any], wav_path: Path) -> tuple[Path, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("AUDIO waveform must be a torch.Tensor")
    # [B, C, L] -> [C, L]
    wave = waveform.detach().cpu().float()
    if wave.ndim == 3:
        wave = wave[0]
    if wave.ndim != 2:
        raise ValueError(f"Unexpected waveform shape {tuple(wave.shape)}")
    # Limit to stereo
    if wave.shape[0] > 2:
        wave = wave[:2]
    try:
        import torchaudio

        torchaudio.save(str(wav_path), wave, sample_rate)
    except Exception:
        import wave as wave_mod

        pcm = (wave.clamp(-1, 1) * 32767.0).short().numpy()
        interleaved = pcm.T.reshape(-1)
        with wave_mod.open(str(wav_path), "wb") as handle:
            handle.setnchannels(pcm.shape[0])
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(interleaved.tobytes())
    return wav_path, sample_rate


def encode_images_to_mp4(
    dest: Path,
    images: torch.Tensor,
    *,
    audio: dict[str, Any] | None = None,
    fps: float = 24.0,
    crf: int = 18,
    preset: str = "medium",
) -> dict[str, Any]:
    """Encode IMAGE (+ optional AUDIO) to dest as H.264 MP4. Returns encode stats."""
    frames = _images_to_uint8(images)
    n, height, width, _ = frames.shape
    if n < 1:
        raise ValueError("No frames to encode")

    fps = float(fps) if fps and fps > 0 else 24.0
    crf = max(0, min(51, int(crf)))
    preset = str(preset or "medium")
    enc_w = width - (width % 2)
    enc_h = height - (height % 2)
    if enc_w != width or enc_h != height:
        resized = []
        for i in range(n):
            img = Image.fromarray(frames[i], mode="RGB").resize(
                (enc_w, enc_h), Image.Resampling.LANCZOS
            )
            resized.append(np.asarray(img))
        frames = np.stack(resized, axis=0)
        width, height = enc_w, enc_h

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        _encode_with_ffmpeg(
            ffmpeg,
            frames,
            dest,
            fps=fps,
            audio=audio,
            crf=crf,
            preset=preset,
        )
    else:
        _encode_with_pyav(frames, dest, fps=fps, audio=audio, crf=crf, preset=preset)
    if not dest.is_file() or dest.stat().st_size < 32:
        raise RuntimeError(f"MP4 encode produced no file: {dest}")
    return {
        "path": dest,
        "fps": fps,
        "frame_count": n,
        "width": width,
        "height": height,
        "has_audio": bool(audio is not None and "waveform" in audio),
    }


def write_preview_from_images(
    package_dir: Path,
    images: torch.Tensor,
    *,
    audio: dict[str, Any] | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Write preview.mp4 + thumb.jpg into package_dir. Returns meta fields."""
    frames = _images_to_uint8(images)
    n, height, width, _ = frames.shape
    if n < 1:
        raise ValueError("No frames to encode for LQ preview")

    thumb_path = package_dir / THUMB_FILE
    preview_path = package_dir / PREVIEW_FILE
    _write_thumb(frames, thumb_path)

    try:
        encode_images_to_mp4(
            preview_path,
            images,
            audio=audio,
            fps=fps,
            crf=20,
            preset="veryfast",
        )
    except Exception as exc:
        LOG.warning("preview encode failed (%s); keeping thumb only", exc)
        return {
            "has_preview": False,
            "has_thumb": True,
            "thumb": THUMB_FILE,
            "preview": None,
            "fps": float(fps) if fps and fps > 0 else 24.0,
            "frame_count": n,
        }

    return {
        "has_preview": preview_path.is_file(),
        "has_thumb": thumb_path.is_file(),
        "thumb": THUMB_FILE if thumb_path.is_file() else None,
        "preview": PREVIEW_FILE if preview_path.is_file() else None,
        "fps": float(fps) if fps and fps > 0 else 24.0,
        "frame_count": n,
    }


def _encode_with_ffmpeg(
    ffmpeg: str,
    frames: np.ndarray,
    preview_path: Path,
    *,
    fps: float,
    audio: dict[str, Any] | None,
    crf: int = 20,
    preset: str = "veryfast",
) -> None:
    n, height, width, _ = frames.shape
    with tempfile.TemporaryDirectory(prefix="h3_lq_preview_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGB").save(tmp_dir / f"frame_{i:06d}.png")

        wav_path = None
        if audio is not None and "waveform" in audio:
            wav_path = tmp_dir / "audio.wav"
            _audio_to_wav(audio, wav_path)

        out_tmp = tmp_dir / "out.mp4"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
        ]
        if wav_path is not None and wav_path.is_file():
            cmd += ["-i", str(wav_path)]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            str(preset or "veryfast"),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-g",
            "12",
            "-movflags",
            "+faststart",
        ]
        if wav_path is not None and wav_path.is_file():
            cmd += ["-c:a", "aac", "-ac", "2", "-shortest"]
        else:
            cmd += ["-an"]
        cmd.append(str(out_tmp))

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0 or not out_tmp.is_file():
            raise RuntimeError(
                f"ffmpeg preview failed ({proc.returncode}): {(proc.stderr or '')[-1500:]}"
            )
        if preview_path.exists():
            preview_path.unlink()
        shutil.copy2(out_tmp, preview_path)


def _encode_with_pyav(
    frames: np.ndarray,
    preview_path: Path,
    *,
    fps: float,
    audio: dict[str, Any] | None,
    crf: int = 20,
    preset: str = "veryfast",
) -> None:
    import av

    n, height, width, _ = frames.shape
    container = av.open(str(preview_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(int(crf)), "preset": str(preset or "veryfast")}

    for frame in frames:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None and "waveform" in audio:
        LOG.info("PyAV fallback wrote video-only preview (audio skipped)")

    container.close()
