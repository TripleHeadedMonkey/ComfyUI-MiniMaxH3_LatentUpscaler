"""Chunked MiniMax H3 pass-2: temporal slice → upscale/re-noise → sample → decode → stitch."""

from __future__ import annotations

import gc
import importlib
import logging

import torch

import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_EmptyNoise, Noise_RandomNoise

from .utils import (
    UPSCALE_METHODS,
    add_noise_nested_latent,
    extract_tensor,
    finalize_latent_for_handoff,
    lock_audio_stream_mask,
    upscale_minimax_conditioning,
    upscale_nested_latent,
    wrap_tensor,
)

_LOG = logging.getLogger("minimax_h3_latent_upscaler")
FPS = 24.0
AUDIO_HZ = 40.0


def approx_frames_from_latent_t(latent_t: int) -> int:
    """H3 encode grid: frames ≈ 5 + ((T-2)/5)*17 at 24 fps (empty-latent creation formula)."""
    t = max(2, int(latent_t))
    if t <= 2:
        return 5
    return 5 + ((t - 2) // 5) * 17


def map_audio_range(t0: int, t1: int, video_t: int, audio_t: int) -> tuple[int, int]:
    """Map video latent [t0, t1) onto audio latent indices."""
    if video_t <= 0:
        return 0, 0
    ta0 = int(round(t0 / video_t * audio_t))
    ta1 = int(round(t1 / video_t * audio_t))
    ta0 = max(0, min(audio_t, ta0))
    ta1 = max(ta0, min(audio_t, ta1))
    if ta1 <= ta0 and audio_t > 0:
        ta1 = min(audio_t, ta0 + 1)
    return ta0, ta1


def pixel_frames_before_token(latent_t: int) -> int:
    """Pixel frames covered by video latent tokens [0, latent_t) on the H3 1-4-4-4-4 grid."""
    t = max(0, int(latent_t))
    return int(sum(FRAME_PER_TOKEN[k % 5] for k in range(t)))


def token_range_covering_pixel_frames(
    frame_start: int,
    frame_end: int,
    latent_t: int,
) -> tuple[int, int]:
    """Latent-token range covering pixel frames [frame_start, frame_end).

    H3 temporal tokens represent 1/4/4/4/4 pixel frames, so cuts inside a token
    cannot be represented exactly. A sliced latent restarts that pattern at token
    zero, so start also snaps backward to a five-token cycle boundary. End snaps
    forward to a token boundary, preserving every requested frame without changing
    the temporal phase seen by the model.
    """
    latent_t = max(0, int(latent_t))
    if latent_t <= 0:
        return 0, 0
    total_frames = pixel_frames_before_token(latent_t)
    frame_start = max(0, min(total_frames - 1, int(frame_start)))
    frame_end = min(total_frames, max(frame_start + 1, int(frame_end)))

    t0 = 0
    while t0 < latent_t and pixel_frames_before_token(t0 + 1) <= frame_start:
        t0 += 1
    t0 -= t0 % len(FRAME_PER_TOKEN)

    t1 = t0 + 1
    while t1 < latent_t and pixel_frames_before_token(t1) < frame_end:
        t1 += 1
    return t0, min(latent_t, max(t0 + 1, t1))


def snap_h3_context_tokens(t: int) -> int:
    """Largest valid H3 continuation prefix in latent tokens: 0 or 2+5k (5/22/39/... frames)."""
    t = int(t)
    if t < 2:
        return 0
    return 2 + 5 * ((t - 2) // 5)


def _enable_h3_av_mask_compat() -> bool:
    """Use Motion-Context-MultiRef PR #15375 hooks when that pack is installed."""
    try:
        compat = importlib.import_module("ComfyUI-H3-Motion-Context-MultiRef.h3_compat")
        compat.ensure_existing_video_compat()
        return True
    except Exception as exc:
        _LOG.info("H3 AV-mask compat not loaded (%s); sampler inpaint blend still freezes the prefix.", exc)
        return False


def _crop_audio_latent(audio: torch.Tensor | None, keep: int) -> torch.Tensor | None:
    if audio is None or not isinstance(audio, torch.Tensor):
        return None
    keep = max(0, min(int(keep), int(audio.shape[-1])))
    if keep <= 0:
        return None
    if keep >= int(audio.shape[-1]):
        return audio
    return audio[..., :keep].contiguous()


def _window_keyframe(kf: dict, f0: int, f1: int) -> dict | None:
    """Keep a guide only if it starts inside this pixel-frame window; remap index."""
    if "resolved_frame_index" not in kf:
        return dict(kf)
    idx = int(kf["resolved_frame_index"])
    if idx < f0 or idx >= f1:
        return None
    out = dict(kf)
    out["resolved_frame_index"] = idx - f0
    remain_frames = max(1, f1 - idx)
    z = out.get("latent")
    if isinstance(z, torch.Tensor) and z.ndim == 5:
        max_t = 0
        covered = 0
        for k in range(int(z.shape[2])):
            nxt = covered + FRAME_PER_TOKEN[k % 5]
            if nxt > remain_frames and k > 0:
                break
            covered = nxt
            max_t = k + 1
        if max_t <= 0:
            return None
        if max_t < int(z.shape[2]):
            out["latent"] = z[:, :, :max_t].contiguous()
    max_audio = max(1, int(round(remain_frames * AUDIO_HZ / FPS)))
    cropped = _crop_audio_latent(out.get("audio_latent"), max_audio)
    if cropped is None:
        out.pop("audio_latent", None)
    else:
        out["audio_latent"] = cropped
    if out.get("latent") is None and out.get("audio_latent") is None:
        return None
    return out


def _window_ref_block(blk: dict) -> dict | None:
    """Keep identity image/video refs; drop standalone audio refs (replay dialogue on every chunk)."""
    kind = blk.get("kind")
    if kind == "audio":
        return None
    return dict(blk)


def window_minimax_conditioning(
    conditioning: list | None,
    t0: int,
    t1: int,
) -> list | None:
    """Clone CONDITIONING onto this chunk's timeline: remap in-window keyframes, drop off-window guides."""
    if conditioning is None:
        return None
    f0 = pixel_frames_before_token(t0)
    f1 = pixel_frames_before_token(t1)
    out: list = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            out.append(entry)
            continue
        emb, meta = entry[0], entry[1]
        new_meta = meta.copy()
        refs = meta.get("minimax_refs")
        if refs is not None:
            kept_refs = []
            for blk in refs:
                if not isinstance(blk, dict):
                    continue
                w = _window_ref_block(blk)
                if w is not None:
                    kept_refs.append(w)
            new_meta["minimax_refs"] = kept_refs
        keyframes = meta.get("minimax_keyframes")
        if keyframes is not None:
            kept_kf = []
            for kf in keyframes:
                if not isinstance(kf, dict):
                    continue
                w = _window_keyframe(kf, f0, f1)
                if w is not None:
                    kept_kf.append(w)
            new_meta["minimax_keyframes"] = kept_kf
        out.append([emb, new_meta])
    return out


def build_av_noise_mask(video: torch.Tensor, audio: torch.Tensor | None, v_protect: int, a_protect: int):
    """1 = denoise, 0 = preserve. Video [B,1,T,H,W], audio [B,1,2,Ta]."""
    v_protect = max(0, min(int(v_protect), int(video.shape[2])))
    video_mask = torch.ones(
        (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
        device=video.device,
        dtype=torch.float32,
    )
    if v_protect > 0:
        video_mask[:, :, :v_protect] = 0.0
    if audio is None:
        return video_mask
    a_protect = max(0, min(int(a_protect), int(audio.shape[-1])))
    audio_mask = torch.ones(
        (audio.shape[0], 1, audio.shape[2], audio.shape[3]),
        device=audio.device,
        dtype=torch.float32,
    )
    if a_protect > 0:
        audio_mask[..., :a_protect] = 0.0
    return wrap_tensor([video_mask, audio_mask], was_nested=True)


def paste_protected_prefix(current: dict, previous: dict, v_protect: int, a_protect: int) -> dict:
    """Copy previous sampled HD tail into the current prefix (Generated AV Masked Context)."""
    cur_m, cur_nested = extract_tensor(current["samples"])
    prev_m, _ = extract_tensor(previous["samples"])
    video = cur_m[0]
    prev_v = prev_m[0]
    v_protect = max(0, min(int(v_protect), int(video.shape[2]) - 1, int(prev_v.shape[2])))
    if v_protect <= 0:
        return current
    if tuple(prev_v.shape[1:2] + prev_v.shape[3:]) != tuple(video.shape[1:2] + video.shape[3:]):
        raise ValueError(
            "Chunk continuation geometry mismatch: previous %s vs current %s"
            % (tuple(prev_v.shape), tuple(video.shape))
        )
    out_video = video.clone()
    src = prev_v[:, :, -v_protect:].to(device=out_video.device, dtype=out_video.dtype)
    out_video[:, :, :v_protect] = src
    out_members = [out_video]
    a_keep = 0
    audio = None
    if cur_nested and len(cur_m) >= 2 and len(prev_m) >= 2:
        audio = cur_m[1].clone()
        prev_a = prev_m[1]
        a_keep = max(0, min(int(a_protect), int(audio.shape[-1]) - 1, int(prev_a.shape[-1])))
        if a_keep > 0:
            audio[..., :a_keep] = prev_a[..., -a_keep:].to(device=audio.device, dtype=audio.dtype)
        out_members.append(audio)
        out_members.extend(cur_m[2:])
    out = current.copy()
    out["samples"] = wrap_tensor(out_members, was_nested=cur_nested)
    out["noise_mask"] = build_av_noise_mask(out_video, audio if len(out_members) > 1 else None, v_protect, a_keep)
    return out


def iter_temporal_windows(
    video_t: int,
    chunk_t: int,
    overlap_t: int,
) -> list[tuple[int, int]]:
    """Return [t0, t1) windows covering [0, video_t) with stride chunk-overlap."""
    if video_t <= 0:
        return []
    chunk_t = max(1, int(chunk_t))
    overlap_t = max(0, min(int(overlap_t), chunk_t - 1))
    stride = max(1, chunk_t - overlap_t)

    windows: list[tuple[int, int]] = []
    t0 = 0
    while t0 < video_t:
        t1 = min(video_t, t0 + chunk_t)
        # Pad last window backward so it is at least overlap+1 when possible.
        if t1 - t0 < overlap_t + 1 and t1 == video_t and video_t > overlap_t:
            t0 = max(0, video_t - max(overlap_t + 1, min(chunk_t, video_t)))
            t1 = video_t
            if windows and windows[-1][0] == t0 and windows[-1][1] == t1:
                break
        windows.append((t0, t1))
        if t1 >= video_t:
            break
        t0 += stride
    return windows


def slice_av_latent(latent: dict, t0: int, t1: int) -> dict:
    """Slice NestedTensor video on T and audio on Ta (fractional map)."""
    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')
    members, was_nested = extract_tensor(latent["samples"])
    video = members[0]
    if video.ndim != 5:
        raise ValueError(f"Expected video latent BxCxTxHxW, got {tuple(video.shape)}")
    video_t = video.shape[2]
    t0 = max(0, min(video_t, int(t0)))
    t1 = max(t0, min(video_t, int(t1)))
    out_members = [video[:, :, t0:t1].contiguous()]
    if was_nested and len(members) >= 2:
        audio = members[1]
        audio_t = audio.shape[-1]
        ta0, ta1 = map_audio_range(t0, t1, video_t, audio_t)
        out_members.append(audio[..., ta0:ta1].contiguous())
        out_members.extend(members[2:])
    out = latent.copy()
    out["samples"] = wrap_tensor(out_members, was_nested=was_nested)
    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        mask_members, mask_nested = extract_tensor(noise_mask)
        mask_out = [mask_members[0][:, :, t0:t1].contiguous()]
        if mask_nested and len(mask_members) >= 2:
            ma_t = mask_members[1].shape[-1]
            m0, m1 = map_audio_range(t0, t1, video_t, ma_t)
            mask_out.append(mask_members[1][..., m0:m1].contiguous())
            mask_out.extend(mask_members[2:])
        out["noise_mask"] = wrap_tensor(mask_out, was_nested=mask_nested)
    return out


def _drop_or_blend_prefix(
    previous: torch.Tensor,
    incoming: torch.Tensor,
    drop: int,
    *,
    blend: bool,
    time_dim: int,
) -> torch.Tensor:
    """Append incoming onto previous, dropping/blending the first `drop` along time_dim."""
    drop = max(0, min(int(drop), incoming.shape[time_dim], previous.shape[time_dim]))
    if drop <= 0:
        return torch.cat([previous, incoming], dim=time_dim)

    if not blend:
        return torch.cat(
            [previous, incoming.narrow(time_dim, drop, incoming.shape[time_dim] - drop)],
            dim=time_dim,
        )

    prev_ov = previous.narrow(time_dim, previous.shape[time_dim] - drop, drop)
    new_ov = incoming.narrow(time_dim, 0, drop)
    # alphas broadcast over all non-time dims
    shape = [1] * incoming.ndim
    shape[time_dim] = drop
    alphas = torch.linspace(0.0, 1.0, drop, device=incoming.device, dtype=torch.float32)
    alphas = alphas.reshape(shape)
    blended = prev_ov.float() * (1.0 - alphas) + new_ov.float() * alphas
    blended = blended.to(dtype=incoming.dtype)
    rest = incoming.narrow(time_dim, drop, incoming.shape[time_dim] - drop)
    head = previous.narrow(time_dim, 0, previous.shape[time_dim] - drop)
    return torch.cat([head, blended, rest], dim=time_dim)


def stitch_images(
    chunks: list[torch.Tensor],
    overlap_latent_t: int | list[int],
    chunk_latent_lengths: list[int],
    *,
    blend: bool,
) -> torch.Tensor:
    """Cat Comfy IMAGE tensors [F,H,W,C], dropping/blending overlap prefixes."""
    if not chunks:
        raise ValueError("No image chunks to stitch")
    out = chunks[0]
    for i in range(1, len(chunks)):
        clen = max(1, chunk_latent_lengths[i])
        ov = overlap_latent_t[i] if isinstance(overlap_latent_t, (list, tuple)) else overlap_latent_t
        drop = int(round(int(ov) / clen * chunks[i].shape[0]))
        out = _drop_or_blend_prefix(out, chunks[i], drop, blend=blend, time_dim=0)
    return out


def stitch_audio(
    chunks: list[dict],
    overlap_latent_t: int | list[int],
    chunk_latent_lengths: list[int],
    *,
    blend: bool,
) -> dict:
    """Cat AUDIO dicts {waveform [B,C,L], sample_rate}, dropping/blending overlap."""
    if not chunks:
        raise ValueError("No audio chunks to stitch")
    sample_rate = chunks[0]["sample_rate"]
    wave = chunks[0]["waveform"]
    for i in range(1, len(chunks)):
        clen = max(1, chunk_latent_lengths[i])
        ov = overlap_latent_t[i] if isinstance(overlap_latent_t, (list, tuple)) else overlap_latent_t
        drop = int(round(int(ov) / clen * chunks[i]["waveform"].shape[-1]))
        wave = _drop_or_blend_prefix(
            wave, chunks[i]["waveform"], drop, blend=blend, time_dim=-1
        )
    return {"waveform": wave, "sample_rate": sample_rate}


def _basic_scheduler_sigmas(model, scheduler: str, steps: int, denoise: float) -> torch.Tensor:
    """Match Comfy BasicScheduler denoise-tail trim."""
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    return sigmas[-(steps + 1) :]


def _decode_video_images(vae, latent: dict) -> torch.Tensor:
    """Mirror core VAEDecode on NestedTensor video stream → IMAGE [F,H,W,C]."""
    samples = latent["samples"]
    members, _ = extract_tensor(samples)
    video = members[0]
    images = vae.decode(video)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _soft_cache() -> None:
    try:
        gc.collect()
        model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MiniMaxH3LatentUpscaleChunked:
    """Temporally chunked pass-2: upscale → CONST re-noise → DisableNoise sample → decode → stitch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "chunk_latent_t": (
                    "INT",
                    {
                        "default": 7,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Video latent tokens per chunk (T). "
                            f"~{approx_frames_from_latent_t(7) / 24.0:.1f}s at T=7 (24 fps). "
                            "Lower if OOM on HD."
                        ),
                    },
                ),
                "overlap_latent_t": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Protected continuation prefix in video latent tokens "
                            "(snapped to H3 runs 2/7/12... = 5/22/39 frames when lock_overlap is on). "
                            "Previous chunk's sampled HD tail is copied here and not denoised."
                        ),
                    },
                ),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "method": (list(UPSCALE_METHODS), {"default": "nearest-exact"}),
                "blur": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.05,
                        "round": False,
                    },
                ),
                "audio_denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "0 = lock pass-1 audio (noise_mask 0 on the whole audio stream); >=0.5 remix. Prefer 0 for chunked pass 2.",
                    },
                ),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "lcm"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "normal"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "BasicScheduler denoise-tail (not raw sigma). Match your working pass-2 setup.",
                    },
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "overlap_blend": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Linear pixel/waveform crossfade on the (already locked) overlap. Usually leave off.",
                    },
                ),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "lock_overlap": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Motion-context masking: paste previous HD latent tail into the overlap "
                            "and set noise_mask 0 so H3 cannot rewrite it."
                        ),
                    },
                ),
                "window_conditioning": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Per-chunk guides: remap in-window AddGuide/FL2VA keyframes, drop off-window "
                            "ones and standalone audio refs so dialogue does not replay in every slice. "
                            "The text prompt is still global (H3 has no per-token timeline)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Chunked MiniMax H3 pass-2: slice pass-1 AV latent, window guides to each slice, "
        "spatially upscale + CONST re-noise, lock the previous HD overlap with an H3 noise mask, "
        "DisableNoise-sample, decode, stitch IMAGE + AUDIO."
    )

    def run(
        self,
        samples,
        model,
        positive,
        vae,
        audio_vae,
        chunk_latent_t,
        overlap_latent_t,
        scale_by,
        method,
        blur,
        audio_denoise,
        sampler_name,
        scheduler,
        steps,
        denoise,
        noise_seed,
        overlap_blend=False,
        negative=None,
        lock_overlap=True,
        window_conditioning=True,
    ):
        if "samples" not in samples:
            raise KeyError('LATENT dict missing "samples"')

        # Park full pass-1 on CPU; only slices go to GPU.
        full = finalize_latent_for_handoff(samples)
        members, was_nested = extract_tensor(full["samples"])
        video = members[0]
        if video.ndim != 5:
            raise ValueError(f"Expected video latent BxCxTxHxW, got {tuple(video.shape)}")
        video_t = int(video.shape[2])

        overlap_used = int(overlap_latent_t)
        if lock_overlap:
            snapped = snap_h3_context_tokens(overlap_used)
            if snapped != overlap_used:
                _LOG.warning(
                    "overlap_latent_t=%d snapped to H3 prefix %d (2+5k tokens)",
                    overlap_used,
                    snapped,
                )
            overlap_used = snapped
            _enable_h3_av_mask_compat()

        windows = iter_temporal_windows(video_t, chunk_latent_t, overlap_used)
        if not windows:
            raise ValueError("No temporal windows to process")

        pos_up = upscale_minimax_conditioning(positive, scale_by, method)
        if pos_up is None:
            raise ValueError("positive CONDITIONING is required")
        del negative

        sigmas = _basic_scheduler_sigmas(model, scheduler, steps, denoise)
        if len(sigmas) == 0:
            raise ValueError("denoise produced an empty sigma schedule")
        sampler = comfy.samplers.ksampler(sampler_name)
        empty_noise = Noise_EmptyNoise()

        image_chunks: list[torch.Tensor] = []
        audio_chunks: list[dict] = []
        chunk_lengths: list[int] = []
        stitch_overlap: list[int] = []
        prev_hd: dict | None = None
        prev_t1 = 0
        pbar = comfy.utils.ProgressBar(len(windows))

        for i, (t0, t1) in enumerate(windows):
            chunk_len = t1 - t0
            chunk_lengths.append(chunk_len)
            coord_overlap = min(prev_t1 - t0, chunk_len) if prev_hd is not None and t0 < prev_t1 else 0
            stitch_overlap.append(coord_overlap)
            chunk_latent = slice_av_latent(full, t0, t1)

            if window_conditioning:
                pos_chunk = window_minimax_conditioning(pos_up, t0, t1)
            else:
                pos_chunk = pos_up

            upscaled = upscale_nested_latent(chunk_latent, scale_by, method)
            if blur > 0.0:
                from .utils import blur_nested_latent

                upscaled = blur_nested_latent(upscaled, blur)

            if lock_overlap and prev_hd is not None and coord_overlap > 0:
                protect_v = snap_h3_context_tokens(min(coord_overlap, chunk_len - 1))
                chunk_members, chunk_nested = extract_tensor(upscaled["samples"])
                chunk_audio_t = (
                    int(chunk_members[1].shape[-1]) if chunk_nested and len(chunk_members) >= 2 else 0
                )
                protect_a = 0
                if protect_v > 0 and chunk_audio_t > 0:
                    protect_a = map_audio_range(0, protect_v, chunk_len, chunk_audio_t)[1]
                    protect_a = min(protect_a, max(0, chunk_audio_t - 1))
                if protect_v > 0:
                    upscaled = paste_protected_prefix(upscaled, prev_hd, protect_v, protect_a)

            zero_noise: tuple[int, ...] = ()
            up_members, up_nested = extract_tensor(upscaled["samples"])
            if up_nested and len(up_members) >= 2 and float(audio_denoise) < 0.5:
                zero_noise = (1,)
                upscaled = lock_audio_stream_mask(upscaled)

            noised = add_noise_nested_latent(
                model,
                Noise_RandomNoise(int(noise_seed) + i),
                sigmas,
                upscaled,
                zero_noise_indices=zero_noise,
            )
            noised = finalize_latent_for_handoff(noised)

            guider = Guider_Basic(model)
            guider.set_conds(pos_chunk)

            latent = noised.copy()
            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(
                guider.model_patcher,
                latent_image,
                latent.get("downscale_ratio_spacial", None),
                latent.get("downscale_ratio_temporal", None),
            )
            latent["samples"] = latent_image

            noise_mask = latent.get("noise_mask")
            x0_output: dict = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            sampled = guider.sample(
                empty_noise.generate_noise(latent),
                latent_image,
                sampler,
                sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=empty_noise.seed,
            )
            sampled = sampled.to(model_management.intermediate_device())
            out_latent = latent.copy()
            out_latent.pop("downscale_ratio_spacial", None)
            out_latent.pop("downscale_ratio_temporal", None)
            out_latent["samples"] = sampled
            out_latent.pop("noise_mask", None)

            images = _decode_video_images(vae, out_latent)
            audio = vae_decode_audio(audio_vae, out_latent)

            image_chunks.append(images.detach().to("cpu"))
            audio_chunks.append(
                {
                    "waveform": audio["waveform"].detach().to("cpu"),
                    "sample_rate": audio["sample_rate"],
                }
            )

            if lock_overlap:
                prev_hd = finalize_latent_for_handoff({"samples": sampled})
            prev_t1 = t1

            del chunk_latent, upscaled, noised, sampled, out_latent, images, audio, guider, latent_image
            _soft_cache()
            pbar.update(1)

        del prev_hd, pos_up
        images_out = stitch_images(
            image_chunks,
            stitch_overlap,
            chunk_lengths,
            blend=bool(overlap_blend),
        )
        audio_out = stitch_audio(
            audio_chunks,
            stitch_overlap,
            chunk_lengths,
            blend=bool(overlap_blend),
        )
        return (images_out, audio_out)
