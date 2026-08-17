"""MiniMax H3 NestedTensor latent helpers: extract, wrap, upscale, CONST re-noise."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils

LEARNED_UPSCALE_METHOD = "learned model"
# Older graphs stored this combo value before the display rename. Keep it in the
# list so Comfy does not silently remap those workflows onto bilinear/nearest.
LEARNED_UPSCALE_ALIASES = (LEARNED_UPSCALE_METHOD, "mamad8-learned")
# The learned model is trained against pixel-upscaled/re-encoded teacher latents and
# is the only method here that predicts a clean, on-manifold 2x latent. The remaining
# methods are retained as dependency-free fallbacks.
UPSCALE_METHODS = (
    *LEARNED_UPSCALE_ALIASES,
    "nearest-exact",
    "nearest",
    "bilinear",
    "bicubic",
    "area",
    "bislerp",
)
INTERPOLATION_METHODS = tuple(m for m in UPSCALE_METHODS if m not in LEARNED_UPSCALE_ALIASES)
NOISE_RESAMPLE_MODES = ("upscale", "independent")


def is_learned_upscale_method(method: str) -> bool:
    return str(method or "").strip() in LEARNED_UPSCALE_ALIASES

# MiniMax DiT patch_size is (1, 2, 2); cond patchify does not pad, so H/W must be even.
_SPATIAL_MULTIPLE = 2


def _snap_spatial(size: int, multiple: int = _SPATIAL_MULTIPLE) -> int:
    return max(multiple, ((int(size) + multiple - 1) // multiple) * multiple)


def is_nested_tensor(obj: Any) -> bool:
    return isinstance(obj, comfy.nested_tensor.NestedTensor) or getattr(obj, "is_nested", False)


def extract_tensor(samples: Any) -> tuple[list[torch.Tensor], bool]:
    """Unwrap NestedTensor members, or wrap a plain torch.Tensor in a one-element list.

    Returns (tensors, was_nested).
    """
    if is_nested_tensor(samples):
        tensors = list(samples.unbind())
        if not tensors:
            raise ValueError("NestedTensor has an empty .tensors list")
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"NestedTensor member [{i}] is {type(t)}, expected torch.Tensor")
        return tensors, True
    if isinstance(samples, torch.Tensor):
        return [samples], False
    raise TypeError(f"Expected NestedTensor or torch.Tensor, got {type(samples)}")


def wrap_tensor(
    tensors: Sequence[torch.Tensor],
    *,
    was_nested: bool,
) -> Any:
    """Rebuild NestedTensor with ComfyUI's constructor, or return a single plain tensor."""
    if was_nested:
        if not tensors:
            raise ValueError("Cannot wrap empty tensor list as NestedTensor")
        return comfy.nested_tensor.NestedTensor(tensors)
    if len(tensors) != 1:
        raise ValueError(f"Plain latent path expects one tensor, got {len(tensors)}")
    return tensors[0]


def upscale_video_latent(
    video: torch.Tensor,
    scale_by: float,
    method: str,
) -> torch.Tensor:
    """Spatially upscale a video latent [B, C, T, H, W] (or [B, C, H, W]).

    Uses Comfy's common_upscale (fp32) so nearest-exact / bislerp match core latent
    upscalers. Skips interpolate when H/W already match the snapped target.
    """
    if method == "nearest":
        method = "nearest-exact"
    if method not in INTERPOLATION_METHODS and method != "nearest-exact":
        raise ValueError(f"Unsupported method {method!r}; expected one of {INTERPOLATION_METHODS}")
    if video.ndim < 4:
        raise ValueError(f"Video latent needs at least 4 dims [B,C,H,W], got shape {tuple(video.shape)}")

    height = _snap_spatial(max(1, round(video.shape[-2] * scale_by)))
    width = _snap_spatial(max(1, round(video.shape[-1] * scale_by)))
    if height == video.shape[-2] and width == video.shape[-1]:
        return video

    orig_dtype = video.dtype
    # fp16/bf16 interpolate quantizes the new grid and shows up as pass-2 speckles.
    samples = video.float() if video.dtype in (torch.float16, torch.bfloat16) else video
    out = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return out.to(dtype=orig_dtype)


def _spatial_gaussian_blur(video: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian blur over H/W only. radius is in output-latent pixels.

    Replicate padding, not zero padding: a latent's per-channel mean is not 0, so
    zero-padding pulls the frame's outermost tokens toward the origin and leaves a
    one-token border of short (out-of-distribution) vectors for pass 2 to reinvent.
    """
    if radius <= 0.0:
        return video
    sigma = float(radius)
    kernel_size = int(max(3, (int(sigma * 4) | 1)))
    coords = torch.arange(kernel_size, dtype=torch.float32, device=video.device)
    coords = coords - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    orig_dtype = video.dtype
    x = video.float()
    orig_shape = tuple(x.shape)
    if x.ndim == 5:
        b, c, t, h, w = orig_shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    elif x.ndim != 4:
        return video
    else:
        c = x.shape[1]

    padding = kernel_size // 2
    kw = kernel_1d.view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size)
    kh = kernel_1d.view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1)
    x = F.conv2d(F.pad(x, (padding, padding, 0, 0), mode="replicate"), kw, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, padding, padding), mode="replicate"), kh, groups=c)

    if len(orig_shape) == 5:
        x = x.reshape(orig_shape[0], orig_shape[2], orig_shape[1], orig_shape[3], orig_shape[4])
        x = x.permute(0, 2, 1, 3, 4)
    return x.to(dtype=orig_dtype)


def _smooth_latent_preserving_norm(video: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaussian-smooth the *direction* of each latent token, keep its original length.

    A latent token is a 24-dim vector, not a pixel. Plain blurring averages adjacent
    tokens componentwise, so wherever neighbours point in different directions — i.e.
    at every content edge — the directional components cancel and the resulting vector
    is far shorter than any code the VAE ever produced. Measured on a 24-channel field
    at 2x, blur=0.5 collapses 19% of tokens below 70% of the mean norm (mean 4.86 ->
    3.93); blur=1.0 collapses 99%. Those short tokens are strongly out of distribution,
    so pass 2 invents content at them, and because the collapse tracks content edges the
    invented content reads as thin strings / stray hairs following edges.

    Rescaling each token back to its pre-smoothing norm keeps essentially all of the
    anti-blocking benefit (block-edge/interior contrast 7.8 vs plain blur's 7.6 at
    radius 0.5) with zero norm error. Same idea as Comfy's bislerp, which slerps
    direction and interpolates magnitude separately rather than mixing componentwise.

    Note this fixes token *magnitude* only. The smoothed direction is still a linear
    blend of neighbouring latent directions, which is not guaranteed to be on-manifold,
    so radius > 0 can still cost some fidelity. radius=0 remains the safest setting.
    """
    if radius <= 0.0 or video.ndim < 4:
        return video
    smoothed = _spatial_gaussian_blur(video, radius)
    orig_dtype = video.dtype
    # channel dim is 1 for both [B,C,H,W] and [B,C,T,H,W]
    src_norm = video.float().norm(dim=1, keepdim=True)
    out_norm = smoothed.float().norm(dim=1, keepdim=True)
    rescaled = smoothed.float() * (src_norm / out_norm.clamp_min(1e-6))
    return rescaled.to(dtype=orig_dtype)


def _sharpen_latent_preserving_norm(
    video: torch.Tensor,
    amount: float,
    *,
    radius: float = 1.0,
) -> torch.Tensor:
    """Directional unsharp mask that preserves every latent token's magnitude.

    A conventional unsharp mask, x + amount * (x - blur(x)), increases the length
    of 24-D H3 latent vectors most strongly at edges. Those large edge tokens are as
    out-of-distribution as the short tokens created by plain blur and can become
    outlines, strings, or temporal shimmer in pass 2. We sharpen the vector direction
    but project each result back to its pre-sharpen norm. This cannot guarantee that
    the new direction is on the learned VAE manifold, so the control defaults to zero
    and should be used sparingly.

    radius is fixed by the node at one output-latent pixel: broad enough to counter
    the learned upscaler's softness without targeting single-value quantization noise.
    """
    amount = max(float(amount), 0.0)
    if amount <= 0.0 or video.ndim < 4:
        return video
    source = video.float()
    lowpass = _spatial_gaussian_blur(source, radius).float()
    sharpened = source + amount * (source - lowpass)
    source_norm = source.norm(dim=1, keepdim=True)
    sharpened_norm = sharpened.norm(dim=1, keepdim=True)
    sharpened = sharpened * (source_norm / sharpened_norm.clamp_min(1e-6))
    return sharpened.to(dtype=video.dtype)


def _contrast_latent(
    video: torch.Tensor,
    contrast: float,
    *,
    preserve_norm: bool = True,
) -> torch.Tensor:
    """Per-channel contrast around the global channel mean.

    contrast=1 is a no-op. Values >1 stretch deviation from the mean; <1 compress it.
    When preserve_norm is True (default), each 24-D token is projected back to its
    pre-contrast magnitude so H3 edge tokens do not become oversized/short.
    """
    contrast = float(contrast)
    if abs(contrast - 1.0) < 1e-6 or video.ndim < 4:
        return video
    source = video.float()
    dims = tuple(range(2, source.ndim))
    mean = source.mean(dim=dims, keepdim=True)
    contrasted = mean + contrast * (source - mean)
    if preserve_norm:
        source_norm = source.norm(dim=1, keepdim=True)
        out_norm = contrasted.norm(dim=1, keepdim=True)
        contrasted = contrasted * (source_norm / out_norm.clamp_min(1e-6))
    return contrasted.to(dtype=video.dtype)


def _map_video_latent(latent: dict, video_fn) -> dict:
    """Apply video_fn to the NestedTensor video member; leave audio / noise_mask alone."""
    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')
    members, was_nested = extract_tensor(latent["samples"])
    if not members:
        raise ValueError("LATENT samples are empty")
    out_members = list(members)
    out_members[0] = video_fn(members[0])
    result = latent.copy()
    result["samples"] = wrap_tensor(out_members, was_nested=was_nested)
    return result


def blur_nested_latent(latent: dict, radius: float) -> dict:
    """Norm-preserving spatial smooth on the video stream only."""
    radius = max(float(radius), 0.0)
    if radius <= 0.0:
        return latent
    return _map_video_latent(
        latent,
        lambda video: _smooth_latent_preserving_norm(video, radius),
    )


def sharpen_nested_latent(
    latent: dict,
    amount: float,
    *,
    radius: float = 1.0,
) -> dict:
    """Norm-preserving latent unsharp on the video stream only."""
    amount = max(float(amount), 0.0)
    if amount <= 0.0:
        return latent
    return _map_video_latent(
        latent,
        lambda video: _sharpen_latent_preserving_norm(video, amount, radius=radius),
    )


def contrast_nested_latent(
    latent: dict,
    contrast: float,
    *,
    preserve_norm: bool = True,
) -> dict:
    """Per-channel contrast on the video stream only."""
    if abs(float(contrast) - 1.0) < 1e-6:
        return latent
    return _map_video_latent(
        latent,
        lambda video: _contrast_latent(video, contrast, preserve_norm=preserve_norm),
    )


def _match_channel_energy(
    original: torch.Tensor,
    upscaled: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Rescale each channel of `upscaled` toward the per-channel mean/std of `original`.

    nearest-exact upscale duplicates pixels exactly, so it does not change a video
    latent's global per-channel statistics. Any low-pass step after that — the
    optional Gaussian blur, or bilinear/bicubic/area upscale methods themselves —
    reduces variance. CONST/flow mixing (x_sigma = sigma*noise + (1-sigma)*x0) is
    calibrated on the DiT's expected x0 energy at each sigma; a quietly lower-energy
    x0 mixes in as if it were a higher (noisier) sigma than the number says, so the
    sampler denoises harder and hallucinates detail at settings that looked fine at
    native resolution.

    Caveat: nearest-exact's block edges and a mild blur's softened ramps live in the
    same frequency band as genuine per-pixel detail at this resolution — there is no
    clean spectral split between "the grid" and "real texture" here. This is a global
    per-channel contrast stretch, not a re-sharpen: it cannot literally reconstruct
    the sharp block edges blur removed, but stretching contrast around the channel
    mean raises the softened edge ramps back up right along with real texture. At
    strength=1.0 (full restoration) this puts the residual block-edge energy back to
    ~97% of its pre-blur amplitude — the blur is barely still doing its job — which is
    what shows up as faint "string"/hair artefacts once pass 2 samples it. strength<1
    only partially restores the lost std, trading a bit of that energy-calibration
    benefit for keeping more of blur's grid suppression intact.

    Reduces over every non-batch, non-channel dim (T, H, W independently sized between
    original and upscaled is fine — this matches global distribution, not per-pixel).
    """
    dims = tuple(range(2, original.ndim))
    if not dims or strength <= 0.0:
        return upscaled
    orig = original.float()
    up = upscaled.float()
    orig_mean = orig.mean(dim=dims, keepdim=True)
    orig_std = orig.std(dim=dims, keepdim=True)
    up_mean = up.mean(dim=dims, keepdim=True)
    up_std = up.std(dim=dims, keepdim=True)
    full_scale = orig_std / up_std.clamp_min(1e-6)
    scale = 1.0 + min(max(strength, 0.0), 1.0) * (full_scale - 1.0)
    matched = (up - up_mean) * scale + up_mean.lerp(orig_mean, min(max(strength, 0.0), 1.0))
    return matched.to(dtype=upscaled.dtype)


def upscale_nested_latent(
    latent: dict,
    scale_by: float,
    method: str,
    *,
    match_stats: float = 1.0,
    upscaler: Any | None = None,
) -> dict:
    """Copy LATENT dict; upscale video spatial dims; pass audio through; rebuild NestedTensor.

    match_stats: 0..1 strength for rescaling the upscaled video toward pass-1's
    per-channel energy. See _match_channel_energy. 0 disables (no-op).
    Blur / sharpen are separate standalone nodes — not applied here.
    """
    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')

    learned = is_learned_upscale_method(method)
    if learned:
        if float(scale_by) != 2.0:
            raise ValueError("Learned upscale is trained and hard-locked to scale_by=2.0")
        from .learned import apply_h3_latent_upscale

        # Same apply contract as H3CleanLatentUpscale2x: locate the Bx24xTxHxW
        # stream, run the patcher, replace only that stream, drop noise_mask.
        return apply_h3_latent_upscale(upscaler, latent)

    out = latent.copy()
    members, was_nested = extract_tensor(latent["samples"])

    if was_nested:
        # MiniMax H3 / AV: tensors[0]=video [B,24,T,H,W], tensors[1]=audio [B,32,2,Ta]
        video_up = upscale_video_latent(members[0], scale_by, method)
        if match_stats > 0.0:
            video_up = _match_channel_energy(members[0], video_up, strength=match_stats)
        out["samples"] = wrap_tensor([video_up, *members[1:]], was_nested=True)

        noise_mask = latent.get("noise_mask")
        if noise_mask is not None and is_nested_tensor(noise_mask):
            mask_members, _ = extract_tensor(noise_mask)
            if not mask_members:
                raise ValueError("noise_mask NestedTensor is empty")
            # Masks are geometry, not latents: always resize them exactly.
            video_mask_up = upscale_video_latent(mask_members[0], scale_by, "nearest-exact")
            out["noise_mask"] = wrap_tensor([video_mask_up, *mask_members[1:]], was_nested=True)
        elif noise_mask is not None and isinstance(noise_mask, torch.Tensor) and noise_mask.ndim >= 4:
            out["noise_mask"] = upscale_video_latent(noise_mask, scale_by, "nearest-exact")
    else:
        video_up = upscale_video_latent(members[0], scale_by, method)
        if match_stats > 0.0:
            video_up = _match_channel_energy(members[0], video_up, strength=match_stats)
        out["samples"] = video_up
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None and isinstance(noise_mask, torch.Tensor) and noise_mask.ndim >= 4:
            out["noise_mask"] = upscale_video_latent(noise_mask, scale_by, "nearest-exact")

    return out


def _has_nonzero(samples: Any) -> bool:
    """NestedTensor-safe replacement for torch.count_nonzero(samples) > 0."""
    members, _ = extract_tensor(samples)
    return any(torch.count_nonzero(t) > 0 for t in members)


def _zero_noise_members(noisy: Any, indices: Sequence[int]) -> Any:
    members, was_nested = extract_tensor(noisy)
    out = list(members)
    for i in indices:
        if i < 0 or i >= len(out):
            raise IndexError(f"noise zero index {i} out of range for {len(out)} members")
        out[i] = torch.zeros_like(out[i])
    return wrap_tensor(out, was_nested=was_nested)


def _mask_noise_by_denoise_mask(noisy: Any, denoise_mask: Any) -> Any:
    """Zero noise where denoise_mask < 0.5 (0 = preserve). Broadcasts mask channels."""
    noise_members, was_nested = extract_tensor(noisy)
    mask_members, _ = extract_tensor(denoise_mask)
    if len(mask_members) < len(noise_members):
        raise ValueError(
            f"noise_mask has {len(mask_members)} members but noise has {len(noise_members)}"
        )
    out: list[torch.Tensor] = []
    for noi, msk in zip(noise_members, mask_members):
        m = msk.to(device=noi.device, dtype=noi.dtype)
        while m.ndim < noi.ndim:
            m = m.unsqueeze(1)
        if m.shape[1] == 1 and noi.shape[1] != 1:
            m = m.expand(noi.shape)
        elif m.shape != noi.shape:
            m = m.expand_as(noi)
        out.append(noi * m)
    return wrap_tensor(out, was_nested=was_nested)


def _nan_to_num_samples(samples: Any) -> Any:
    members, was_nested = extract_tensor(samples)
    return wrap_tensor(
        [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in members],
        was_nested=was_nested,
    )


def add_noise_nested_latent(
    model: Any,
    noise: Any,
    sigmas: torch.Tensor,
    latent: dict,
    *,
    zero_noise_indices: Sequence[int] = (),
    noisy: Any | None = None,
) -> dict:
    """NestedTensor-aware AddNoise for CONST/flow DisableNoise continuation.

    Matches core AddNoise: process_latent_in/out on the full NestedTensor (so MiniMax
    audio_scale is applied to the audio stream, not a packed tail of the video tensor),
    mix at sigmas[0], then inverse_noise_scaling so SamplerCustomAdvanced + DisableNoise
    reconstitutes σ·ε + (1-σ)·x instead of (1-σ)²·x.

    zero_noise_indices: NestedTensor members that stay clean (typically audio=1).
    They are still inverse-scaled so DisableNoise reconstitutes clean, not (1-σ)·clean.

    latent['noise_mask'] (1 = denoise, 0 = preserve) zeros noise on protected tokens
    so CONST re-noise does not rewrite a locked continuation prefix.
    """
    if len(sigmas) == 0:
        return latent

    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')

    out = latent.copy()
    latent_image = latent["samples"]
    if noisy is None:
        noisy = noise.generate_noise(latent)
    if zero_noise_indices:
        noisy = _zero_noise_members(noisy, zero_noise_indices)
    denoise_mask = latent.get("noise_mask")
    if denoise_mask is not None:
        noisy = _mask_noise_by_denoise_mask(noisy, denoise_mask)

    model_sampling = model.get_model_object("model_sampling")
    process_latent_out = model.get_model_object("process_latent_out")
    process_latent_in = model.get_model_object("process_latent_in")

    sigma_start = sigmas[0]
    if _has_nonzero(latent_image):
        # Whole NestedTensor so MiniMaxH3 audio_scale hits the audio stream.
        latent_image = process_latent_in(latent_image)

    # CONST.noise_scaling does `s * noise`; NestedTensor has no __rmul__.
    lat_members, was_nested = extract_tensor(latent_image)
    noise_members, noise_was_nested = extract_tensor(noisy)
    if len(lat_members) != len(noise_members):
        raise ValueError(
            f"Noise NestedTensor has {len(noise_members)} members but latent has {len(lat_members)}"
        )
    mixed_members: list[torch.Tensor] = []
    for lat, noi in zip(lat_members, noise_members):
        mixed = model_sampling.noise_scaling(sigma_start, noi, lat)
        if hasattr(model_sampling, "inverse_noise_scaling"):
            mixed = model_sampling.inverse_noise_scaling(sigma_start, mixed)
        mixed_members.append(mixed)
    mixed = wrap_tensor(mixed_members, was_nested=was_nested or noise_was_nested)
    mixed = process_latent_out(mixed)
    out["samples"] = _nan_to_num_samples(mixed)
    return out


def _move_samples_to_cpu(samples: Any) -> Any:
    """Detach NestedTensor / Tensor samples onto CPU for cheap cache between samplers."""
    if is_nested_tensor(samples):
        return samples.cpu()
    if isinstance(samples, torch.Tensor):
        return samples.detach().to("cpu")
    return samples


def finalize_latent_for_handoff(latent: dict) -> dict:
    """Park LATENT on CPU and soft-clear CUDA cache without unloading models.

    Forced unload / Easy-Use Empty Cache between MiniMax passes is unsafe with
    quantized weights + --disable-dynamic-vram (ends as 0MB loaded / illegal
    SageAttention access). Soft empty_cache only frees allocator fragments.
    """
    out = latent.copy()
    if "samples" in out:
        out["samples"] = _move_samples_to_cpu(out["samples"])
    if "noise_mask" in out:
        out["noise_mask"] = _move_samples_to_cpu(out["noise_mask"])

    try:
        import comfy.model_management as mm
        import gc

        gc.collect()
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def lock_audio_stream_mask(latent: dict) -> dict:
    """Force audio denoise_mask to 0 (preserve). Keep or create a video mask of 1s.

    H3 still steps the joint DiT unless noise_mask is 0; zeroing injected noise is not enough.
    """
    if "samples" not in latent:
        return latent
    members, was_nested = extract_tensor(latent["samples"])
    if not was_nested or len(members) < 2:
        return latent
    video, audio = members[0], members[1]
    extra_masks: list[torch.Tensor] = []
    existing = latent.get("noise_mask")
    if existing is not None:
        mask_m, _ = extract_tensor(existing)
        video_mask = mask_m[0]
        extra_masks = list(mask_m[2:])
    else:
        video_mask = torch.ones(
            (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
            device=video.device,
            dtype=torch.float32,
        )
    audio_mask = torch.zeros(
        (audio.shape[0], 1, audio.shape[2], audio.shape[3]),
        device=audio.device,
        dtype=torch.float32,
    )
    out = latent.copy()
    out["noise_mask"] = wrap_tensor([video_mask, audio_mask, *extra_masks], was_nested=True)
    return out


def upscale_and_add_noise(
    latent: dict,
    scale_by: float,
    method: str,
    model: Any,
    noise: Any,
    sigmas: torch.Tensor,
    *,
    audio_denoise: float = 1.0,
    noise_resample: str = "independent",
    match_stats: float = 1.0,
    upscaler: Any | None = None,
) -> dict:
    """Upscale video spatially; re-noise video fully; optionally re-noise audio.

    audio_denoise: 0 = keep pass-1 audio clean (zero audio noise + audio noise_mask 0),
    1 = full CONST remix of audio at sigmas[0] so pass 2 can improve it.
    Values in between are treated as lock (<0.5) or full (>=0.5).

    noise_resample is accepted for widget compatibility. Video noise is always
    drawn iid on the upscaled grid. Nearest-copying low-res noise is OOD for H3.

    match_stats: 0..1 strength, see _match_channel_energy. Blur/sharpen are
    standalone nodes applied outside this path.
    """
    del noise_resample
    upscaled = upscale_nested_latent(
        latent,
        scale_by,
        method,
        match_stats=match_stats,
        upscaler=upscaler,
    )
    members, was_nested = extract_tensor(upscaled["samples"])
    zero_noise: tuple[int, ...] = ()
    if was_nested and len(members) >= 2 and float(audio_denoise) < 0.5:
        zero_noise = (1,)
        upscaled = lock_audio_stream_mask(upscaled)
    noisy = None
    noised = add_noise_nested_latent(
        model,
        noise,
        sigmas,
        upscaled,
        zero_noise_indices=zero_noise,
        noisy=noisy,
    )
    return finalize_latent_for_handoff(noised)


def _upscale_video_like_latent(z: torch.Tensor, scale_by: float, method: str) -> torch.Tensor:
    """Upscale a MiniMax visual latent; supports 5D video or 4D image-like tensors.

    Snaps H/W to the DiT spatial patch multiple (2). Cond path patchify_video does not
    pad, so odd sizes after 1.5× (e.g. 50→75) crash reshape.
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for visual latent, got {type(z)}")
    if z.ndim == 4:
        z5 = upscale_video_latent(z.unsqueeze(2), scale_by, method)
        # pad T,H,W with MiniMax patch (1,2,2)
        import comfy.ldm.common_dit as common_dit
        z5 = common_dit.pad_to_patch_size(z5, (1, _SPATIAL_MULTIPLE, _SPATIAL_MULTIPLE))
        return z5.squeeze(2)
    if z.ndim == 5:
        z5 = upscale_video_latent(z, scale_by, method)
        import comfy.ldm.common_dit as common_dit
        return common_dit.pad_to_patch_size(z5, (1, _SPATIAL_MULTIPLE, _SPATIAL_MULTIPLE))
    raise ValueError(f"Visual latent needs 4 or 5 dims, got shape {tuple(z.shape)}")


def upscale_minimax_ref_block(block: dict, scale_by: float, method: str) -> dict:
    """Spatially upscale one minimax_refs block; leave audio-only fields unchanged."""
    out = dict(block)
    kind = out.get("kind")
    if kind == "audio":
        return out

    if "latent" in out and out["latent"] is not None:
        z = _upscale_video_like_latent(out["latent"], scale_by, method)
        out["latent"] = z
        # PackedLayout reads these for RoPE / row counts — keep in sync with tensor.
        if z.ndim == 5:
            out["latent_h"] = int(z.shape[-2])
            out["latent_w"] = int(z.shape[-1])
            if "latent_t" in out:
                out["latent_t"] = int(z.shape[2])
        elif z.ndim == 4:
            out["latent_h"] = int(z.shape[-2])
            out["latent_w"] = int(z.shape[-1])

    # audio_latent / ref_audio_t intentionally untouched (no spatial dims)
    return out


def upscale_minimax_keyframe(kf: dict, scale_by: float, method: str) -> dict:
    """Upscale a minimax_keyframes entry so it matches the new target spatial grid."""
    out = dict(kf)
    if "latent" in out and out["latent"] is not None:
        out["latent"] = _upscale_video_like_latent(out["latent"], scale_by, method)
    return out


def upscale_minimax_conditioning(
    conditioning: list | None,
    scale_by: float,
    method: str,
) -> list | None:
    """Clone CONDITIONING and spatially upscale MiniMax ref/keyframe visual latents.

    Updates minimax_refs[*].latent (+ latent_h/w/t) and minimax_keyframes[*].latent.
    Text / audio cond tensors are left alone. Rebuild the Guider from the returned
    conditioning for sampler #2 (do not reuse a Guider built on pre-upscale cond).
    """
    if conditioning is None:
        return None

    out: list = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            out.append(entry)
            continue
        emb, meta = entry[0], entry[1]
        new_meta = meta.copy()

        refs = meta.get("minimax_refs")
        if refs is not None:
            new_meta["minimax_refs"] = [
                upscale_minimax_ref_block(blk, scale_by, method) for blk in refs
            ]

        keyframes = meta.get("minimax_keyframes")
        if keyframes is not None:
            new_meta["minimax_keyframes"] = [
                upscale_minimax_keyframe(kf, scale_by, method) for kf in keyframes
            ]

        out.append([emb, new_meta])
    return out
