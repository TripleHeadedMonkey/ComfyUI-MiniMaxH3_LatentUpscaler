"""H3 video VAE decode with spatial-tile / output-device knobs.

Stock VAEDecode / VAEDecodeTiled cannot change MiniMaxH3VideoVAE.tile_size
(decode_tiled ignores kwargs). This node snapshots those flags, decodes, then
restores them so the rest of the graph is unchanged.

H3's reference decoder is 256px tiles with 64px overlap. Larger tiles need the
same overlap *ratio* (tile/4) or seams look like a block grid. Comfy's VAE.decode
OOM path also falls back to a generic 3D tiler that ignores H3's blend rules —
this node calls the H3 decoder directly so that fallback cannot silently ruin quality.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

import comfy.model_management as model_management

from .utils import extract_tensor

_H3_VAE_ATTRS = ("tiling", "tile_size", "tile_overlap_min")
_H3_TILE_MULTIPLE = 16  # native spatial VAE ratio
_STOCK_TILE = 256
_STOCK_OVERLAP = 64


def _h3_video_vae(vae):
    inner = getattr(vae, "first_stage_model", None)
    if inner is None or not all(hasattr(inner, name) for name in _H3_VAE_ATTRS):
        raise ValueError(
            "MiniMax H3 VAE Decode (fast) expects the MiniMax H3 *video* VAE. "
            "Connect the video VAE, not the audio VAE."
        )
    return inner


def _video_latent(samples: dict) -> torch.Tensor:
    if "samples" not in samples:
        raise KeyError('LATENT dict missing "samples"')
    members, _ = extract_tensor(samples["samples"])
    video = members[0]
    if video.ndim != 5:
        raise ValueError(
            f"Expected H3 video latent BxCxTxHxW, got shape {tuple(video.shape)}"
        )
    return video


def _snap_multiple(value: int, multiple: int, minimum: int) -> int:
    value = max(int(minimum), int(value))
    return max(minimum, (value // multiple) * multiple)


def _effective_overlap(tile_size: int, tile_overlap: int) -> int:
    """Keep at least the stock 64/256 overlap ratio so larger tiles do not seam."""
    scaled = max(_STOCK_OVERLAP, (int(tile_size) * _STOCK_OVERLAP) // _STOCK_TILE)
    overlap = max(int(tile_overlap), scaled)
    overlap = _snap_multiple(overlap, _H3_TILE_MULTIPLE, _H3_TILE_MULTIPLE)
    return min(overlap, max(_H3_TILE_MULTIPLE, tile_size - _H3_TILE_MULTIPLE))


def _decoder_channel_packing(inner) -> tuple[int, int]:
    """Infer packed output channels and PixelShuffle factor from projection rows."""
    decoder = inner.decoder
    rows = int(decoder.proj_out.weight.shape[0])
    temporal = int(decoder.patch_size_t)
    spatial = int(decoder.patch_size)
    native_channels = 3
    patch_volume = temporal * spatial * spatial
    if rows % patch_volume:
        raise ValueError(
            f"H3 decoder projection has {rows} rows, not divisible by its "
            f"temporal/spatial patch volume {patch_volume}."
        )
    packed_channels = rows // patch_volume
    if packed_channels % native_channels:
        raise ValueError(
            f"H3 decoder projection implies {packed_channels} output channels, "
            f"not divisible by {native_channels} RGB channels."
        )
    ratio_squared = packed_channels // native_channels
    ratio = math.isqrt(ratio_squared)
    if ratio * ratio != ratio_squared:
        raise ValueError(
            f"H3 decoder's {packed_channels} packed channels do not form a "
            "square PixelShuffle ratio."
        )
    return packed_channels, ratio


def _expand_rgb_stat(stat: torch.Tensor, packed_channels: int) -> torch.Tensor:
    """Expand RGB normalization buffers to packed PixelShuffle channels.

    The converted projection is laid out R phases, G phases, B phases, so each
    native RGB statistic must be repeated consecutively rather than repeating
    the whole RGB triplet.
    """
    if packed_channels == 3:
        return stat
    if any(size == packed_channels for size in stat.shape):
        return stat
    if packed_channels % 3:
        raise ValueError(f"Cannot expand RGB statistics to {packed_channels} channels.")

    channel_dims = [i for i, size in enumerate(stat.shape) if size == 3]
    if not channel_dims:
        raise ValueError(
            f"Expected an RGB normalization buffer with a size-3 dimension, got {tuple(stat.shape)}."
        )
    # H3 normally stores these as 1x3x1x1x1. Prefer dimension 1 for batched
    # buffers and otherwise use the first size-3 dimension.
    channel_dim = 1 if stat.ndim > 1 and stat.shape[1] == 3 else channel_dims[0]
    return stat.repeat_interleave(packed_channels // 3, dim=channel_dim)


def _decode_h3(vae, video: torch.Tensor, output_device: torch.device, upscale: int) -> torch.Tensor:
    """Same load + chunked-io path as VAE.decode, without the generic tiled OOM fallback."""
    vae.throw_exception_if_invalid()
    inner = vae.first_stage_model
    with model_management.cuda_device_context(vae.device):
        memory_used = vae.memory_used_decode(video.shape, vae.vae_dtype)
        model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        # Dynamic loading can restore the checkpoint's original 3-channel
        # normalization buffers, so expand them only after the load completes.
        # This must be the final mutation before native decode/finalization.
        if upscale > 1:
            packed_channels = int(inner.decoder.out_channels)
            inner.pixel_mean = _expand_rgb_stat(inner.pixel_mean, packed_channels)
            inner.pixel_std = _expand_rgb_stat(inner.pixel_std, packed_channels)
        pixel_samples = torch.empty(
            inner.decode_output_shape(video.shape),
            device=output_device,
            dtype=vae.vae_output_dtype(),
        )
        samples = video.to(device=vae.device, dtype=vae.vae_dtype)
        inner.decode(samples, output_buffer=pixel_samples)
        vae.process_output(pixel_samples)
        if upscale > 1:
            # torch.pixel_shuffle expects N,C,H,W. For H3's B,C,T,H,W video,
            # fold time into the batch, shuffle each frame, then restore time.
            batch, channels, frames, height, width = pixel_samples.shape
            frame_batch = pixel_samples.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
            frame_batch = F.pixel_shuffle(frame_batch, upscale_factor=upscale)
            out_channels = channels // (upscale * upscale)
            pixel_samples = frame_batch.reshape(
                batch, frames, out_channels, height * upscale, width * upscale
            ).permute(0, 2, 1, 3, 4).contiguous()
    return pixel_samples.to(output_device).movedim(1, -1)


class MiniMaxH3VAEDecodeFast:
    """Decode H3 video latents with larger spatial tiles and optional GPU output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "On: H3 spatial tiles (same algorithm as stock VAEDecode). "
                            "Off: one full-frame ViT per temporal chunk — faster, more VRAM. "
                            "If it OOMs, Comfy will not silently switch to a different tiler."
                        ),
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": (
                            "Spatial tile size in pixels. 256 + overlap 64 matches stock VAEDecode. "
                            "Larger tiles are faster; overlap is raised automatically to keep the "
                            "64/256 ratio so seams do not show as a block grid."
                        ),
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 16,
                        "max": 1024,
                        "step": 16,
                        "tooltip": (
                            "Minimum pixel overlap. Actual overlap is max(this, tile_size/4). "
                            "Stock is 64 at 256px. Ignored if tiling is off."
                        ),
                    },
                ),
                "output_device": (
                    ["cpu", "gpu"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "Where finished RGB is stored. cpu = stock Comfy. gpu = skip "
                            "GPU→CPU copies (needs VRAM for the whole clip). Quality is the same."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "H3 video VAE decode with spatial-tile and output-device controls. "
        "Default 256px / 64 overlap matches stock VAEDecode quality. Larger tile_size "
        "keeps the same overlap ratio. Decodes through H3's own tiled path only — no "
        "generic 3D-tiler fallback. Temporal 17-frame chunking is unchanged."
    )

    def decode(self, samples, vae, tiling=True, tile_size=256, tile_overlap=64, output_device="cpu"):
        inner = _h3_video_vae(vae)
        video = _video_latent(samples)
        packed_channels, upscale = _decoder_channel_packing(inner)
        tile_size = _snap_multiple(tile_size, _H3_TILE_MULTIPLE, _STOCK_TILE)
        overlap = _effective_overlap(tile_size, tile_overlap)

        if output_device == "gpu":
            out_dev = model_management.get_torch_device()
        else:
            out_dev = model_management.intermediate_device()

        saved = {name: getattr(inner, name) for name in _H3_VAE_ATTRS}
        saved_out_channels = inner.decoder.out_channels
        saved_pixel_mean = inner.pixel_mean
        saved_pixel_std = inner.pixel_std
        try:
            # Like the creator's WAN 2x VAE, the converted MiniMax projection emits
            # packed RGB subpixel channels at its native spatial ratio. PixelShuffle
            # turns those channels into the final 2x image after H3's own tiled decode.
            inner.decoder.out_channels = packed_channels
            inner.tiling = bool(tiling)
            inner.tile_size = tile_size
            inner.tile_overlap_min = overlap
            images = _decode_h3(vae, video, out_dev, upscale)
        finally:
            inner.decoder.out_channels = saved_out_channels
            inner.pixel_mean = saved_pixel_mean
            inner.pixel_std = saved_pixel_std
            for name, value in saved.items():
                setattr(inner, name, value)

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)
