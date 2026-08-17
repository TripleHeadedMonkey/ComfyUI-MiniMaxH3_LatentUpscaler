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

import torch

import comfy.model_management as model_management

from .utils import extract_tensor

_H3_VAE_ATTRS = ("tiling", "tile_size", "tile_overlap_min")
_H3_TILE_MULTIPLE = 16  # spatial VAE ratio; split_tiles assumes pixel sizes on this grid
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


def _decode_h3(vae, video: torch.Tensor, output_device: torch.device) -> torch.Tensor:
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
        pixel_samples = torch.empty(
            inner.decode_output_shape(video.shape),
            device=output_device,
            dtype=vae.vae_output_dtype(),
        )
        samples = video.to(device=vae.device, dtype=vae.vae_dtype)
        inner.decode(samples, output_buffer=pixel_samples)
        vae.process_output(pixel_samples)
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
        tile_size = _snap_multiple(tile_size, _H3_TILE_MULTIPLE, _STOCK_TILE)
        overlap = _effective_overlap(tile_size, tile_overlap)

        if output_device == "gpu":
            out_dev = model_management.get_torch_device()
        else:
            out_dev = model_management.intermediate_device()

        saved = {name: getattr(inner, name) for name in _H3_VAE_ATTRS}
        try:
            inner.tiling = bool(tiling)
            inner.tile_size = tile_size
            inner.tile_overlap_min = overlap
            images = _decode_h3(vae, video, out_dev)
        finally:
            for name, value in saved.items():
                setattr(inner, name, value)

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)
