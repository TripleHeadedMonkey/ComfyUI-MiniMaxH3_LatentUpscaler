"""MiniMaxH3_LatentUpscaler — NestedTensor upscale + CONST re-noise for MiniMax H3."""

from .nodes import (
    MiniMaxH3LatentBlur,
    MiniMaxH3LatentContrast,
    MiniMaxH3LatentSharpen,
    MiniMaxH3LatentUpscaleCombined,
)
from .schedulers import MiniMaxH3Pass2StaggeredScheduler
from .stash import MiniMaxH3LQPackageLoad, MiniMaxH3LQPackageSave
from .stash_collect import MiniMaxH3UpscaleCollect
from .stash_server import register_routes
from .vae_decode import MiniMaxH3VAEDecodeFast

# The chunked pass-2 node (chunked.py) is intentionally not registered: temporal
# chunking of a single H3 generation could not maintain cross-chunk consistency.
# The module is kept for reference but no class is exposed to ComfyUI.

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentUpscaleCombined": MiniMaxH3LatentUpscaleCombined,
    "MiniMaxH3LatentBlur": MiniMaxH3LatentBlur,
    "MiniMaxH3LatentSharpen": MiniMaxH3LatentSharpen,
    "MiniMaxH3LatentContrast": MiniMaxH3LatentContrast,
    "MiniMaxH3Pass2StaggeredScheduler": MiniMaxH3Pass2StaggeredScheduler,
    "MiniMaxH3VAEDecodeFast": MiniMaxH3VAEDecodeFast,
    "MiniMaxH3LQPackageSave": MiniMaxH3LQPackageSave,
    "MiniMaxH3LQPackageLoad": MiniMaxH3LQPackageLoad,
    "MiniMaxH3UpscaleCollect": MiniMaxH3UpscaleCollect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentUpscaleCombined": "MiniMax H3 Latent Upscale Combined 2x",
    "MiniMaxH3LatentBlur": "MiniMax H3 Latent Blur",
    "MiniMaxH3LatentSharpen": "MiniMax H3 Latent Sharpen",
    "MiniMaxH3LatentContrast": "MiniMax H3 Latent Contrast",
    "MiniMaxH3Pass2StaggeredScheduler": "MiniMax H3 Pass-2 Staggered Scheduler",
    "MiniMaxH3VAEDecodeFast": "MiniMax H3 VAE Decode (fast)",
    "MiniMaxH3LQPackageSave": "MiniMax H3 Save Package",
    "MiniMaxH3LQPackageLoad": "MiniMax H3 Load Package",
    "MiniMaxH3UpscaleCollect": "MiniMax H3 Upscale Collect",
}

register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
