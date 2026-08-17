"""Load the trained H3 clean-latent 2x upscaler from the sibling reference pack.

Architecture and checkpoint contract are imported from that pack rather than
vendored, so there is one source of truth if it is updated. Loading uses the
legacy Comfy ModelPatcher (not DynamicVRAM / CoreModelPatcher) so vanilla
Conv3d weights actually land on GPU.
"""

from __future__ import annotations

import importlib.util
import hashlib
import logging
import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any

MODEL_FOLDER = "h3_latent_upscalers"
DEFAULT_CHECKPOINT = "h3_clean_latent_upscaler_film_epoch200.safetensors"
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/Tridae/H3LatentUpscaler/resolve/main/"
    f"{DEFAULT_CHECKPOINT}?download=true"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "984afb58f11d01274b90d880596ce2f93bc9c512db66fdeecd4e2d99b371d3e4"
)
# Sibling custom_nodes folder that provides upscaler.py / latent_io.py (install path; see README).
REFERENCE_PACK = "ComfyUI-H3-Latent-Upscaler-Mamad8"
_REFERENCE_MODULE = "comfyui_minimaxh3_latentupscaler._learned_upscaler"
_REFERENCE_LATENT_IO = "comfyui_minimaxh3_latentupscaler._learned_latent_io"
NO_CHECKPOINT = "(no checkpoint installed)"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_download_lock = threading.Lock()
log = logging.getLogger(__name__)

# One entry, keyed by path+mtime: switching checkpoints drops the previous patcher.
_patcher_cache: dict[str, Any] = {}


def _load_reference(filename: str, module_name: str):
    """Import a file out of the sibling reference pack."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    source = Path(__file__).resolve().parent.parent / REFERENCE_PACK / filename
    if not source.is_file():
        raise FileNotFoundError(
            f"Cannot find {source}. Install the reference H3 latent-upscaler "
            f"architecture pack under ComfyUI/custom_nodes/{REFERENCE_PACK} "
            f"(see README for the original repo)."
        )
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _reference_module():
    """Import upscaler.py out of the sibling reference pack."""
    return _load_reference("upscaler.py", _REFERENCE_MODULE)


def _reference_latent_io():
    """Import latent_io.py (split/replace H3 video streams) from the sibling pack."""
    return _load_reference("latent_io.py", _REFERENCE_LATENT_IO)


def register_model_folder() -> None:
    """Make ComfyUI/models/h3_latent_upscalers discoverable even if we load first."""
    import folder_paths

    if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            MODEL_FOLDER,
            os.path.join(folder_paths.models_dir, MODEL_FOLDER),
            is_default=True,
        )
    folder_paths.folder_names_and_paths[MODEL_FOLDER][1].add(".safetensors")


def available_checkpoints() -> list[str]:
    """Checkpoint names for the widget. Never raises during schema construction."""
    try:
        import folder_paths

        register_model_folder()
        installed = list(folder_paths.get_filename_list(MODEL_FOLDER))
        # Keep the bundled default selectable before it exists. It is downloaded
        # lazily on first execution so schema construction never blocks on network.
        return [DEFAULT_CHECKPOINT, *[name for name in installed if name != DEFAULT_CHECKPOINT]]
    except Exception:
        return [DEFAULT_CHECKPOINT]


def _default_checkpoint_path() -> Path:
    """Return the default model destination in ComfyUI's first registered folder."""
    import folder_paths

    register_model_folder()
    folders = folder_paths.get_folder_paths(MODEL_FOLDER)
    if not folders:
        raise RuntimeError(f"No model folder registered for {MODEL_FOLDER!r}")
    return Path(folders[0]) / DEFAULT_CHECKPOINT


def _download_default_checkpoint() -> Path:
    """Download and verify the default checkpoint if it is not installed."""
    destination = _default_checkpoint_path()
    if destination.is_file():
        return destination

    with _download_lock:
        if destination.is_file():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.part")
        log.warning(
            "H3 latent upscaler checkpoint not found; downloading %s to %s",
            DEFAULT_CHECKPOINT_URL,
            destination,
        )
        request = urllib.request.Request(
            DEFAULT_CHECKPOINT_URL,
            headers={"User-Agent": "ComfyUI-MiniMaxH3-LatentUpscaler"},
        )
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as out:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    out.write(chunk)
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != DEFAULT_CHECKPOINT_SHA256:
                raise RuntimeError(
                    f"Downloaded {DEFAULT_CHECKPOINT} failed SHA-256 verification "
                    f"(expected {DEFAULT_CHECKPOINT_SHA256}, got {actual_sha256})"
                )
            os.replace(partial, destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

        log.warning("Downloaded and verified H3 latent upscaler checkpoint: %s", destination)
        return destination


def load_learned_upscaler(model_name: str) -> Any:
    """Build (or reuse) the Comfy model patcher for a learned upscaler checkpoint."""
    import comfy.model_management as model_management
    import comfy.model_patcher
    import comfy.utils
    import folder_paths
    import torch

    register_model_folder()
    if not model_name or model_name == NO_CHECKPOINT:
        model_name = DEFAULT_CHECKPOINT
    if model_name == DEFAULT_CHECKPOINT:
        path = str(_download_default_checkpoint())
    else:
        path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, model_name)
    key = f"{path}:{os.path.getmtime(path)}:fp32"
    cached = _patcher_cache.get(key)
    if cached is not None:
        return cached

    reference = _reference_module()
    info = reference.read_checkpoint_info(path)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    # Aimdo mmap tensors are read-only views. Clone onto CPU so load_state_dict
    # owns real storage and DynamicVRAM cannot leave the correction branch empty.
    state = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in state.items()
    }
    model = reference.build_upscaler(state, info)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    weight_abs = next(model.parameters()).detach().float().abs().mean().item()
    del state

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    # ~27MB. unet_dtype() is for the 20GB DiT and will pick bf16; that wipes the
    # high-frequency residual epoch200 uses to cancel the 2x2 latent grid.
    dtype = torch.float32
    model.to(device=offload_device, dtype=dtype)
    model.h3_upscaler_step = info.step
    # CoreModelPatcher is ModelPatcherDynamic when Aimdo is on. That loader is
    # for UNets with comfy.ops; this module is vanilla Conv3d / MHA. Dynamic
    # staging can leave the residual near zero, so the 2x looks like bilinear.
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )
    _patcher_cache.clear()
    _patcher_cache[key] = patcher
    log.warning(
        "H3 learned upscaler loaded %s (%s params, dtype=%s, |w|=%.5f)",
        os.path.basename(path),
        f"{parameter_count:,}",
        dtype,
        weight_abs,
    )
    print(
        f"[MiniMax H3 Combined] loaded learned 2x {os.path.basename(path)} "
        f"({parameter_count:,} params, {dtype}, |w|={weight_abs:.5f})"
    )
    return patcher


def apply_h3_latent_upscale(upscaler: Any, latent: dict) -> dict:
    """Apply the 2x upscaler the same way H3CleanLatentUpscale2x.execute does.

    Combined still CONST-re-noises after this. The sibling node stops at the
    clean 2x latent: find the unique Bx24xTxHxW stream, run the patcher at its
    own dtype (with CUDA autocast for fp16/bf16), replace only that stream, and
    drop noise_mask so a low-res mask cannot imprint a 2x2 grid on pass 2.
    """
    import math
    from contextlib import nullcontext

    import comfy.model_management as model_management
    import comfy.nested_tensor
    import torch
    import torch.nn.functional as F

    if upscaler is None:
        raise ValueError(
            "No learned upscaler loaded. Pick a checkpoint in learned_model "
            "(ComfyUI/models/h3_latent_upscalers/) or wire learned_upscaler."
        )
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a ComfyUI LATENT dictionary containing 'samples'")
    if not hasattr(upscaler, "model") or not hasattr(upscaler, "load_device"):
        raise TypeError(
            "learned_model did not produce a valid Comfy model patcher for the upscaler"
        )

    latent_io = _reference_latent_io()
    video, streams, video_index, original_container = latent_io.split_h3_video_samples(
        latent["samples"],
        torch_module=torch,
        nested_tensor_type=comfy.nested_tensor.NestedTensor,
    )
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            "Expected H3 video latent shape Bx24xTxHxW, "
            f"got {tuple(video.shape)}"
        )
    if not video.is_floating_point():
        raise TypeError("H3 video latents must use a floating-point dtype")

    # Same conservative activation allowance as the sibling apply node.
    activation_memory = math.prod(video.shape) * max(video.element_size(), 2) * 512
    load_kwargs = {"memory_required": upscaler.model_size() + activation_memory}
    # Sibling uses CoreModelPatcher; force_full_load asserts on DynamicVRAM.
    # Combined's self-loader uses legacy ModelPatcher and needs a full GPU load.
    if not upscaler.is_dynamic():
        load_kwargs["force_full_load"] = True
    model_management.load_models_gpu([upscaler], **load_kwargs)

    model = upscaler.model
    device = upscaler.load_device
    weight = next(model.parameters())
    if weight.device.type == "cpu" and device.type != "cpu":
        raise RuntimeError(
            "Learned upscaler weights are still on CPU after load_models_gpu. "
            "The 2x would collapse to bilinear. Restart ComfyUI after this update."
        )
    model_dtype = weight.dtype
    source_dtype = video.dtype
    video_input = video.to(device=device, dtype=model_dtype)
    use_autocast = device.type == "cuda" and model_dtype in (
        torch.float16,
        torch.bfloat16,
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=model_dtype)
        if use_autocast
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        upscaled = model(video_input)

    expected_shape = (*video.shape[:-2], video.shape[-2] * 2, video.shape[-1] * 2)
    if tuple(upscaled.shape) != expected_shape:
        raise RuntimeError(
            f"Upscaler returned {tuple(upscaled.shape)}, expected {expected_shape}"
        )

    b, c, t, h, w = video_input.shape
    bilinear = video_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    bilinear = F.interpolate(
        bilinear,
        size=(h * 2, w * 2),
        mode="bilinear",
        align_corners=False,
    )
    bilinear = bilinear.view(b, t, c, h * 2, w * 2).permute(0, 2, 1, 3, 4)
    residual_ratio = (
        (upscaled.float() - bilinear.float()).abs().mean()
        / bilinear.float().abs().mean().clamp_min(1e-6)
    ).item()
    print(
        f"[MiniMax H3 Combined] mamad8-apply stream={video_index} "
        f"shape={tuple(video.shape)} residual/bilinear={residual_ratio:.4f} "
        f"device={weight.device} dtype={model_dtype} autocast={use_autocast}"
    )
    if residual_ratio < 0.005:
        raise RuntimeError(
            "Learned 2x residual is nearly zero "
            f"(residual/bilinear={residual_ratio:.4f}); output would look like bilinear. "
            "The checkpoint did not actually correct the upsample."
        )

    upscaled = upscaled.to(
        device=model_management.intermediate_device(),
        dtype=source_dtype,
    )
    output = latent.copy()
    output["samples"] = latent_io.replace_h3_video_samples(
        original_container,
        streams,
        video_index,
        upscaled,
        nested_tensor_type=comfy.nested_tensor.NestedTensor,
    )
    # A low-resolution spatial mask cannot describe the new latent grid.
    output.pop("noise_mask", None)
    return output
