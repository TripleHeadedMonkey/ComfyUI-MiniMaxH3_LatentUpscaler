"""ComfyUI node: MiniMax H3 NestedTensor latent upscale + CONST re-noise."""

from __future__ import annotations

from .learned import NO_CHECKPOINT, available_checkpoints, load_learned_upscaler
from .utils import (
    LEARNED_UPSCALE_METHOD,
    NOISE_RESAMPLE_MODES,
    UPSCALE_METHODS,
    blur_nested_latent,
    contrast_nested_latent,
    is_learned_upscale_method,
    sharpen_nested_latent,
    upscale_and_add_noise,
    upscale_minimax_conditioning,
)


class MiniMaxH3LatentUpscaleCombined:
    """Upscale NestedTensor video latents + MiniMax ref/keyframe cond, then re-noise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "method": (
                    list(UPSCALE_METHODS),
                    {
                        "default": LEARNED_UPSCALE_METHOD,
                        "tooltip": (
                            "learned model (default) predicts a clean on-manifold 2x H3 "
                            "latent from learned_model; match_stats is bypassed for it. "
                            "Interpolation methods remain as fallbacks: nearest / "
                            "nearest-exact / area are identical at 2x, and bislerp is the only "
                            "latent-aware interpolation. Use the standalone Latent Blur / "
                            "Sharpen / Contrast nodes after Combined if you need them."
                        ),
                    },
                ),
                "learned_model": (
                    available_checkpoints() or [NO_CHECKPOINT],
                    {
                        "tooltip": (
                            "Checkpoint from ComfyUI/models/h3_latent_upscalers. The default "
                            "checkpoint downloads automatically on its first learned-model run."
                        )
                    },
                ),
                "model": ("MODEL",),
                "noise": ("NOISE",),
                "sigmas": ("SIGMAS",),
                "audio_denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "0 = lock pass-1 audio (zero audio noise + audio noise_mask 0). "
                            "1 = full re-noise at sigmas[0] so sampler 2 can rewrite audio. "
                            "Values below 0.5 lock; 0.5 and above remix. "
                            "Partial remix is not a lighter denoise."
                        ),
                    },
                ),
                "noise_resample": (
                    list(NOISE_RESAMPLE_MODES),
                    {
                        "default": "independent",
                        "tooltip": (
                            "Kept for compatibility. Both modes draw iid noise on the 2× "
                            "grid. Do not use nearest-copied low-res noise with H3."
                        ),
                    },
                ),
            },
            "optional": {
                "learned_upscaler": (
                    "H3_LATENT_UPSCALER",
                    {
                        "tooltip": (
                            "Optional override. Wire Load H3 Latent Upscaler if you already "
                            "have one in the graph; otherwise Combined loads learned_model."
                        ),
                    },
                ),
                "match_stats": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "Interpolation fallback only; ignored by learned model. Strength "
                            "(0-1) for matching each upscaled latent channel to pass-1 mean/std. "
                            "Leave at 0 for the trained upscaler: its output distribution is "
                            "learned and must not be globally stretched."
                        ),
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "upscale_noise"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Hard-locked 2x MiniMax H3 clean-latent upscale + CONST re-noise for pass 2. "
        "Default learned model predicts the pixel-upscale/re-encode teacher latent and bypasses "
        "match_stats. Blur / sharpen / contrast are separate nodes. "
        "Interpolation remains as a fallback. "
        "Use a low pass-2 sigma start (~0.25–0.45). Video is re-noised at sigmas[0]. "
        "audio_denoise 0 locks pass-1 audio (noise_mask 0 on the audio stream). "
        "Optionally upscale minimax_refs / keyframes — rebuild Guider from returned CONDITIONING."
    )

    def upscale_noise(
        self,
        samples,
        method,
        model,
        noise,
        sigmas,
        learned_model=None,
        learned_upscaler=None,
        audio_denoise=0.0,
        noise_resample="independent",
        match_stats=0.0,
        positive=None,
        negative=None,
    ):
        scale_by = 2.0
        upscaler = None
        if is_learned_upscale_method(method):
            if learned_upscaler is not None:
                print(
                    f"[MiniMax H3 Combined] method={method!r} using wired learned_upscaler"
                )
                upscaler = learned_upscaler
            else:
                print(
                    f"[MiniMax H3 Combined] method={method!r} checkpoint={learned_model!r}"
                )
                upscaler = load_learned_upscaler(learned_model)
        else:
            print(
                f"[MiniMax H3 Combined] method={method!r} "
                "(interpolation fallback, learned model not used)"
            )
        latent = upscale_and_add_noise(
            samples,
            scale_by,
            method,
            model,
            noise,
            sigmas,
            audio_denoise=audio_denoise,
            noise_resample=noise_resample,
            match_stats=match_stats,
            upscaler=upscaler,
        )
        # The learned checkpoint is trained for the sampled Bx24xTxHxW video stream.
        # MiniMax ref/keyframe latents use separate 4D/metadata layouts, so resize
        # those geometrically while keeping their dimensions synchronized.
        conditioning_method = "nearest-exact" if is_learned_upscale_method(method) else method
        pos_out = upscale_minimax_conditioning(positive, scale_by, conditioning_method)
        neg_out = upscale_minimax_conditioning(negative, scale_by, conditioning_method)
        return (latent, pos_out, neg_out)


class MiniMaxH3LatentBlur:
    """Standalone norm-preserving latent blur for H3 NestedTensor video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "radius": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "Norm-preserving smoothing radius in latent pixels. 0 disables. "
                            "Only each token's direction is smoothed; its 24-D magnitude is "
                            "restored. Raise to ~0.4-0.6 only if you see a hard 2x2 grid — "
                            "smoothed directions are not guaranteed on-manifold."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blur"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Norm-preserving spatial smooth on the H3 video latent only "
        "(audio stream unchanged). Former Combined blur control."
    )

    def blur(self, samples, radius=0.0):
        return (blur_nested_latent(samples, radius),)


class MiniMaxH3LatentSharpen:
    """Standalone norm-preserving latent unsharp mask for H3 NestedTensor video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "Unsharp strength. 0 disables. Start around 0.15-0.30. "
                            "High values can produce outlines or strings because changed "
                            "token directions are not guaranteed to stay on-manifold."
                        ),
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.05,
                        "max": 8.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "Gaussian low-pass radius in latent pixels. Combined uses 1.0."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sharpen"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Norm-preserving directional unsharp on the H3 video latent only "
        "(audio stream unchanged)."
    )

    def sharpen(self, samples, amount=0.2, radius=1.0):
        return (sharpen_nested_latent(samples, amount, radius=radius),)


class MiniMaxH3LatentContrast:
    """Standalone per-channel latent contrast for H3 NestedTensor video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": (
                            "1.0 = unchanged. >1 stretches each channel away from its "
                            "global mean; <1 compresses. Typical gentle boost: 1.05-1.20."
                        ),
                    },
                ),
                "preserve_norm": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "On (default): restore each 24-D token's magnitude after the "
                            "contrast stretch so edge tokens do not become oversized. "
                            "Off: allow channel energy to change (closer to a raw "
                            "match_stats-style stretch)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "contrast"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Per-channel contrast around each channel's global mean on the H3 video "
        "latent only (audio unchanged). Defaults to norm-preserving for pass-2 safety."
    )

    def contrast(self, samples, contrast=1.0, preserve_norm=True):
        return (contrast_nested_latent(samples, contrast, preserve_norm=preserve_norm),)
