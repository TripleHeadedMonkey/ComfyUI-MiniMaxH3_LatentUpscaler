"""Front-loaded pass-2 sigma schedule: kill the upscale grid in one aggressive early
step, then spend the remaining (much smaller) step budget on texture.

Why not just swap BasicScheduler's `scheduler` name to karras/exponential:
BasicScheduler's denoise<1 trim (`total_steps = steps/denoise`, then take the last
`steps+1` sigmas of a `total_steps`-long schedule) lands at a very different
sigmas[0] depending on the schedule's shape, because karras/exponential interpolate
directly in raw sigma space (see comfy.samplers.SCHEDULER_HANDLERS use_ms=False)
while normal/sgm_uniform interpolate in the model's timestep space first. The "same"
steps/denoise numbers can give a much lower (weaker) starting sigma under karras than
under normal - silently under-denoising the grid instead of removing it.

This node keeps sigmas[0] pinned to whatever a proven base_scheduler/reference_steps/
denoise recipe reaches (e.g. your working normal/8/0.45), then builds a *separate*,
much shorter, front-loaded curve from that fixed sigma down to ~0.
"""

from __future__ import annotations

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.samplers
import torch

SHAPES = ("karras", "exponential", "polyexponential")
_SIGMA_MIN_FLOOR = 1e-3


def reference_sigma_start(model, base_scheduler: str, reference_steps: int, denoise: float) -> float:
    """Same trim BasicScheduler(denoise<1) uses, but we only need sigmas[0] off the tail."""
    reference_steps = max(1, int(reference_steps))
    denoise = float(denoise)
    if denoise <= 0.0:
        return 0.0
    total_steps = reference_steps if denoise >= 1.0 else int(reference_steps / denoise)
    total_steps = max(total_steps, reference_steps)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), base_scheduler, total_steps
    )
    tail = sigmas[-(reference_steps + 1):]
    return float(tail[0])


def staggered_sigmas(
    model,
    steps: int,
    denoise: float,
    rho: float,
    shape: str,
    *,
    reference_steps: int = 8,
    base_scheduler: str = "normal",
) -> torch.Tensor:
    """Front-loaded [sigma_start .. ~0] curve with a fixed, denoise-anchored sigma_start."""
    sigma_start = reference_sigma_start(model, base_scheduler, reference_steps, denoise)
    if sigma_start <= 0.0:
        return torch.FloatTensor([])

    model_sampling = model.get_model_object("model_sampling")
    sigma_min = max(float(getattr(model_sampling, "sigma_min", _SIGMA_MIN_FLOOR)), _SIGMA_MIN_FLOOR)
    sigma_min = min(sigma_min, sigma_start * 0.5)  # keep a visible last step even for a small sigma_start

    steps = max(1, int(steps))
    if shape == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(steps, sigma_min, sigma_start)
    elif shape == "polyexponential":
        sigmas = k_diffusion_sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_start, rho)
    else:
        sigmas = k_diffusion_sampling.get_sigmas_karras(steps, sigma_min, sigma_start, rho)
    return sigmas.cpu()


class MiniMaxH3Pass2StaggeredScheduler:
    """SIGMAS for pass 2: pin the proven denoise starting sigma, front-load the steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 100,
                        "tooltip": (
                            "Total pass-2 sampler steps. The staggered shape front-loads the sigma "
                            "drop into the first step(s), so this can be much lower than an evenly "
                            "spaced schedule (e.g. BasicScheduler normal) at the same denoise."
                        ),
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Same meaning as BasicScheduler's denoise, measured against "
                            "base_scheduler/reference_steps. Keep whatever value already killed "
                            "the grid in your normal/8-step recipe — this only changes how the "
                            "steps between that sigma and 0 are spent, not the starting sigma."
                        ),
                    },
                ),
                "rho": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.5,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": (
                            "Higher = more of the sigma drop happens in step 1 (more aggressive grid "
                            "removal, less budget left for texture). Lower = closer to evenly spaced. "
                            "Ignored for shape=exponential."
                        ),
                    },
                ),
                "shape": (
                    list(SHAPES),
                    {
                        "default": "karras",
                        "tooltip": (
                            "karras / polyexponential: rho controls how front-loaded the drop is. "
                            "exponential: fixed log-uniform front-loading (no rho)."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 10000,
                        "tooltip": (
                            "The steps value from the BasicScheduler recipe that already worked for "
                            "you (e.g. 8). Only used to anchor the starting sigma — does not affect "
                            "the actual number of pass-2 sampler steps (use 'steps' for that)."
                        ),
                    },
                ),
                "base_scheduler": (
                    comfy.samplers.SCHEDULER_NAMES,
                    {
                        "default": "normal",
                        "tooltip": "scheduler name from that proven recipe. Only used to anchor the starting sigma.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "run"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Pass-2 sigma schedule for MiniMax H3 latent-upscale continuation. Anchors sigmas[0] to "
        "the same starting noise level your base_scheduler/reference_steps/denoise recipe reaches "
        "(e.g. normal/8/0.45), then spends only 'steps' sampler steps on a front-loaded karras/"
        "exponential/polyexponential curve: one aggressive early drop to overwrite the nearest-exact "
        "upscale grid, then progressively finer steps for texture. Do not get this starting sigma by "
        "swapping the scheduler name inside a single BasicScheduler trim — karras/exponential "
        "interpolate in raw sigma space, so the 'same' steps/denoise reaches a very different "
        "(usually much weaker) sigmas[0] than normal does, which under-denoises the grid instead of "
        "removing it."
    )

    def run(self, model, steps, denoise, rho, shape, reference_steps=8, base_scheduler="normal"):
        sigmas = staggered_sigmas(
            model,
            steps,
            denoise,
            rho,
            shape,
            reference_steps=reference_steps,
            base_scheduler=base_scheduler,
        )
        return (sigmas,)
