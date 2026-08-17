"""Save decoded HQ frames into the originating package folder."""

from __future__ import annotations

import os
from pathlib import Path

from .stash import (
    H3_FPS,
    PACKAGE_PIPE_TYPE,
    output_root,
    parse_package_pipe,
    resolve_package_dir,
)
from .stash_media import next_upscaled_path
from .stash_preview import encode_images_to_mp4


class MiniMaxH3UpscaleCollect:
    """Encode pass-2 IMAGE into the source package as <name>_upscaled_NNN.mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "package_pipe": (
                    PACKAGE_PIPE_TYPE,
                    {
                        "tooltip": (
                            "Wire from MiniMax H3 Load Package. Identifies the source "
                            "package folder under ComfyUI/output."
                        ),
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Decoded HQ frames (VAE decode of pass 2).",
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional audio muxed into the HQ MP4.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 51,
                        "tooltip": (
                            "H.264 CRF. Lower is higher quality / larger files. "
                            "18 is a good production default."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "collect"
    CATEGORY = "latent/minimax_h3"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save the upscaled clip into its source package folder as "
        "<package>_upscaled_001.mp4 (then _002, …). Open Timeline on this node "
        "to edit the Load Package cut, right-click clips to swap LQ/HQ versions, "
        "and export MP4 / FCP7 / bundle."
    )

    def collect(self, package_pipe, images, audio=None, crf=18, **_unused):
        pipe = parse_package_pipe(package_pipe)
        identity = pipe.get("package_rel") or pipe.get("output_rel")
        package_dir = resolve_package_dir(identity)
        dest = next_upscaled_path(package_dir)
        partial = dest.with_name(dest.name + ".part")
        try:
            if partial.exists():
                partial.unlink()
            stats = encode_images_to_mp4(
                partial,
                images,
                audio=audio,
                fps=H3_FPS,
                crf=int(crf),
                preset="medium",
            )
            os.replace(partial, dest)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

        rel = dest.resolve().relative_to(output_root()).as_posix()
        subfolder = str(Path(rel).parent).replace("\\", "/")
        print(f"Upscale Collect wrote {rel}")
        preview = {
            "filename": dest.name,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/h264-mp4",
            "frame_rate": int(H3_FPS),
            "has_audio": bool(stats.get("has_audio")),
            "width": int(stats.get("width") or 0),
            "height": int(stats.get("height") or 0),
        }
        return {
            "ui": {
                "gifs": [preview],
                "h3_collect_preview": [preview],
            },
            "result": (rel,),
        }
