"""PromptServer HTTP routes for the H3 saved-packages gallery browser."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from .stash import (
    META_FILE,
    STASH_ROOT_NAME,
    is_package_dir,
    output_root,
    resolve_package_dir,
    stash_root,
)
from .stash_media import resolve_package_media, variant_payload
from .stash_preview import PREVIEW_FILE, THUMB_FILE

LOG = logging.getLogger("h3_lq_stash.server")

# 1x1 JPEG placeholder (gray)
_PLACEHOLDER_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xDB,
        0x00,
        0x43,
        0x00,
        0x08,
        0x06,
        0x06,
        0x07,
        0x06,
        0x05,
        0x08,
        0x07,
        0x07,
        0x07,
        0x09,
        0x09,
        0x08,
        0x0A,
        0x0C,
        0x14,
        0x0D,
        0x0C,
        0x0B,
        0x0B,
        0x0C,
        0x19,
        0x12,
        0x13,
        0x0F,
        0x14,
        0x1D,
        0x1A,
        0x1F,
        0x1E,
        0x1D,
        0x1A,
        0x1C,
        0x1C,
        0x20,
        0x24,
        0x2E,
        0x27,
        0x20,
        0x22,
        0x2C,
        0x23,
        0x1C,
        0x1C,
        0x28,
        0x37,
        0x29,
        0x2C,
        0x30,
        0x31,
        0x34,
        0x34,
        0x34,
        0x1F,
        0x27,
        0x39,
        0x3D,
        0x38,
        0x32,
        0x3C,
        0x2E,
        0x33,
        0x34,
        0x32,
        0xFF,
        0xC0,
        0x00,
        0x0B,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x01,
        0x01,
        0x11,
        0x00,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x08,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x10,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xFF,
        0xDA,
        0x00,
        0x08,
        0x01,
        0x01,
        0x00,
        0x00,
        0x3F,
        0x00,
        0x7F,
        0xFF,
        0xD9,
    ]
)


def _json_error(message: str, status: int = 400):
    return web.json_response({"ok": False, "error": message}, status=status)


def _safe_under_output(path: Path) -> Path:
    out = output_root()
    resolved = path.resolve()
    try:
        resolved.relative_to(out)
    except ValueError as exc:
        raise ValueError(f"Path escapes output directory: {path}") from exc
    return resolved


def _normalize_rel(value: str) -> str:
    raw = (value or "").strip().replace("\\", "/")
    if not raw or raw in (".", "./"):
        return ""
    parts = [p for p in Path(raw).parts if p not in ("", ".", "/")]
    if ".." in parts:
        raise ValueError("Path must not contain '..'")
    return Path(*parts).as_posix() if parts else ""


def _resolve_browse_dir(root_param: str, path_param: str) -> tuple[Path, str, str]:
    """Return (absolute dir, root_rel under output, path_rel under root)."""
    root_rel = _normalize_rel(root_param) or STASH_ROOT_NAME
    path_rel = _normalize_rel(path_param)
    base = _safe_under_output(output_root() / root_rel)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        base = _safe_under_output(base)
    target = _safe_under_output(base / path_rel) if path_rel else base
    if not target.is_dir():
        raise FileNotFoundError(f"Not a directory: {root_rel}/{path_rel}".rstrip("/"))
    return target, root_rel, path_rel


def _package_rel_from_param(package: str) -> str:
    rel = _normalize_rel(package)
    if not rel:
        raise ValueError("package is required")
    return rel


def _read_meta(package_dir: Path) -> dict:
    meta_path = package_dir / META_FILE
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


async def tree_handler(request: web.Request):
    try:
        root_param = request.rel_url.query.get("root") or STASH_ROOT_NAME
        path_param = request.rel_url.query.get("path") or ""
        target, root_rel, path_rel = _resolve_browse_dir(root_param, path_param)
    except ValueError as e:
        return _json_error(str(e), status=400)
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except Exception as e:
        LOG.exception("tree failed")
        return _json_error(str(e), status=500)

    folders = []
    for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        # Skip package dirs in the tree (they show in list)
        if is_package_dir(child):
            continue
        try:
            _safe_under_output(child)
        except ValueError:
            continue
        child_rel = f"{path_rel}/{child.name}".lstrip("/") if path_rel else child.name
        folders.append({"name": child.name, "path": child_rel})

    return web.json_response(
        {
            "ok": True,
            "root": root_rel,
            "path": path_rel,
            "folders": folders,
        }
    )


async def list_handler(request: web.Request):
    try:
        root_param = request.rel_url.query.get("root") or STASH_ROOT_NAME
        path_param = request.rel_url.query.get("path") or ""
        target, root_rel, path_rel = _resolve_browse_dir(root_param, path_param)
    except ValueError as e:
        return _json_error(str(e), status=400)
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except Exception as e:
        LOG.exception("list failed")
        return _json_error(str(e), status=500)

    # package_rel is always relative to stash root (h3_lq_stash), for Load node
    stash = stash_root()
    packages = []
    for child in sorted(
        target.iterdir(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    ):
        if not is_package_dir(child):
            continue
        try:
            child = _safe_under_output(child)
            package_rel = child.relative_to(stash).as_posix()
        except ValueError:
            # Outside default stash root but under output — use path relative to output
            try:
                package_rel = child.relative_to(output_root()).as_posix()
            except ValueError:
                continue

        meta = _read_meta(child)
        prompt = str(meta.get("prompt") or "")
        note = str(meta.get("note") or "")
        has_preview = bool(meta.get("has_preview")) or (child / PREVIEW_FILE).is_file()
        has_thumb = bool(meta.get("has_thumb")) or (child / THUMB_FILE).is_file()
        packages.append(
            {
                "name": child.name,
                "package": package_rel,
                "note": note,
                "prompt_snippet": prompt[:160],
                "has_preview": has_preview,
                "has_thumb": has_thumb,
                "thumb_url": f"/h3_lq_stash/thumb?package={package_rel}",
                "preview_url": f"/h3_lq_stash/preview?package={package_rel}"
                if has_preview
                else None,
                "mtime": child.stat().st_mtime,
                "created_at": meta.get("created_at"),
                "video_shape": meta.get("video_shape"),
                "audio_shape": meta.get("audio_shape"),
                "frame_count": meta.get("frame_count"),
                "fps": meta.get("fps"),
                "upscaled": variant_payload(
                    package_rel, child, fps=float(meta.get("fps") or 24.0), probe=False
                ),
            }
        )

    return web.json_response(
        {
            "ok": True,
            "root": root_rel,
            "path": path_rel,
            "packages": packages,
        }
    )


async def meta_handler(request: web.Request):
    try:
        package = _package_rel_from_param(request.rel_url.query.get("package") or "")
        package_dir = resolve_package_dir(package)
        # Also allow packages under output that aren't under stash root
        if not (package_dir / META_FILE).is_file():
            alt = _safe_under_output(output_root() / package)
            if (alt / META_FILE).is_file():
                package_dir = alt
        meta_path = package_dir / META_FILE
        if not meta_path.is_file():
            return _json_error(f"meta not found: {package}", status=404)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return web.json_response({"ok": True, "package": package, "meta": meta})
    except ValueError as e:
        return _json_error(str(e), status=400)
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except Exception as e:
        LOG.exception("meta failed")
        return _json_error(str(e), status=500)


def _resolve_media_package(package: str) -> Path:
    try:
        package_dir = resolve_package_dir(package)
        if package_dir.is_dir():
            return _safe_under_output(package_dir)
    except Exception:
        pass
    return _safe_under_output(output_root() / _normalize_rel(package))


async def thumb_handler(request: web.Request):
    try:
        package = _package_rel_from_param(request.rel_url.query.get("package") or "")
        package_dir = _resolve_media_package(package)
        thumb = package_dir / THUMB_FILE
        if thumb.is_file():
            return web.FileResponse(path=thumb, headers={"Content-Type": "image/jpeg"})
        return web.Response(body=_PLACEHOLDER_JPEG, content_type="image/jpeg")
    except ValueError as e:
        return _json_error(str(e), status=400)
    except Exception as e:
        LOG.exception("thumb failed")
        return _json_error(str(e), status=500)


async def preview_handler(request: web.Request):
    try:
        package = _package_rel_from_param(request.rel_url.query.get("package") or "")
        package_dir = _resolve_media_package(package)
        preview = package_dir / PREVIEW_FILE
        if not preview.is_file():
            return _json_error("preview.mp4 not found", status=404)
        return web.FileResponse(path=preview, headers={"Content-Type": "video/mp4"})
    except ValueError as e:
        return _json_error(str(e), status=400)
    except Exception as e:
        LOG.exception("preview failed")
        return _json_error(str(e), status=500)


async def media_handler(request: web.Request):
    try:
        package = _package_rel_from_param(request.rel_url.query.get("package") or "")
        filename = request.rel_url.query.get("file") or PREVIEW_FILE
        media = resolve_package_media(package, filename)
        if not media.is_file():
            return _json_error(f"media not found: {filename}", status=404)
        return web.FileResponse(path=media, headers={"Content-Type": "video/mp4"})
    except ValueError as e:
        return _json_error(str(e), status=400)
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except Exception as e:
        LOG.exception("media failed")
        return _json_error(str(e), status=500)


async def variants_handler(request: web.Request):
    try:
        package = _package_rel_from_param(request.rel_url.query.get("package") or "")
        package_dir = _resolve_media_package(package)
        meta = _read_meta(package_dir)
        fps = float(meta.get("fps") or 24.0)
        has_preview = (package_dir / PREVIEW_FILE).is_file()
        return web.json_response(
            {
                "ok": True,
                "package": package,
                "name": package_dir.name,
                "has_preview": has_preview,
                "has_thumb": (package_dir / THUMB_FILE).is_file(),
                "thumb_url": f"/h3_lq_stash/thumb?package={package}",
                "preview_url": (
                    f"/h3_lq_stash/preview?package={package}" if has_preview else None
                ),
                "frame_count": meta.get("frame_count"),
                "fps": meta.get("fps"),
                "upscaled": variant_payload(package, package_dir, fps=fps),
            }
        )
    except ValueError as e:
        return _json_error(str(e), status=400)
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except Exception as e:
        LOG.exception("variants failed")
        return _json_error(str(e), status=500)


def _sanitize_export_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^\w.\-]+", "_", (value or "edit").strip())
    return (cleaned.strip("._") or "edit")[:96]


async def export_mp4_handler(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return _json_error("Invalid JSON body")
    clips = body.get("clips") or []
    fps = float(body.get("fps") or 24.0)
    name = _sanitize_export_name(str(body.get("name") or "edit"))
    if not isinstance(clips, list) or not clips:
        return _json_error("clips must be a non-empty list")

    loop = asyncio.get_event_loop()
    try:
        from .stash_edit import export_mp4

        result = await loop.run_in_executor(
            None,
            lambda: export_mp4(clips, fps=fps, name=name),
        )
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except ValueError as e:
        return _json_error(str(e), status=400)
    except Exception as e:
        LOG.exception("export_mp4 failed")
        return _json_error(str(e), status=500)
    return web.json_response({"ok": True, **result})


async def export_xml_handler(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return _json_error("Invalid JSON body")
    clips = body.get("clips") or []
    fps = float(body.get("fps") or 24.0)
    name = _sanitize_export_name(str(body.get("name") or "edit"))
    if not isinstance(clips, list) or not clips:
        return _json_error("clips must be a non-empty list")

    loop = asyncio.get_event_loop()
    try:
        from .stash_edit import build_fcp7_xml

        result = await loop.run_in_executor(
            None,
            lambda: build_fcp7_xml(clips, fps=fps, name=name),
        )
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except ValueError as e:
        return _json_error(str(e), status=400)
    except Exception as e:
        LOG.exception("export_xml failed")
        return _json_error(str(e), status=500)
    return web.json_response({"ok": True, **result})


async def export_bundle_handler(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return _json_error("Invalid JSON body")
    clips = body.get("clips") or []
    fps = float(body.get("fps") or 24.0)
    name = _sanitize_export_name(str(body.get("name") or "edit"))
    if not isinstance(clips, list) or not clips:
        return _json_error("clips must be a non-empty list")

    loop = asyncio.get_event_loop()
    try:
        from .stash_edit import export_bundle

        result = await loop.run_in_executor(
            None,
            lambda: export_bundle(clips, fps=fps, name=name),
        )
    except FileNotFoundError as e:
        return _json_error(str(e), status=404)
    except ValueError as e:
        return _json_error(str(e), status=400)
    except Exception as e:
        LOG.exception("export_bundle failed")
        return _json_error(str(e), status=500)
    return web.json_response({"ok": True, **result})


def register_routes():
    try:
        from server import PromptServer
    except Exception:
        LOG.warning("PromptServer unavailable; H3 package routes not registered")
        return

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        LOG.warning("PromptServer.instance not ready; H3 package routes not registered")
        return

    if getattr(PromptServer, "_h3_lq_stash_routes_version", 0) >= 4:
        return

    routes = instance.routes
    routes.get("/h3_lq_stash/tree")(tree_handler)
    routes.get("/h3_lq_stash/list")(list_handler)
    routes.get("/h3_lq_stash/meta")(meta_handler)
    routes.get("/h3_lq_stash/thumb")(thumb_handler)
    routes.get("/h3_lq_stash/preview")(preview_handler)
    routes.get("/h3_lq_stash/media")(media_handler)
    routes.get("/h3_lq_stash/variants")(variants_handler)
    routes.post("/h3_lq_stash/export_mp4")(export_mp4_handler)
    routes.post("/h3_lq_stash/export_xml")(export_xml_handler)
    routes.post("/h3_lq_stash/export_bundle")(export_bundle_handler)

    PromptServer._h3_lq_stash_routes_version = 4
    LOG.info("Registered H3 package gallery + edit export routes")
