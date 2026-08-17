"""Focused tests for package provenance, HQ media, and Collect path allocation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_package():
    import types

    pkg = types.ModuleType("h3pack")
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = "h3pack"
    sys.modules["h3pack"] = pkg

    def load_sub(name):
        spec = importlib.util.spec_from_file_location(
            f"h3pack.{name}",
            ROOT / f"{name}.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"h3pack.{name}"] = module
        spec.loader.exec_module(module)
        return module

    stash = load_sub("stash")
    media = load_sub("stash_media")
    edit = load_sub("stash_edit")
    return stash, media, edit


def _install_temp_output(stash_mod, tmp: Path):
    output = tmp / "output"
    stash_dir = output / "h3_lq_stash"
    stash_dir.mkdir(parents=True)
    stash_mod.output_root = lambda: output.resolve()
    stash_mod.stash_root = lambda: stash_dir.resolve()
    return output, stash_dir


def test_canonical_paths_and_pipe(stash):
    with tempfile.TemporaryDirectory() as tmp:
        _output, stash_dir = _install_temp_output(stash, Path(tmp))
        pkg = stash_dir / "testStash" / "android_conservatory_00005"
        pkg.mkdir(parents=True)
        assert stash.package_rel_path(pkg) == "testStash/android_conservatory_00005"
        assert stash.package_output_rel(pkg) == "h3_lq_stash/testStash/android_conservatory_00005"
        assert (
            stash.canonical_package_name(pkg)
            == "h3_lq_stash/testStash/android_conservatory_00005"
        )
        assert (
            stash.canonical_package_name(pkg, scene_index=1, scene_count=3)
            == "h3_lq_stash/testStash/android_conservatory_00005#scene01"
        )
        pipe = stash.build_package_pipe(pkg, scene_index=1, scene_count=3)
        assert pipe["package_rel"] == "testStash/android_conservatory_00005"
        assert pipe["output_rel"] == "h3_lq_stash/testStash/android_conservatory_00005"
        assert "#" not in pipe["package_rel"]
        parsed = stash.parse_package_pipe(
            {
                "package_rel": "testStash/android_conservatory_00005#scene01",
                "output_rel": "h3_lq_stash/testStash/android_conservatory_00005#scene01",
            }
        )
        assert parsed["package_rel"] == "testStash/android_conservatory_00005"
        assert parsed["output_rel"] == "h3_lq_stash/testStash/android_conservatory_00005"


def test_parse_edit_json_preserves_media(stash):
    clips = stash.parse_edit_json(
        json.dumps(
            {
                "clips": [
                    {
                        "package": "testStash/clip_a",
                        "in_frame": 0,
                        "out_frame": 24,
                        "media": "clip_a_upscaled_002.mp4",
                    },
                    {
                        "package": "testStash/clip_a",
                        "in_frame": 0,
                        "out_frame": 12,
                        "media": "../secret.mp4",
                    },
                ]
            }
        )
    )
    assert clips[0]["media"] == "clip_a_upscaled_002.mp4"
    assert "media" not in clips[1]


def test_next_upscaled_and_variants(media):
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "android_conservatory_00005"
        pkg.mkdir()
        first = media.next_upscaled_path(pkg)
        assert first.name == "android_conservatory_00005_upscaled_001.mp4"
        first.write_bytes(b"one")
        (pkg / "preview.mp4").write_bytes(b"lq")
        (pkg / "android_conservatory_00005_upscaled_003.mp4").write_bytes(b"three")
        second = media.next_upscaled_path(pkg)
        assert second.name == "android_conservatory_00005_upscaled_002.mp4"
        names = [item["name"] for item in media.list_upscaled_variants(pkg)]
        assert names == [
            "android_conservatory_00005_upscaled_001.mp4",
            "android_conservatory_00005_upscaled_003.mp4",
        ]
        assert media.is_allowed_media_filename(pkg, "preview.mp4")
        assert media.is_allowed_media_filename(pkg, "android_conservatory_00005_upscaled_001.mp4")
        assert not media.is_allowed_media_filename(pkg, "../secret.mp4")
        assert not media.is_allowed_media_filename(pkg, "other_upscaled_001.mp4")


def test_media_traversal_rejected(stash, media):
    with tempfile.TemporaryDirectory() as tmp:
        _output, stash_dir = _install_temp_output(stash, Path(tmp))
        media.resolve_package_dir = stash.resolve_package_dir
        media.output_root = stash.output_root
        pkg = stash_dir / "proj" / "clip_a"
        pkg.mkdir(parents=True)
        (pkg / "meta.json").write_text("{}", encoding="utf-8")
        (pkg / "latent.safetensors").write_bytes(b"x")
        (pkg / "preview.mp4").write_bytes(b"lq")
        (pkg / "clip_a_upscaled_001.mp4").write_bytes(b"hq")
        ok = media.resolve_package_media("proj/clip_a", "clip_a_upscaled_001.mp4")
        assert ok.name == "clip_a_upscaled_001.mp4"
        try:
            media.resolve_package_media("proj/clip_a", "../clip_a_upscaled_001.mp4")
            raise AssertionError("traversal was accepted")
        except ValueError:
            pass
        try:
            media.resolve_package_media("proj/clip_a", "not_a_variant.mp4")
            raise AssertionError("random mp4 was accepted")
        except ValueError:
            pass


def test_load_outputs_keep_indices(stash):
    names = stash.MiniMaxH3LQPackageLoad.RETURN_NAMES
    types = stash.MiniMaxH3LQPackageLoad.RETURN_TYPES
    listed = stash.MiniMaxH3LQPackageLoad.OUTPUT_IS_LIST
    assert names[:6] == (
        "latent",
        "positive",
        "negative",
        "meta",
        "package_name",
        "selected_count",
    )
    assert names[6] == "package_pipe"
    assert types[6] == stash.PACKAGE_PIPE_TYPE
    assert listed == (True, True, True, True, True, False, True)


def test_partial_ignored_by_allocator(media):
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "clip_a"
        pkg.mkdir()
        (pkg / "clip_a_upscaled_001.mp4.part").write_bytes(b"incomplete")
        nxt = media.next_upscaled_path(pkg)
        assert nxt.name == "clip_a_upscaled_001.mp4"


def test_unique_copy_name(edit):
    used = set()
    assert edit._unique_copy_name("clip.mp4", used) == "clip.mp4"
    assert edit._unique_copy_name("clip.mp4", used) == "clip_02.mp4"
    assert edit._unique_copy_name("other.mp4", used) == "other.mp4"


def test_export_bundle_copies_clips_and_edit(stash, edit):
    with tempfile.TemporaryDirectory() as tmp:
        output, stash_dir = _install_temp_output(stash, Path(tmp))
        edit.output_root = stash.output_root
        edit.stash_root = stash.stash_root
        src_a = stash_dir / "proj" / "clip_a" / "clip_a_upscaled_001.mp4"
        src_b = stash_dir / "proj" / "clip_b" / "clip_b_upscaled_002.mp4"
        src_a.parent.mkdir(parents=True)
        src_b.parent.mkdir(parents=True)
        src_a.write_bytes(b"aaa")
        src_b.write_bytes(b"bbb")
        resolved = [
            {
                "package": "proj/clip_a",
                "name": "clip_a",
                "media": "clip_a_upscaled_001.mp4",
                "preview": src_a,
                "in_frame": 0,
                "out_frame": 24,
                "frame_count": 24,
                "fps": 24.0,
                "note": "",
                "prompt": "",
                "width": 1280,
                "height": 720,
                "has_audio": False,
            },
            {
                "package": "proj/clip_b",
                "name": "clip_b",
                "media": "clip_b_upscaled_002.mp4",
                "preview": src_b,
                "in_frame": 12,
                "out_frame": 48,
                "frame_count": 96,
                "fps": 24.0,
                "note": "",
                "prompt": "",
                "width": 1280,
                "height": 720,
                "has_audio": False,
            },
            {
                "package": "proj/clip_a",
                "name": "clip_a",
                "media": "clip_a_upscaled_001.mp4",
                "preview": src_a,
                "in_frame": 0,
                "out_frame": 10,
                "frame_count": 24,
                "fps": 24.0,
                "note": "",
                "prompt": "",
                "width": 1280,
                "height": 720,
                "has_audio": False,
            },
        ]
        edit.resolve_edit_clips = lambda clips, fps=24.0: resolved
        result = edit.export_bundle([], fps=24.0, name="final_cut")
        folder = Path(result["path"])
        assert folder.is_dir()
        assert (folder / "final_cut.xml").is_file()
        assert (folder / "final_cut.json").is_file()
        assert (folder / "clip_a_upscaled_001.mp4").read_bytes() == b"aaa"
        assert (folder / "clip_b_upscaled_002.mp4").read_bytes() == b"bbb"
        xml = (folder / "final_cut.xml").read_text(encoding="utf-8")
        assert "clip_a_upscaled_001.mp4" in xml
        assert "clip_b_upscaled_002.mp4" in xml
        data = json.loads((folder / "final_cut.json").read_text(encoding="utf-8"))
        assert len(data["clips"]) == 3
        assert data["clips"][0]["media"] == "clip_a_upscaled_001.mp4"
        assert data["clips"][2]["media"] == "clip_a_upscaled_001.mp4"
        _ = output


def main():
    sys.path.insert(0, str(ROOT.parent.parent))
    stash, media, edit = _load_package()
    test_canonical_paths_and_pipe(stash)
    test_parse_edit_json_preserves_media(stash)
    test_next_upscaled_and_variants(media)
    test_media_traversal_rejected(stash, media)
    test_partial_ignored_by_allocator(media)
    test_load_outputs_keep_indices(stash)
    test_unique_copy_name(edit)
    test_export_bundle_copies_clips_and_edit(stash, edit)
    print("PACKAGE_COLLECT_OK")


if __name__ == "__main__":
    main()
