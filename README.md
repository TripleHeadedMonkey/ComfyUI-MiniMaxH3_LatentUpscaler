# MiniMaxH3_LatentUpscaler

ComfyUI custom node for **fixed 2× latent spatial upscaling** between MiniMax H3 samplers.

The default path runs a trained clean-latent 2× upscaler, then performs the NestedTensor-aware CONST re-noise required to continue sampling at high resolution. Stock `LatentUpscaleBy` / `AddNoise` break on MiniMax’s joint AV latents (`video [B,24,T,H/16,W/16]` + `audio [B,32,2,T_audio]`).

The learned 2× architecture and training approach were inspired by [ComfyUI-H3-Latent-Upscaler-Mamad8](https://github.com/mamad8c/ComfyUI-H3-Latent-Upscaler-Mamad8).

## Nodes

**MiniMax H3 Latent Upscale Combined** (`latent/minimax_h3`)

**Required inputs:** `LATENT`, `method`, `learned_model`, `MODEL`, `NOISE`, `SIGMAS`, `audio_denoise`, `noise_resample`  
**Optional inputs:** `match_stats`, `positive`, `negative`  
**Outputs:** `latent`, `positive`, `negative`

**MiniMax H3 Latent Blur** (`latent/minimax_h3`) — standalone norm-preserving spatial smooth on the video NestedTensor stream. Audio unchanged. Use after Combined only if you still see a hard 2×2 grid (~0.4–0.6).

**MiniMax H3 Latent Sharpen** (`latent/minimax_h3`) — standalone norm-preserving unsharp on the video NestedTensor stream. Audio unchanged.

**MiniMax H3 Latent Contrast** (`latent/minimax_h3`) — per-channel contrast around each channel mean on the video stream. `contrast=1` is a no-op; defaults to restoring each token’s 24-D magnitude (`preserve_norm`).

**MiniMax H3 Save / Load Package** (`latent/minimax_h3`) — save Sampler-1 NestedTensor + conditioning under `output/h3_lq_stash/` for deferred Combined + Sampler-2. See [Day / night packages](#day--night-packages-deferred-hq).

**MiniMax H3 Upscale Collect** (`latent/minimax_h3`) — output node. Wire `package_pipe` from Load plus decoded `IMAGE` (optional `AUDIO`). Writes `h3_lq_stash/<project>/<package>/<package>_upscaled_001.mp4` (then `_002`, …) into the source package folder and opens an HQ timeline to pick iterations and export MP4 / FCP7 XML.

Does:

1. Hard-lock spatial upscale to exactly **2×**.
2. By default (`method` = `learned model`), run the trained clean-latent 2× network on the fully denoised H3 video stream. It learned a correction over bilinear latent interpolation against pixel-upscaled/re-encoded teacher latents, with decoder-aware, SSIM, spatial and temporal losses.
3. Preserve the audio stream unchanged. The learned model only receives the `Bx24xTxHxW` video latent.
4. Bypass `match_stats` on the learned path: the network already predicts the clean output distribution and post-hoc stretching would move it away from that learned manifold.
5. Keep nearest/bilinear/bicubic/area/bislerp as fallback methods. `match_stats` applies only to these interpolation fallbacks.
6. Re-noise **video** at `sigmas[0]` (`noise_scaling` + `inverse_noise_scaling` on the full NestedTensor)
7. **`audio_denoise`**: `0` locks pass-1 audio (zero injected noise **and** audio `noise_mask` 0 so sampler 2 cannot rewrite it); `1` fully remixes it. Do not use values in between.
8. If CONDITIONING is connected: spatially upscale `minimax_refs` / `minimax_keyframes` visual latents and sync `latent_h` / `latent_w`.
9. Park LATENT on CPU + `soft_empty_cache` (no model unload).

Blur / sharpen / contrast are **not** inside Combined — wire the standalone nodes after Combined (before Sampler 2) if needed.

### Learned-upscaler installation

Architecture code is loaded from the sibling reference pack (install path):

```text
ComfyUI/custom_nodes/ComfyUI-H3-Latent-Upscaler-Mamad8
```

The default
[`h3_clean_latent_upscaler_film_epoch200.safetensors`](https://huggingface.co/Tridae/H3LatentUpscaler/blob/main/h3_clean_latent_upscaler_film_epoch200.safetensors)
checkpoint downloads automatically on the first `learned model` run if it is
not already installed. It is SHA-256 verified and saved under:

```text
ComfyUI/models/h3_latent_upscalers/
```

Combined loads the file selected in `learned_model` itself. The small upscaler is wrapped in Comfy's **legacy** `ModelPatcher` (not DynamicVRAM) so its Conv3d weights actually load — under Aimdo the residual can collapse and the 2× looks like bilinear. Other compatible checkpoints can still be placed in that folder and selected manually. The patcher is cached, so repeated runs do not reload it. See the original repo above for architecture details and checkpoint sources.
### Why latent `blur` causes strings/hair (and why it is now norm-preserving, default 0)

A latent token is a 24-dim **vector**, not a pixel. A plain Gaussian blur averages adjacent tokens componentwise, so wherever neighbours point in different directions — i.e. at every content edge — the directional components cancel and the result is a vector far shorter than any code the VAE ever emits. Measured on a 24-channel field at 2×:

| smoothing | mean token norm | tokens below 70% of mean | block-edge/interior contrast |
|---|---|---|---|
| none (`nearest-exact`) | 4.86 | 0% | ~1e9 (hard 2×2 steps) |
| plain blur 0.5 | 3.93 | 19% | 7.6 |
| plain blur 1.0 | 2.46 | 99% | 1.5 |
| norm-preserving 0.5 | 4.86 | 0% | 7.8 |

Those short tokens are strongly out of distribution, so pass 2 invents content at them, and because the collapse tracks content edges the invented content reads as faint strings / stray hairs following edges. `match_stats` cannot fix this — it is one global scalar per channel and cannot restore *per-token* norms that collapsed locally.

`blur` (standalone **Latent Blur** node) now smooths only each token's **direction** and restores its original length, which keeps essentially all the anti-blocking benefit (7.8 vs 7.6) at zero norm error. Same principle as Comfy's `bislerp`, which slerps direction and interpolates magnitude separately instead of mixing componentwise. Leave it off unless you actually see a hard 2×2 grid; raise to ~0.4–0.6 only then. The smoothed *direction* is a linear blend of neighbouring latent directions and is not guaranteed on-manifold either.

### `method`: nearest, area and nearest-exact are the same thing here

When **upscaling**, torch's `area` mode is adaptive average pooling with an output larger than the input, so every output cell maps to exactly one input cell and it degenerates to pixel replication — bit-for-bit identical to `nearest-exact`. Swapping between `nearest`, `nearest-exact` and `area` at `scale_by=2` is a no-op and cannot change your result. Only `bilinear`, `bicubic` and `bislerp` are genuinely different. Of those, `bislerp` is the latent-aware one (norm-preserving); `bilinear`/`bicubic` mix latent vectors componentwise and shorten them at edges, i.e. the same failure mode as plain blur.

### Why `match_stats` exists (and why it's a 0-1 dial, not a checkbox)

`nearest-exact` upscale duplicates each latent pixel exactly, so it does not change a channel's global mean/std. The Gaussian `blur`, and any of the non-nearest `method`s (bilinear/bicubic/area/bislerp), are low-pass — they quietly *reduce* per-channel variance. CONST/flow mixing (`x_σ = σ·noise + (1-σ)·x0`) is calibrated on the energy the DiT expects from `x0` at a given σ. A lower-energy `x0` mixed at the same `sigmas[0]` looks like a noisier sample than the number says, so pass 2 denoises harder / hallucinates more detail than the identical settings produced on a native-resolution (unscaled) sample. `match_stats` rescales the upscaled+blurred video back toward pass-1's per-channel mean/std so `sigmas[0]` keeps closer to the same meaning regardless of `method`/`blur`.

The catch: at this resolution there is no clean frequency split between "the grid" and "real fine detail" — nearest-exact's block edges and a mild blur's softened ramps live in the same band as genuine per-pixel texture. `match_stats` is a global per-channel *contrast stretch*, not a re-sharpen; it can't literally rebuild the sharp edges blur removed, but stretching contrast around the channel mean raises the softened edge ramps back up right along with real texture. At full strength (`1.0`) this puts the residual block-edge energy back to roughly its pre-blur amplitude — blur is barely still doing its job — and pass 2 samples that residual as faint "string" / hair artefacts. Its default is now `0`, and it is always bypassed by `learned model`.

## Wiring

1. SamplerCustomAdvanced #1 → high/majority σ at low res
2. Take **`denoised_output`**
3. **MiniMax H3 Latent Upscale Combined**
   - latent = denoised_output
   - positive/negative = same cond used for pass 1
   - RandomNoise + **low** sigmas + model
   - `method` = `learned model` (default; scale is fixed internally at 2×)
   - `learned_model` = your 2× checkpoint under `models/h3_latent_upscalers/`
   - `match_stats` = `0` (ignored by the learned path)
   - Optionally wire **Latent Sharpen** after Combined at `0.15–0.30` to gently counter learned-upscaler softness; leave it out (or amount `0`) to disable
   - Optionally wire **Latent Blur** only if a hard 2×2 grid remains
   - `audio_denoise` = `0` unless you explicitly want pass 2 to rewrite audio
4. Build a **new Guider** from Combined’s returned `positive` / `negative`
5. SamplerCustomAdvanced #2 — DisableNoise + the **same** low sigmas + Combined latent + new Guider
6. Use a non-ancestral sampler on pass 2 (`euler`, `res_multistep`, …)

### Day / night packages (deferred HQ)

Use **MiniMax H3 Save Package** / **MiniMax H3 Load Package** to iterate creatively at LQ and upscale later. Load opens a nearly-fullscreen **Browse Saved Packages** gallery (folder tree, staged multi-select → Selected list, hover-video cards, details preview) so overnight HQ can queue across many shots. A **Timeline** tab lets you mock up cut order from the Selected list before upscale.

**Day (iterate LQ):**

1. Run Sampler 1 as usual.
2. Wire `denoised_output` + the same `positive` / `negative` into **Save Package**.
3. Set optional `project` (e.g. `my_film`) so packages nest under `output/h3_lq_stash/my_film/…`.
4. For gallery / timeline previews: wire LQ **VAE decode** `IMAGE` into Save’s optional `images`, and optional `AUDIO` into `audio` (muxed into `preview.mp4`). Without images, the card shows a placeholder and the clip cannot be used on the timeline.
5. Optionally continue the day graph from the pass-through outputs for on-screen review.
6. Packages land in `ComfyUI/output/h3_lq_stash/[project/]<name>/` as:
   - `latent.safetensors` (`video` + `audio`)
   - `conditioning.pt` (positive / negative, including `minimax_refs` / keyframes)
   - `meta.json` (shapes, note, optional seed/prompt, preview flags)
   - `preview.mp4` + `thumb.jpg` when `images` was connected

**Night (unattended HQ):**

1. **Load Package** → **Browse Saved Packages**:
   - Click / Shift-click cards to **stage** them (amber outline).
   - Press **Add to list** to append staged packages to the **Selected list** (column 3). Reorder with ↑/↓ or drag; remove with ×; **Clear all** asks for confirmation.
   - Details pane shows a looping preview video plus meta for the focused package.
   - **Apply selection** writes the Selected list order into the node.
2. Choose `load_source`, then set `index` to `0` with `index_mode` = `increment` (the node advances the index itself after each queued job). Queue **N** jobs for the count shown on the node; use `fixed` to re-run one entry.
   - `selection list` — current behavior: load packages in the Browse Selected-list order.
   - `timeline order` — load only clips placed on the timeline, preserving timeline order and duplicate uses of a package, but ignore IN/OUT trims.
   - `timeline order and crops` — use timeline order and crop each package's latent, audio latent, noise mask, and positive/negative conditioning to that clip's IN/OUT range.
   - H3 temporal latent tokens cover a repeating 1/4/4/4/4 frames. A cut start therefore snaps backward to a five-token cycle boundary (so the cropped latent keeps the correct temporal phase), while its end snaps forward to a token boundary. No requested frame is lost. Returned `meta.timeline_crop` records requested, clamped, and effective frame/token ranges.
3. Choose `chunk_mode`:
   - `direct load` — one latent per queued entry. Outputs are still ComfyUI **lists** of length 1 (required so scene mode can fan out).
   - `scene aware chunks (experimental)` — run PySceneDetect (`scenedetect` ContentDetector) on that entry’s `preview.mp4`, crop latent+conditioning per scene, and emit a list so Combined / Sampler run once per scene in a **single** queue job. `index` still selects the package/timeline entry; scenes are an inner list dimension.
   - **Requires `preview.mp4`.** Missing preview is a hard error — re-save with `images` wired.
   - Long continuous takes with no hard cut are soft-split by `max_scene_seconds` (default 5s) into equal pieces with **no** overlap lock. Tune `scene_threshold` / `min_scene_seconds` if cuts are too eager or too sparse.
   - Caveats (also shown on the node when scene mode is on): the text prompt stays global; only off-window keyframes/guides are dropped; cuts snap to the H3 token grid (see crop note above). Each scene still writes its own `*_upscaled_NNN.mp4` via Collect.
4. Wire `latent` / `positive` / `negative` into **Latent Upscale Combined**; `package_name` is now output-relative (`h3_lq_stash/project/clip`). `selected_count` is still the entry count for `index` wrapping. `package_pipe` carries the same origin for **Upscale Collect**.
5. Rebuild the Guider from the returned conditioning (do not reuse a day Guider).
6. Sampler 2 (DisableNoise) → VAE decode → **Upscale Collect** (`package_pipe` + `IMAGE`, optional `AUDIO`). Each run writes the next `*_upscaled_NNN.mp4` beside the source package.
7. On Collect, the node shows a playable frame of the HQ MP4 just written. **Open Timeline** edits the **same** Load Package timeline (clips, order, IN/OUT) — not a separate edit. Blue clips are still LQ `preview.mp4`; gold clips are an upscaled `_NNN`. Right-click a strip clip to swap **LQ preview** / `_upscaled_001` / `_002` / …. Export MP4, FCP7 XML, or an **Export bundle** folder (copied chosen files + XML + edit JSON).

**Night output layout:**

```text
ComfyUI/output/h3_lq_stash/testStash/android_conservatory_00005/
  latent.safetensors
  conditioning.pt
  meta.json
  preview.mp4
  thumb.jpg
  android_conservatory_00005_upscaled_001.mp4
  android_conservatory_00005_upscaled_002.mp4
```

**Timeline mockup (optional):**

1. With packages on the Selected list, open the **Timeline** tab.
2. Drag clips from the bin onto the timeline strip (or double-click to append). Click or drag anywhere on the strip to move the playhead; only the clip edge handles trim instead of scrubbing. **Space** plays/pauses from the playhead and the edit loops at the end. `Delete` removes the selected clip, `Alt`+drag reorders one.
   - Clip bodies show their IN and OUT frames as thumbnails (decoded from the LQ preview), collapsing to just IN when the clip is too narrow for both.
   - Transport and zoom controls (`Play`, `Fit`, `+`, `−`) sit in a row between the monitor and the strip, with the timecode readout on the right. Cuts are double-buffered — the outgoing frame is held while the next clip decodes, so the monitor never flashes black at a cut.
   - **Fit** scales the whole edit to the strip; **+** / **−** or `Ctrl`+wheel zoom in around the playhead. While zoomed, the strip follows the playhead only when it nears an edge, at a capped rate so dragging near the edge scans steadily instead of running away. Plain wheel scrolls the zoomed strip.
3. Export buttons (writes under `output/h3_lq_stash/_edits/`):
   - **Export MP4** — combined H.264 of the trimmed segments (feel the pacing).
   - **Export FCP7 XML** — xmeml v5 cut list with absolute paths to each clip file. Imports into **DaVinci Resolve** and **Premiere Pro**. CapCut has no reliable XML import — use the MP4 there.
   - **Export bundle** — copies the chosen clip files into a new folder together with `edit.xml` and `edit.json`. XML `pathurl`s point at the copies in that folder.
4. The timeline is stored in a hidden `edit_json` widget on the Load node (workflow-persistent) and drives the two timeline `load_source` modes. Collect opens that same edit: LQ clips stay blue, HQ replacements gold; right-click a clip to pick which file plays and exports.

Pass-2 knobs (learned model, sigmas, LoRA, denoise, sharpen) stay free at load time — only the clean LQ latent and conditioning are saved.

### Pass-2 artefacts (what actually causes them)

Anything that mixes 24-channel H3 latent vectors componentwise (plain blur, bilinear, bicubic) shortens tokens at content edges and pushes them off-manifold. Pass 2 then invents content there (strings, stray hairs, sparkle, mushy faces). Fixes that help, in order:

- Prefer no **Latent Blur**. If you need it for a hard grid, it is now norm-preserving, but off is still safest.
- The node is hard-locked to integer **2×**; non-integer latent resampling is no longer exposed.
- **`nearest-exact`** (or `area`/`nearest` — identical), not bilinear/bicubic. Try **`bislerp`** if the 2×2 grid is the dominant problem: it is the one smooth method that preserves token magnitude.
- Start pass 2 around **σ 0.25–0.45**, not a full schedule from ~1. Too little σ cannot hide the upsample; too much σ rewrites the clip and looks like artefacts.
- Do **not** scale the noise tensor to “denoise less”. CONST still samples at `sigmas[0]`, so weaker noise + the same σ is over-denoise.
- Tune `match_stats` (0-1). Note that at `blur=0` with `nearest-exact` it is a **no-op** (exact pixel duplication already preserves per-channel statistics), so it cannot be the cause of artefacts in that configuration. It only starts mattering once you use a smoothing/interpolating method.
- Still seeing lines with an interpolation fallback? Use `learned model`: it was trained against pixel-upscaled/re-encoded teacher latents specifically to avoid the piecewise-constant 2×2 block structure.
- Rebuild the Guider from Combined’s CONDITIONING (ref2va identity warp otherwise).

If audio garbles, set `audio_denoise=0` and finish more of the schedule in pass 1 (audio settles late).

### Why conditioning must scale (ref2va)

`minimax_refs` packs each ref with its own `latent` + `latent_h`/`latent_w`. After the target canvas grows 2×, refs sized for the 0.5MP “match” canvas sit at the wrong relative scale and RoPE row layout vs the new target — classic identity warp. Combined doubles ref visual latents and metadata together.

### VRAM between samplers

Avoid Easy-Use Empty Cache / force-unload between passes (especially with `--disable-dynamic-vram` + quantized MiniMax + SageAttention).

## Fewer pass-2 steps without under-denoising the grid

**MiniMax H3 Pass-2 Staggered Scheduler** (`latent/minimax_h3`, `SIGMAS` output)

If you're running e.g. `BasicScheduler` (`normal`, 8 steps, `denoise=0.45`) and want fewer
steps: **don't** just swap `BasicScheduler`'s `scheduler` dropdown to `karras`/`exponential`
at the same `denoise`. `karras`/`exponential` interpolate directly in raw sigma space, while
`normal` interpolates in the model's timestep space first — so BasicScheduler's denoise-trim
(`total_steps = steps/denoise`, keep the last `steps+1` sigmas) lands at a **much lower**
`sigmas[0]` under karras than under normal for the "same" numbers, silently under-denoising
instead of removing the grid. That's almost certainly why `SplitSigmas` felt too weak.

This node decouples the two things you actually control:

- `base_scheduler` / `reference_steps` / `denoise` — reproduces your **proven** recipe (e.g.
  `normal` / `8` / `0.45`) purely to find its `sigmas[0]`. Nothing else about that recipe is used.
- `steps` / `rho` / `shape` — a **separate**, much shorter front-loaded curve from that same
  `sigmas[0]` down to ~0: one aggressive early step to overwrite the nearest-exact upscale grid,
  then progressively finer steps for texture.

Start with `shape=karras`, `rho=7`, `steps=4` (down from 8) at your existing `denoise`. Raise
`rho` (or switch to `exponential`) for a bigger first step / less texture budget; lower it
toward evenly-spaced if the first step looks too aggressive. Wire the `SIGMAS` output straight
into the same `SamplerCustomAdvanced` you already use for pass 2 — no other graph changes.

## Chunked pass-2 (historical, unregistered)

**MiniMax H3 Latent Upscale Chunked (Pass 2)** (`latent/minimax_h3`)

This experimental node is intentionally not registered because independent temporal
chunks could not maintain acceptable cross-chunk consistency. The source remains only
for reference; it cannot be created from the ComfyUI node menu.

Its original design bundled the stage-2 graph so a long HD clip never sat in VRAM as a single sample:

`slice T` → window guides to that T → Combined upscale + CONST re-noise → **lock previous HD overlap** (`noise_mask` 0) → DisableNoise sample → VAE decode → stitch

**Required inputs:** pass-1 `LATENT`, pass-2 `MODEL` (LoRA already applied), `positive`, video `VAE`, `audio_vae`  
**Optional:** `negative` (ignored by BasicGuider); `lock_overlap` (default on); `window_conditioning` (default on)  
**Outputs:** `IMAGE`, `AUDIO` (stitched; no full HD latent)

**Chunk controls:**
- `chunk_latent_t` (default 7) — video latent tokens per chunk. Start around 7 for 1280×1920; lower if OOM.
- `overlap_latent_t` (default 2) — protected prefix. Snapped to H3 runs `2/7/12…` (5/22/39 frames) when `lock_overlap` is on. Prefer **7** (22 frames) if VRAM allows; **2** is the smallest valid lock.
- `lock_overlap` — copy the previous chunk’s **sampled HD latent tail** into the new window and freeze it (same idea as `H3 Generated AV Masked Context`). CONST re-noise skips those tokens.
- `window_conditioning` — remap in-window `minimax_keyframes` / AddGuide to chunk-local frame 0; drop off-window guides and standalone `minimax_refs` audio so a line spoken in one shot is not re-conditioned into every slice. Image/video identity refs stay. The **text prompt is still global** (H3 has no per-token text timeline); for quoted dialogue in a long prompt, prefer per-shot prompts on pass 1.
- Pass-2 sample widgets match a standalone BasicScheduler: `sampler_name` (default `lcm`), `scheduler` (`normal`), `steps` (8), `denoise` (0.4), `noise_seed`.

If `ComfyUI-H3-Motion-Context-MultiRef` is installed, the node enables that pack’s H3 AV-mask timestep pinning. Otherwise Comfy’s inpaint blend still holds the frozen prefix.

Approx duration for a chunk: frames ≈ `5 + ((T-2)/5)*17` at 24 fps.

Do **not** force-unload models between chunks; the node only soft-clears the allocator after each decode.

## Faster H3 video VAE decode

**MiniMax H3 VAE Decode (fast)** — drop-in replacement for stock `VAEDecode` on the H3 *video* VAE.

Stock decode always uses **256px spatial tiles** and writes each temporal chunk to **CPU**. That is why CUDA sits around 50% with low VRAM. `VAEDecodeTiled` does not change this (H3 ignores those kwargs).

This node snapshots H3's internal flags, decodes through **H3's own tiled path only** (no generic 3D-tiler fallback), then restores flags:

- `tiling` — on = spatial tiles (same algorithm as stock); off = full frame per temporal chunk
- `tile_size` — pixels; default **256** to match stock quality. Larger is faster; overlap is raised to keep the 64/256 ratio (otherwise 512/64 seams look like a block grid)
- `tile_overlap` — minimum overlap; actual overlap is `max(this, tile_size/4)`
- `output_device` — `cpu` (stock) or `gpu` (skip copies). Does not change quality.

Temporal ~17-frame chunking is part of the VAE and is not toggled. Connect the **video** VAE, not the audio VAE.

## Install

`ComfyUI/custom_nodes/ComfyUI-MiniMaxH3_LatentUpscaler/` — restart or reload custom nodes.
