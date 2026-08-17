/** Timeline tab for saved packages — bin + program monitor + compact scrub strip. */

const H3_FPS = 24;
const MIN_PX_PER_FRAME = 0.4;
const FIT_MAX_PX_PER_FRAME = 8;
const ZOOM_MAX_PX_PER_FRAME = 40;
const THUMB_W = 52;
const THUMB_H = 44;
const EDGE_MARGIN_PX = 90;
const SCRUB_SCROLL_STEP_PX = 8;
const PLAY_SCROLL_STEP_PX = 60;

/** dataURL cache shared across modal sessions: `${package}|${frame}` */
const frameThumbCache = new Map();
const thumbQueue = [];
let thumbBusy = false;

export function parseEdit(raw) {
    const text = String(raw || "").trim();
    if (!text) {
        return { version: 1, fps: H3_FPS, root: "h3_lq_stash", name: "edit", clips: [] };
    }
    try {
        const data = JSON.parse(text);
        if (!data || typeof data !== "object") {
            return { version: 1, fps: H3_FPS, root: "h3_lq_stash", name: "edit", clips: [] };
        }
        return {
            version: data.version || 1,
            fps: Number(data.fps) || H3_FPS,
            root: data.root || "h3_lq_stash",
            name: data.name || "edit",
            clips: Array.isArray(data.clips)
                ? data.clips
                      .map((c) => {
                          const clip = {
                              package: String(c.package || "").replace(/\\/g, "/"),
                              in_frame: Math.max(0, Number(c.in_frame) || 0),
                              out_frame: Math.max(1, Number(c.out_frame) || 1),
                          };
                          const media = String(c.media || "").replace(/\\/g, "/");
                          if (media && !media.includes("/") && !media.includes("..")) {
                              clip.media = media;
                          }
                          return clip;
                      })
                      .filter((c) => c.package)
                : [],
        };
    } catch {
        return { version: 1, fps: H3_FPS, root: "h3_lq_stash", name: "edit", clips: [] };
    }
}

function createEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) {
        el.className = className;
    }
    if (text != null) {
        el.textContent = text;
    }
    return el;
}

function formatTimecode(frames, fps) {
    const rate = Math.max(1, Math.round(fps));
    const total = Math.max(0, Math.floor(frames));
    const s = Math.floor(total / rate);
    const f = total % rate;
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}:${String(f).padStart(2, "0")}`;
}

/** Decoder videos are pooled per source so repeated frame grabs reuse one download. */
const decoders = new Map();
const MAX_DECODERS = 4;

function getDecoder(url) {
    const existing = decoders.get(url);
    if (existing) {
        return existing;
    }
    if (decoders.size >= MAX_DECODERS) {
        const [oldestUrl, oldest] = decoders.entries().next().value;
        oldest.video.removeAttribute("src");
        oldest.video.load?.();
        decoders.delete(oldestUrl);
    }
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.playsInline = true;
    const ready = new Promise((resolve, reject) => {
        video.onloadeddata = () => resolve();
        video.onerror = () => reject(new Error("preview decode failed"));
    });
    video.src = url;
    const entry = { video, ready };
    decoders.set(url, entry);
    return entry;
}

async function decodeFrameThumb(url, timeSec) {
    const { video, ready } = getDecoder(url);
    await ready;
    const target = Math.max(0, timeSec);
    // Setting currentTime to its present value fires no seeked event, so skip the wait.
    if (Math.abs(video.currentTime - target) > 0.001) {
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                video.onseeked = null;
                reject(new Error("seek timed out"));
            }, 6000);
            video.onseeked = () => {
                clearTimeout(timer);
                video.onseeked = null;
                resolve();
            };
            try {
                video.currentTime = target;
            } catch (err) {
                clearTimeout(timer);
                video.onseeked = null;
                reject(err);
            }
        });
    }
    const ratio = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 16 / 9;
    const canvas = document.createElement("canvas");
    canvas.height = THUMB_H;
    canvas.width = Math.max(1, Math.round(THUMB_H * ratio));
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.72);
}

let thumbPauseCount = 0;

function pauseThumbDecoders() {
    thumbPauseCount += 1;
    for (const entry of decoders.values()) {
        try {
            entry.video.pause?.();
        } catch {
            /* ignore */
        }
    }
}

function resumeThumbDecoders() {
    thumbPauseCount = Math.max(0, thumbPauseCount - 1);
    if (thumbPauseCount === 0) {
        pumpThumbQueue();
    }
}

function pumpThumbQueue() {
    if (thumbBusy || thumbPauseCount > 0) {
        return;
    }
    const job = thumbQueue.shift();
    if (!job) {
        return;
    }
    thumbBusy = true;
    decodeFrameThumb(job.url, job.timeSec)
        .then((data) => {
            frameThumbCache.set(job.key, data);
            job.onReady?.(data);
        })
        .catch(() => {
            frameThumbCache.set(job.key, "");
        })
        .finally(() => {
            thumbBusy = false;
            pumpThumbQueue();
        });
}

function queueFrameThumb(key, url, timeSec, onReady) {
    if (frameThumbCache.has(key) || thumbQueue.some((job) => job.key === key)) {
        return;
    }
    thumbQueue.push({ key, url, timeSec, onReady });
    pumpThumbQueue();
}

/**
 * @param {object} opts
 * @param {(pkg: string) => object|null} opts.getPackageInfo
 * @param {() => string[]} opts.getListed
 * @param {(edit: object) => void} opts.onEditChange
 * @param {(card: HTMLElement, pkg: string, ev: MouseEvent) => void} opts.showHover
 * @param {() => void} opts.destroyHover
 * @param {(path: string, body: object) => Promise<object>} opts.postJson
 * @param {(item: object, ev: MouseEvent) => Promise<string|null|undefined>} [opts.pickClipVersion]
 * @param {string} [opts.binLabel]
 * @param {string} [opts.binEmptyText]
 * @param {boolean} [opts.hqMode]
 */
export function createTimelineTab(opts) {
    const {
        getPackageInfo,
        getListed,
        onEditChange,
        showHover,
        destroyHover,
        postJson,
        initialEdit,
        pickClipVersion,
        binLabel,
        binEmptyText,
        hqMode = false,
    } = opts;

    // H3 packages are always 24fps, so the sequence rate is fixed.
    const fps = H3_FPS;
    let editName = initialEdit?.name || "edit";
    /** @type {{package: string, in_frame: number, out_frame: number, id: string, media?: string}[]} */
    let items = (initialEdit?.clips || []).map((c, i) => ({
        id: `t${i}_${c.package}`,
        package: c.package,
        in_frame: c.in_frame,
        out_frame: c.out_frame,
        media: c.media,
    }));
    let selectedId = items[0]?.id || null;
    let playheadFrame = 0;
    let playing = false;
    let loadedIndex = -1;
    let pxPerFrame = 4;
    let zoomMode = false;
    let rafId = 0;
    let scrubRaf = 0;
    let scrubbing = false;
    let trimming = false;
    let lastPointerX = 0;
    let lastScrubFrame = -1;
    let lastSeekAt = 0;
    let exporting = false;

    const root = createEl("div", "h3lq-timeline");

    const toolbar = createEl("div", "h3lq-timeline__toolbar");
    const nameInput = document.createElement("input");
    nameInput.className = "h3lq-path";
    nameInput.type = "text";
    nameInput.placeholder = "Edit name";
    nameInput.value = editName;
    nameInput.style.maxWidth = "180px";
    const playBtn = createEl("button", "h3lq-btn", "Play");
    playBtn.type = "button";
    playBtn.title = "Play / pause from the playhead (Space)";
    const fitBtn = createEl("button", "h3lq-btn h3lq-btn--tiny", "Fit");
    fitBtn.type = "button";
    fitBtn.title = "Fit the whole edit to the strip";
    const zoomOutBtn = createEl("button", "h3lq-btn h3lq-btn--tiny", "−");
    zoomOutBtn.type = "button";
    zoomOutBtn.title = "Zoom out";
    const zoomInBtn = createEl("button", "h3lq-btn h3lq-btn--tiny", "+");
    zoomInBtn.type = "button";
    zoomInBtn.title = "Zoom in (Ctrl+wheel over the strip)";
    const zoomLabel = createEl("div", "h3lq-status", "");
    const exportMp4Btn = createEl("button", "h3lq-btn h3lq-btn--primary", "Export MP4");
    exportMp4Btn.type = "button";
    const exportXmlBtn = createEl("button", "h3lq-btn", "Export FCP7 XML");
    exportXmlBtn.type = "button";
    const exportBundleBtn = createEl("button", "h3lq-btn", "Export bundle");
    exportBundleBtn.type = "button";
    exportBundleBtn.title = "Copy clip files into a new folder with FCP7 XML + edit JSON";
    const status = createEl("div", "h3lq-status", "");
    toolbar.append(nameInput, exportMp4Btn, exportXmlBtn, exportBundleBtn, status);

    const body = createEl("div", "h3lq-timeline__body");

    const binPane = createEl("div", "h3lq-pane h3lq-timeline__bin");
    binPane.append(createEl("div", "h3lq-pane__label", binLabel || "Clip bin (from Selected list)"));
    const binList = createEl("div", "h3lq-timeline__bin-list");
    binPane.append(binList);

    const stagePane = createEl("div", "h3lq-pane h3lq-timeline__stage");
    const monitorWrap = createEl("div", "h3lq-timeline__monitor-wrap");

    function makeMonitor() {
        const el = document.createElement("video");
        el.className = "h3lq-timeline__monitor";
        el.controls = false;
        el.playsInline = true;
        el.preload = "auto";
        el.muted = true;
        el.autoplay = false;
        return el;
    }

    // Two stacked players: the outgoing clip stays visible until the next one has a
    // decoded frame, so cuts do not flash black.
    const videoA = makeMonitor();
    const videoB = makeMonitor();
    videoA.classList.add("is-active");
    monitorWrap.append(videoA, videoB);
    let activeVideo = videoA;
    let spareVideo = videoB;
    let spareIndex = -1;
    let spareReady = false;
    let swapPending = false;
    let pendingAutoplay = false;
    let advanceLock = false;
    let loadToken = 0;
    let thumbsHeld = false;

    const controls = createEl("div", "h3lq-timeline__controls");
    const timecodeLabel = createEl("div", "h3lq-timeline__readout", "");
    controls.append(playBtn, fitBtn, zoomOutBtn, zoomInBtn, zoomLabel, timecodeLabel);

    const stripWrap = createEl("div", "h3lq-timeline__strip-wrap");
    const ruler = createEl("div", "h3lq-timeline__ruler");
    const strip = createEl("div", "h3lq-timeline__strip");
    const playhead = createEl("div", "h3lq-timeline__playhead");
    strip.append(playhead);
    stripWrap.append(ruler, strip);
    if (hqMode) {
        const legend = createEl("div", "h3lq-timeline__legend");
        legend.append(
            createEl("span", "h3lq-timeline__swatch is-lq", "LQ"),
            createEl("span", "h3lq-timeline__swatch is-hq", "HQ"),
            createEl("span", "h3lq-timeline__legend-hint", "Right-click a clip to swap versions")
        );
        stagePane.append(monitorWrap, controls, stripWrap, legend);
    } else {
        stagePane.append(monitorWrap, controls, stripWrap);
    }

    body.append(binPane, stagePane);
    root.append(toolbar, body);

    // —— geometry helpers ——

    function totalFrames() {
        return items.reduce((sum, it) => sum + Math.max(1, it.out_frame - it.in_frame), 0);
    }

    function offsetOf(index) {
        let acc = 0;
        for (let i = 0; i < index && i < items.length; i++) {
            acc += Math.max(1, items[i].out_frame - items[i].in_frame);
        }
        return acc;
    }

    function locate(frame) {
        let acc = 0;
        for (let i = 0; i < items.length; i++) {
            const dur = Math.max(1, items[i].out_frame - items[i].in_frame);
            if (frame < acc + dur) {
                return { index: i, local: Math.max(0, frame - acc), item: items[i] };
            }
            acc += dur;
        }
        if (!items.length) {
            return null;
        }
        const last = items.length - 1;
        return { index: last, local: 0, item: items[last] };
    }

    function fitZoom() {
        const avail = Math.max(120, stripWrap.clientWidth - 4);
        const total = Math.max(1, totalFrames());
        pxPerFrame = Math.min(FIT_MAX_PX_PER_FRAME, Math.max(MIN_PX_PER_FRAME, avail / total));
    }

    function updateZoomLabel() {
        const secPx = (pxPerFrame * fps).toFixed(0);
        zoomLabel.textContent = zoomMode ? `zoom ${secPx}px/s` : `fit ${secPx}px/s`;
    }

    function setZoom(nextPx) {
        const clamped = Math.min(ZOOM_MAX_PX_PER_FRAME, Math.max(MIN_PX_PER_FRAME, nextPx));
        // Keep the playhead pinned at its current screen position across zoom changes.
        const screenOffset = playheadFrame * pxPerFrame - stripWrap.scrollLeft;
        zoomMode = true;
        pxPerFrame = clamped;
        renderStrip();
        stripWrap.scrollLeft = Math.max(0, playheadFrame * pxPerFrame - screenOffset);
        updatePlayheadPos();
        updateZoomLabel();
    }

    function setFit() {
        zoomMode = false;
        fitZoom();
        renderStrip();
        stripWrap.scrollLeft = 0;
        updatePlayheadPos();
        updateZoomLabel();
    }

    /** Bounded edge-follow scroll; never recentres, so manual scroll is respected. */
    function autoScroll(maxStepPx) {
        if (!zoomMode) {
            return;
        }
        const x = playheadFrame * pxPerFrame;
        const view = stripWrap.scrollLeft;
        const width = stripWrap.clientWidth;
        let target = null;
        if (x < view + EDGE_MARGIN_PX) {
            target = x - EDGE_MARGIN_PX;
        } else if (x > view + width - EDGE_MARGIN_PX) {
            target = x - width + EDGE_MARGIN_PX;
        }
        if (target === null) {
            return;
        }
        const maxScroll = Math.max(0, stripWrap.scrollWidth - width);
        const clampedTarget = Math.max(0, Math.min(maxScroll, target));
        const delta = clampedTarget - view;
        stripWrap.scrollLeft = view + Math.max(-maxStepPx, Math.min(maxStepPx, delta));
    }

    // —— package info ——

    function sourceFrames(pkg) {
        const info = getPackageInfo?.(pkg);
        if (info?.frame_count) {
            return Math.max(1, Number(info.frame_count));
        }
        return 96;
    }

    function isHqClip(item) {
        return Boolean(hqMode && item?.media && /_upscaled_\d{3}\.mp4$/i.test(item.media));
    }

    function variantFor(item) {
        if (!isHqClip(item)) {
            return null;
        }
        const variants = getPackageInfo?.(item.package)?.upscaled || [];
        return variants.find((v) => v.name === item.media) || null;
    }

    function sourceFramesFor(item) {
        const variant = variantFor(item);
        if (variant?.frame_count) {
            return Math.max(1, Number(variant.frame_count));
        }
        return sourceFrames(item.package);
    }

    function previewUrl(pkg) {
        const info = getPackageInfo?.(pkg);
        return info?.preview_url || `/h3_lq_stash/preview?package=${encodeURIComponent(pkg)}`;
    }

    function clipMediaUrl(item) {
        const variant = variantFor(item);
        if (variant?.url) {
            return variant.url;
        }
        if (isHqClip(item)) {
            return `/h3_lq_stash/media?package=${encodeURIComponent(item.package)}&file=${encodeURIComponent(item.media)}`;
        }
        return previewUrl(item.package);
    }

    function thumbUrl(pkg) {
        const info = getPackageInfo?.(pkg);
        return info?.thumb_url || `/h3_lq_stash/thumb?package=${encodeURIComponent(pkg)}`;
    }

    function hasPreview(pkg) {
        const info = getPackageInfo?.(pkg);
        if (info?.has_preview) {
            return true;
        }
        return Boolean(hqMode && (info?.upscaled || []).length);
    }

    function clipLabelText(item) {
        const base = item.package.split("/").pop() || item.package;
        if (!isHqClip(item)) {
            return hqMode ? `${base} · LQ` : base;
        }
        const match = String(item.media).match(/_upscaled_(\d{3})\.mp4$/i);
        return match ? `${base} · HQ ${match[1]}` : base;
    }

    function serialize() {
        return {
            version: 1,
            fps,
            root: "h3_lq_stash",
            name: editName,
            clips: items.map((it) => {
                const clip = {
                    package: it.package,
                    in_frame: it.in_frame,
                    out_frame: it.out_frame,
                };
                if (it.media) {
                    clip.media = it.media;
                }
                return clip;
            }),
        };
    }

    function notify() {
        // Clip bounds changed, so any warmed-up next frame is stale.
        invalidateSpare();
        onEditChange?.(serialize());
        updateReadout();
    }

    function updateReadout() {
        const total = totalFrames();
        timecodeLabel.textContent = `${formatTimecode(playheadFrame, fps)} / ${formatTimecode(
            total,
            fps
        )} · ${items.length} clip(s) · ${total}f @ ${fps}fps`;
    }

    // —— playback ——

    function updatePlayheadPos() {
        playhead.style.left = `${playheadFrame * pxPerFrame}px`;
        updateReadout();
    }

    /** Mid-frame time, so a seek lands on frame N rather than N-1. */
    function frameTime(frame) {
        return (Math.max(0, frame) + 0.5) / fps;
    }

    function currentSourceFrame(video) {
        return Math.max(0, Math.floor(video.currentTime * fps));
    }

    function mediaHref(url) {
        try {
            return new URL(url, location.origin).href;
        } catch {
            return url;
        }
    }

    function sameMedia(video, url) {
        return Boolean(video.src) && video.src === mediaHref(url);
    }

    function nearFrame(video, frame) {
        return Math.abs(video.currentTime - frameTime(frame)) < 0.75 / fps;
    }

    function holdThumbs() {
        if (thumbsHeld) {
            return;
        }
        thumbsHeld = true;
        pauseThumbDecoders();
    }

    function releaseThumbs() {
        if (!thumbsHeld) {
            return;
        }
        thumbsHeld = false;
        resumeThumbDecoders();
    }

    function finishAdvance() {
        advanceLock = false;
        swapPending = false;
    }

    function playMonitor() {
        const target = activeVideo;
        target.muted = true;
        const attempt = target.play();
        if (attempt?.catch) {
            attempt.catch(() => {
                if (!playing || target !== activeVideo) {
                    return;
                }
                target.muted = true;
                target.play().catch(() => {});
            });
        }
    }

    function assignSrc(video, url) {
        if (sameMedia(video, url) && video.readyState >= 1 && !video.error) {
            return Promise.resolve();
        }
        return new Promise((resolve, reject) => {
            const onMeta = () => {
                cleanup();
                resolve();
            };
            const onErr = () => {
                cleanup();
                reject(new Error("load failed"));
            };
            const cleanup = () => {
                video.removeEventListener("loadedmetadata", onMeta);
                video.removeEventListener("error", onErr);
            };
            video.addEventListener("loadedmetadata", onMeta);
            video.addEventListener("error", onErr);
            video.src = url;
            video.load?.();
        });
    }

    /**
     * Seek and wait until a decoded frame is available. `seeked` is not guaranteed
     * when currentTime is already on the target, so also accept canplay / timeout.
     */
    function whenFrameReady(video, frame, token) {
        return new Promise((resolve) => {
            let settled = false;
            let timer = 0;
            const started = performance.now();
            const done = (ok) => {
                if (settled) {
                    return;
                }
                settled = true;
                video.removeEventListener("seeked", onReady);
                video.removeEventListener("canplay", onReady);
                video.removeEventListener("loadeddata", onReady);
                video.removeEventListener("error", onError);
                clearTimeout(timer);
                resolve(ok);
            };
            const readyNow = () =>
                token === loadToken &&
                video.readyState >= 2 &&
                !video.ended &&
                nearFrame(video, frame);
            const onReady = () => {
                if (readyNow()) {
                    done(true);
                }
            };
            const onError = () => done(false);
            const poll = () => {
                if (settled) {
                    return;
                }
                if (token !== loadToken) {
                    done(false);
                    return;
                }
                if (readyNow()) {
                    done(true);
                    return;
                }
                if (performance.now() - started < 1600) {
                    timer = setTimeout(poll, 50);
                    return;
                }
                done(video.readyState >= 2 && !video.ended);
            };
            video.addEventListener("seeked", onReady);
            video.addEventListener("canplay", onReady);
            video.addEventListener("loadeddata", onReady);
            video.addEventListener("error", onError);
            if (readyNow()) {
                done(true);
                return;
            }
            try {
                video.pause?.();
                video.currentTime = frameTime(frame);
            } catch {
                done(video.readyState >= 1 && !video.ended);
                return;
            }
            timer = setTimeout(poll, 50);
        });
    }

    function invalidateSpare() {
        spareIndex = -1;
        spareReady = false;
        spareVideo.onloadedmetadata = null;
        spareVideo.onseeked = null;
        spareVideo.onerror = null;
    }

    function nextIndex(index) {
        return index + 1 < items.length ? index + 1 : 0;
    }

    function prepareSpare(index) {
        const item = items[index];
        if (!item || spareIndex === index || items.length < 2 || swapPending) {
            return;
        }
        const token = loadToken;
        invalidateSpare();
        spareIndex = index;
        spareVideo.pause();
        const url = clipMediaUrl(item);
        assignSrc(spareVideo, url)
            .then(() => whenFrameReady(spareVideo, item.in_frame, token))
            .then((ok) => {
                if (token !== loadToken || spareIndex !== index) {
                    return;
                }
                spareReady = Boolean(ok && spareVideo.readyState >= 1);
            })
            .catch(() => {
                if (token !== loadToken || spareIndex !== index) {
                    return;
                }
                spareReady = false;
            });
    }

    function swapPlayers() {
        const outgoing = activeVideo;
        activeVideo = spareVideo;
        spareVideo = outgoing;
        activeVideo.classList.add("is-active");
        spareVideo.classList.remove("is-active");
        spareVideo.pause();
        invalidateSpare();
    }

    function loadItem(index, localFrame, autoplay) {
        const item = items[index];
        if (!item) {
            finishAdvance();
            return;
        }
        const local = Math.max(0, Math.round(localFrame));
        const frame = item.in_frame + local;
        playheadFrame = offsetOf(index) + local;
        updatePlayheadPos();
        pendingAutoplay = Boolean(autoplay) || pendingAutoplay;
        const url = clipMediaUrl(item);
        holdThumbs();

        if (loadedIndex === index && swapPending) {
            return;
        }

        const canReuse =
            loadedIndex === index &&
            !swapPending &&
            sameMedia(activeVideo, url) &&
            !activeVideo.ended &&
            activeVideo.readyState >= 2;

        if (canReuse) {
            const token = ++loadToken;
            advanceLock = true;
            whenFrameReady(activeVideo, frame, token).then((ok) => {
                if (token !== loadToken) {
                    return;
                }
                finishAdvance();
                if (ok && pendingAutoplay && playing) {
                    playMonitor();
                }
                pendingAutoplay = false;
                if (!playing) {
                    releaseThumbs();
                }
            });
            return;
        }

        if (spareIndex === index && spareReady && sameMedia(spareVideo, url)) {
            const token = ++loadToken;
            advanceLock = true;
            swapPending = true;
            pendingAutoplay = Boolean(autoplay) || pendingAutoplay;
            whenFrameReady(spareVideo, frame, token).then((ok) => {
                if (token !== loadToken) {
                    return;
                }
                if (ok) {
                    swapPlayers();
                    loadedIndex = index;
                }
                finishAdvance();
                if (ok && pendingAutoplay && playing) {
                    playMonitor();
                }
                pendingAutoplay = false;
                if (ok) {
                    prepareSpare(nextIndex(index));
                }
                if (!playing) {
                    releaseThumbs();
                }
            });
            return;
        }

        const token = ++loadToken;
        loadedIndex = index;
        advanceLock = true;
        swapPending = true;
        pendingAutoplay = Boolean(autoplay) || pendingAutoplay;
        activeVideo.pause();
        spareVideo.pause();
        invalidateSpare();

        assignSrc(spareVideo, url)
            .then(() => whenFrameReady(spareVideo, frame, token))
            .then((ok) => {
                if (token !== loadToken) {
                    return;
                }
                if (!ok) {
                    finishAdvance();
                    status.textContent = `Cannot load preview for ${item.package}`;
                    pendingAutoplay = false;
                    releaseThumbs();
                    return;
                }
                swapPlayers();
                finishAdvance();
                if (pendingAutoplay && playing) {
                    playMonitor();
                }
                pendingAutoplay = false;
                prepareSpare(nextIndex(index));
                if (!playing) {
                    releaseThumbs();
                }
            })
            .catch(() => {
                if (token !== loadToken) {
                    return;
                }
                finishAdvance();
                status.textContent = `Cannot load preview for ${item.package}`;
                pendingAutoplay = false;
                releaseThumbs();
            });
    }

    function tick() {
        if (!playing) {
            return;
        }
        const item = items[loadedIndex];
        if (!item) {
            pause();
            return;
        }
        if (swapPending || advanceLock) {
            rafId = requestAnimationFrame(tick);
            return;
        }
        const srcFrame = currentSourceFrame(activeVideo);
        if (srcFrame >= item.out_frame || activeVideo.ended) {
            // Loop back to the head of the edit after the last clip.
            loadItem(nextIndex(loadedIndex), 0, true);
        } else {
            playheadFrame = offsetOf(loadedIndex) + Math.max(0, srcFrame - item.in_frame);
            updatePlayheadPos();
            prepareSpare(nextIndex(loadedIndex));
        }
        autoScroll(PLAY_SCROLL_STEP_PX);
        rafId = requestAnimationFrame(tick);
    }

    function play() {
        if (!items.length) {
            status.textContent = "Timeline is empty";
            return;
        }
        playing = true;
        playBtn.textContent = "Pause";
        holdThumbs();
        const total = totalFrames();
        const from = playheadFrame >= total ? 0 : playheadFrame;
        const hit = locate(from) || { index: 0, local: 0 };
        loadItem(hit.index, hit.local, true);
        cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(tick);
    }

    function pause() {
        playing = false;
        pendingAutoplay = false;
        playBtn.textContent = "Play";
        activeVideo.pause();
        spareVideo.pause();
        cancelAnimationFrame(rafId);
        releaseThumbs();
    }

    function togglePlay() {
        if (playing) {
            pause();
        } else {
            play();
        }
    }

    // —— clip thumbnails ——

    function attachFrameThumb(itemEl, item, frame, side) {
        const key = `${item.package}|${item.media || "preview"}|${frame}`;
        const img = document.createElement("img");
        img.className = `h3lq-timeline__item-thumb is-${side}`;
        img.alt = "";
        img.draggable = false;
        const cached = frameThumbCache.get(key);
        if (cached) {
            img.src = cached;
        } else {
            img.src = thumbUrl(item.package);
            if (!trimming && !frameThumbCache.has(key)) {
                queueFrameThumb(key, clipMediaUrl(item), frameTime(frame), (data) => {
                    if (img.isConnected && data) {
                        img.src = data;
                    }
                });
            }
        }
        itemEl.append(img);
    }

    // —— strip rendering ——

    function renderStrip() {
        strip.querySelectorAll(".h3lq-timeline__item").forEach((el) => el.remove());
        ruler.replaceChildren();
        const total = Math.max(totalFrames(), 1);
        const width = Math.max(total * pxPerFrame, stripWrap.clientWidth - 4);
        strip.style.width = `${width}px`;
        ruler.style.width = `${width}px`;

        // One tick per second, thinned out so labels never collide.
        const secondPx = fps * pxPerFrame;
        const everyNSeconds = Math.max(1, Math.ceil(56 / Math.max(1, secondPx)));
        for (let f = 0; f <= total; f += Math.round(fps) * everyNSeconds) {
            const tick = createEl("div", "h3lq-timeline__tick", formatTimecode(f, fps));
            tick.style.left = `${f * pxPerFrame}px`;
            ruler.append(tick);
        }

        let acc = 0;
        items.forEach((it) => {
            const dur = Math.max(1, it.out_frame - it.in_frame);
            const clipPx = Math.max(6, dur * pxPerFrame);
            const el = createEl(
                "div",
                `h3lq-timeline__item${selectedId === it.id ? " is-selected" : ""}${
                    hqMode ? (isHqClip(it) ? " is-hq" : " is-lq") : ""
                }`
            );
            el.style.left = `${acc * pxPerFrame}px`;
            el.style.width = `${clipPx}px`;
            el.dataset.id = it.id;

            // IN thumb always (when there is room at all); OUT thumb only when both fit.
            if (hasPreview(it.package) && clipPx >= 26) {
                attachFrameThumb(el, it, it.in_frame, "in");
                if (clipPx >= THUMB_W * 2 + 16) {
                    attachFrameThumb(el, it, Math.max(it.in_frame, it.out_frame - 1), "out");
                }
            }

            const label = createEl(
                "div",
                "h3lq-timeline__item-label",
                clipLabelText(it)
            );
            label.title = `${it.package}${it.media ? ` · ${it.media}` : ""}  in=${it.in_frame} out=${it.out_frame} (${dur}f)`;
            const leftHandle = createEl("div", "h3lq-timeline__handle h3lq-timeline__handle--left");
            leftHandle.title = "Trim in";
            const rightHandle = createEl(
                "div",
                "h3lq-timeline__handle h3lq-timeline__handle--right"
            );
            rightHandle.title = "Trim out";
            el.append(label, leftHandle, rightHandle);

            const startTrim = (side, ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                selectedId = it.id;
                trimming = true;
                const startX = ev.clientX;
                const startIn = it.in_frame;
                const startOut = it.out_frame;
                const maxF = sourceFramesFor(it);
                const onMove = (e) => {
                    const df = Math.round((e.clientX - startX) / pxPerFrame);
                    if (side === "left") {
                        it.in_frame = Math.max(0, Math.min(startOut - 1, startIn + df));
                    } else {
                        it.out_frame = Math.max(startIn + 1, Math.min(maxF, startOut + df));
                    }
                    renderStrip();
                    updateReadout();
                };
                const onUp = () => {
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", onUp);
                    trimming = false;
                    refreshStrip();
                    notify();
                };
                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", onUp);
            };
            leftHandle.addEventListener("pointerdown", (ev) => startTrim("left", ev));
            rightHandle.addEventListener("pointerdown", (ev) => startTrim("right", ev));

            // Alt-drag reorders; a plain drag scrubs the playhead instead.
            el.addEventListener("mousedown", (ev) => {
                el.draggable = ev.altKey;
            });
            el.addEventListener("dragstart", (ev) => {
                ev.dataTransfer?.setData("text/h3lq-timeline-id", it.id);
                ev.dataTransfer.effectAllowed = "move";
            });
            el.addEventListener("dragend", () => {
                el.draggable = false;
            });
            if (hqMode) {
                el.addEventListener("contextmenu", async (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    selectedId = it.id;
                    renderStrip();
                    const picked = await pickClipVersion?.(it, ev);
                    if (picked == null) {
                        return;
                    }
                    if (picked === "") {
                        delete it.media;
                    } else {
                        it.media = picked;
                    }
                    loadedIndex = -1;
                    refreshStrip();
                    notify();
                    const hit = locate(playheadFrame) || { index: items.indexOf(it), local: 0 };
                    loadItem(hit.index, hit.local, playing);
                });
            }

            strip.append(el);
            acc += dur;
        });

        if (!strip.contains(playhead)) {
            strip.append(playhead);
        }
        playhead.style.left = `${playheadFrame * pxPerFrame}px`;
    }

    function refreshStrip() {
        if (!zoomMode) {
            fitZoom();
        }
        renderStrip();
        playheadFrame = Math.min(playheadFrame, totalFrames());
        updatePlayheadPos();
        updateZoomLabel();
    }

    // —— scrubbing ——

    function frameFromClientX(clientX) {
        const rect = strip.getBoundingClientRect();
        const frame = (clientX - rect.left) / pxPerFrame;
        return Math.max(0, Math.min(totalFrames(), Math.round(frame)));
    }

    function scrubTo(clientX, allowSeek = true) {
        const frame = frameFromClientX(clientX);
        if (frame === lastScrubFrame) {
            return;
        }
        lastScrubFrame = frame;
        const hit = locate(frame);
        if (!hit) {
            playheadFrame = 0;
            updatePlayheadPos();
            return;
        }
        if (!allowSeek) {
            playheadFrame = frame;
            updatePlayheadPos();
            return;
        }
        loadItem(hit.index, hit.local, playing);
    }

    /** Runs while dragging so edge scroll advances at a fixed, non-accelerating rate. */
    function scrubLoop() {
        if (!scrubbing) {
            return;
        }
        autoScroll(SCRUB_SCROLL_STEP_PX);
        // Seeks are throttled; the playhead line still tracks the pointer every frame.
        const now = performance.now();
        const seek = now - lastSeekAt >= 60;
        scrubTo(lastPointerX, seek);
        if (seek) {
            lastSeekAt = now;
        }
        scrubRaf = requestAnimationFrame(scrubLoop);
    }

    stripWrap.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) {
            return;
        }
        if (ev.target.closest(".h3lq-timeline__handle")) {
            return; // trim handle owns this gesture
        }
        const itemEl = ev.target.closest(".h3lq-timeline__item");
        if (itemEl) {
            selectedId = itemEl.dataset.id;
            renderStrip();
        }
        if (ev.altKey && itemEl) {
            return; // let the native drag reorder run
        }
        scrubbing = true;
        lastPointerX = ev.clientX;
        lastScrubFrame = -1;
        try {
            stripWrap.setPointerCapture(ev.pointerId);
        } catch {
            /* ignore */
        }
        scrubTo(ev.clientX);
        cancelAnimationFrame(scrubRaf);
        scrubRaf = requestAnimationFrame(scrubLoop);
    });

    stripWrap.addEventListener("pointermove", (ev) => {
        if (scrubbing) {
            lastPointerX = ev.clientX;
        }
    });

    const endScrub = () => {
        if (!scrubbing) {
            return;
        }
        scrubbing = false;
        cancelAnimationFrame(scrubRaf);
        // Land the monitor exactly where the playhead was released.
        const hit = locate(Math.round(playheadFrame));
        if (hit) {
            loadItem(hit.index, hit.local, playing);
        }
    };
    stripWrap.addEventListener("pointerup", endScrub);
    stripWrap.addEventListener("pointercancel", endScrub);
    window.addEventListener("pointerup", endScrub);

    stripWrap.addEventListener(
        "wheel",
        (ev) => {
            if (ev.ctrlKey) {
                ev.preventDefault();
                setZoom(pxPerFrame * (ev.deltaY < 0 ? 1.25 : 0.8));
                return;
            }
            if (zoomMode && ev.deltaY) {
                ev.preventDefault();
                stripWrap.scrollLeft += ev.deltaY;
            }
        },
        { passive: false }
    );

    fitBtn.addEventListener("click", () => setFit());
    zoomInBtn.addEventListener("click", () => setZoom(pxPerFrame * 1.5));
    zoomOutBtn.addEventListener("click", () => setZoom(pxPerFrame / 1.5));

    // —— drops from the bin / reorder ——

    function insertIndexFromX(clientX) {
        const frame = frameFromClientX(clientX);
        let acc = 0;
        for (let i = 0; i < items.length; i++) {
            const dur = Math.max(1, items[i].out_frame - items[i].in_frame);
            if (frame < acc + dur / 2) {
                return i;
            }
            acc += dur;
        }
        return items.length;
    }

    stripWrap.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        ev.dataTransfer.dropEffect = ev.dataTransfer.types.includes("text/h3lq-timeline-id")
            ? "move"
            : "copy";
    });

    async function addPackage(pkg, idx = items.length) {
        if (!hasPreview(pkg)) {
            status.textContent = "Cannot add: no preview.mp4 for this package";
            return;
        }
        const draft = { package: pkg, in_frame: 0, out_frame: 1 };
        draft.out_frame = sourceFramesFor(draft);
        const id = `t${Date.now()}_${pkg}`;
        const item = { id, ...draft };
        items.splice(idx, 0, item);
        selectedId = id;
        loadedIndex = -1;
        refreshStrip();
        notify();
    }

    stripWrap.addEventListener("drop", (ev) => {
        ev.preventDefault();
        const moveId = ev.dataTransfer?.getData("text/h3lq-timeline-id");
        const pkg = ev.dataTransfer?.getData("text/h3lq-package");
        const idx = insertIndexFromX(ev.clientX);
        if (moveId) {
            const from = items.findIndex((it) => it.id === moveId);
            if (from < 0) {
                return;
            }
            const [moved] = items.splice(from, 1);
            const to = from < idx ? idx - 1 : idx;
            items.splice(Math.max(0, to), 0, moved);
            loadedIndex = -1;
            refreshStrip();
            notify();
        } else if (pkg) {
            addPackage(pkg, idx);
        }
    });

    // —— bin ——

    function renderBin() {
        binList.replaceChildren();
        const listed = getListed?.() || [];
        if (!listed.length) {
            binList.append(
                createEl(
                    "div",
                    "h3lq-empty",
                    binEmptyText || "Add packages to the Selected list on the Browse tab first."
                )
            );
            return;
        }
        for (const pkg of listed) {
            const variants = getPackageInfo?.(pkg)?.upscaled || [];
            const row = createEl(
                "div",
                `h3lq-timeline__bin-item${hasPreview(pkg) ? "" : " is-disabled"}`
            );
            row.draggable = hasPreview(pkg);
            row.dataset.package = pkg;
            const img = document.createElement("img");
            img.className = "h3lq-timeline__bin-thumb";
            img.src = thumbUrl(pkg);
            img.alt = "";
            row.append(img);
            const textCol = createEl("div", "h3lq-timeline__bin-text");
            const name = createEl("div", "h3lq-timeline__bin-name", pkg.split("/").pop() || pkg);
            name.title = pkg;
            textCol.append(name);
            if (hqMode && variants.length) {
                textCol.append(
                    createEl(
                        "div",
                        "h3lq-timeline__bin-meta",
                        variants.length === 1
                            ? `HQ ${String(variants[0].index).padStart(3, "0")}`
                            : `${variants.length} HQ iterations`
                    )
                );
            }
            row.append(textCol);
            if (!hasPreview(pkg)) {
                row.append(
                    createEl("div", "h3lq-card__note", hqMode ? "No preview or HQ MP4" : "No preview")
                );
            } else {
                row.addEventListener("dragstart", (ev) => {
                    ev.dataTransfer?.setData("text/h3lq-package", pkg);
                    ev.dataTransfer.effectAllowed = "copy";
                });
                row.addEventListener("mouseenter", (ev) => showHover?.(row, pkg, ev));
                row.addEventListener("mouseleave", () => destroyHover?.());
                row.addEventListener("dblclick", () => {
                    addPackage(pkg);
                });
            }
            binList.append(row);
        }
    }

    // —— export ——

    nameInput.addEventListener("change", () => {
        editName = String(nameInput.value || "edit").trim() || "edit";
        notify();
    });

    playBtn.addEventListener("click", () => togglePlay());

    async function doExport(kind) {
        if (exporting) {
            return;
        }
        if (!items.length) {
            status.textContent = "Nothing to export";
            return;
        }
        exporting = true;
        exportMp4Btn.disabled = true;
        exportXmlBtn.disabled = true;
        exportBundleBtn.disabled = true;
        status.textContent =
            kind === "mp4"
                ? "Exporting MP4…"
                : kind === "bundle"
                  ? "Exporting clip bundle…"
                  : "Exporting FCP7 XML…";
        try {
            const path =
                kind === "mp4"
                    ? "/h3_lq_stash/export_mp4"
                    : kind === "bundle"
                      ? "/h3_lq_stash/export_bundle"
                      : "/h3_lq_stash/export_xml";
            const data = await postJson(path, {
                name: editName,
                fps,
                clips: items.map((it) => {
                    const clip = {
                        package: it.package,
                        in_frame: it.in_frame,
                        out_frame: it.out_frame,
                    };
                    if (it.media) {
                        clip.media = it.media;
                    }
                    return clip;
                }),
            });
            status.textContent = `Wrote ${data.rel || data.path}`;
        } catch (err) {
            status.textContent = String(err.message || err);
        } finally {
            exporting = false;
            exportMp4Btn.disabled = false;
            exportXmlBtn.disabled = false;
            exportBundleBtn.disabled = false;
        }
    }

    exportMp4Btn.addEventListener("click", () => doExport("mp4"));
    exportXmlBtn.addEventListener("click", () => doExport("xml"));
    exportBundleBtn.addEventListener("click", () => doExport("bundle"));

    // —— keys ——

    function isVisible() {
        return root.offsetParent !== null;
    }

    function onKey(ev) {
        if (!isVisible()) {
            return;
        }
        const tag = String(ev.target?.tagName || "").toLowerCase();
        if (tag === "input" || tag === "textarea" || ev.target?.isContentEditable) {
            return;
        }
        if (ev.code === "Space") {
            // Claim Space before the canvas pan shortcut sees it.
            ev.preventDefault();
            ev.stopPropagation();
            togglePlay();
            return;
        }
        if (ev.key === "Delete" || ev.key === "Backspace") {
            if (!selectedId) {
                return;
            }
            ev.preventDefault();
            items = items.filter((it) => it.id !== selectedId);
            selectedId = items[0]?.id || null;
            loadedIndex = -1;
            pause();
            playheadFrame = Math.min(playheadFrame, totalFrames());
            refreshStrip();
            notify();
        }
    }
    window.addEventListener("keydown", onKey, true);

    const onResize = () => {
        if (isVisible()) {
            refreshStrip();
        }
    };
    window.addEventListener("resize", onResize);

    function refresh() {
        for (const it of items) {
            const maxF = sourceFramesFor(it);
            if (it.out_frame > maxF) {
                it.out_frame = maxF;
            }
            if (it.out_frame <= it.in_frame) {
                it.out_frame = Math.min(maxF, it.in_frame + 1);
            }
        }
        renderBin();
        refreshStrip();
        notify();
        if (items.length && loadedIndex < 0) {
            const hit = locate(playheadFrame) || { index: 0, local: 0 };
            loadItem(hit.index, hit.local, false);
        }
    }

    function destroy() {
        window.removeEventListener("keydown", onKey, true);
        window.removeEventListener("resize", onResize);
        window.removeEventListener("pointerup", endScrub);
        cancelAnimationFrame(scrubRaf);
        loadToken += 1;
        pause();
        finishAdvance();
        for (const el of [videoA, videoB]) {
            el.removeAttribute("src");
            el.load?.();
        }
    }

    for (const it of items) {
        if (!it.out_frame || it.out_frame <= it.in_frame) {
            it.out_frame = sourceFramesFor(it);
        }
    }
    refresh();

    return { root, refresh, destroy, serialize, getItems: () => items };
}
