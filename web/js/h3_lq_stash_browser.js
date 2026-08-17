import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createTimelineTab, parseEdit } from "./h3_lq_stash_timeline.js";

const NODE_NAME = "MiniMaxH3LQPackageLoad";
const SAVE_NODE_NAME = "MiniMaxH3LQPackageSave";
const COLLECT_NODE_NAME = "MiniMaxH3UpscaleCollect";
const COLLECT_WIDGET_NAME = "h3lq_collect_ui";
const COLLECT_MIN_W = 280;
const COLLECT_DEFAULT_H = 340;
const COLLECT_PREVIEW_MIN_H = 160;
const COLLECT_VERTICAL_CHROME = 92;
const CSS_HREF = "/extensions/ComfyUI-MiniMaxH3_LatentUpscaler/css/h3_lq_stash.css";
const DEFAULT_ROOT = "h3_lq_stash";
const LOAD_SOURCE_SELECTION = "selection list";
const LOAD_SOURCE_TIMELINE_CROPS = "timeline order and crops";
const CHUNK_MODE_DIRECT = "direct load";
const CHUNK_MODE_SCENES = "scene aware chunks (experimental)";
const CHUNK_MODE_SCENES_LEGACY = "scene aware chunks";

function isSceneChunkMode(value) {
    return value === CHUNK_MODE_SCENES || value === CHUNK_MODE_SCENES_LEGACY;
}
const SCENE_CAVEAT =
    "Scene chunks: preview.mp4 required. Text prompt stays global; cuts snap to H3 tokens; long scenes soft-split (no overlap lock). Outputs are lists (direct = 1 item).";

function ensureCss() {
    if (document.querySelector(`link[href="${CSS_HREF}"]`)) {
        return;
    }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = CSS_HREF;
    document.head.appendChild(link);
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
    node.setDirtyCanvas?.(true, true);
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

function parseSelection(raw) {
    const text = String(raw || "").trim();
    if (!text) {
        return { version: 1, root: DEFAULT_ROOT, packages: [] };
    }
    try {
        const data = JSON.parse(text);
        if (!data || typeof data !== "object") {
            return { version: 1, root: DEFAULT_ROOT, packages: [] };
        }
        return {
            version: data.version || 1,
            root: data.root || DEFAULT_ROOT,
            packages: Array.isArray(data.packages)
                ? data.packages.map((p) => String(p || "").replace(/\\/g, "/")).filter(Boolean)
                : [],
        };
    } catch {
        return { version: 1, root: DEFAULT_ROOT, packages: [] };
    }
}

async function getJson(path) {
    const res = await api.fetchApi(path);
    const data = await res.json();
    if (!res.ok || data?.ok === false) {
        throw new Error(data?.error || `Request failed (${res.status})`);
    }
    return data;
}

async function postJson(path, body) {
    const res = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || data?.ok === false) {
        throw new Error(data?.error || `Request failed (${res.status})`);
    }
    return data;
}

function hideMultilineWidget(node, name) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.computeSize = () => [0, -4];
    if (widget.element) {
        widget.element.style.display = "none";
    }
    const origDraw = widget.draw;
    widget.draw = function () {
        if (this.element) {
            this.element.style.display = "none";
        }
        if (typeof origDraw === "function") {
            return;
        }
    };
}

function hideSelectionWidget(node) {
    hideMultilineWidget(node, "selection_json");
    hideMultilineWidget(node, "edit_json");
}

const SCENE_ONLY_WIDGETS = ["scene_threshold", "min_scene_seconds", "max_scene_seconds"];

function setWidgetVisible(widget, visible) {
    if (!widget) {
        return;
    }
    if (visible) {
        if (!widget.__h3lqHidden) {
            return;
        }
        if (widget.__h3lqOrigComputeSize) {
            widget.computeSize = widget.__h3lqOrigComputeSize;
        } else {
            delete widget.computeSize;
        }
        widget.hidden = false;
        widget.__h3lqHidden = false;
        if (widget.element) {
            widget.element.style.display = "";
        }
        return;
    }
    if (widget.__h3lqHidden) {
        return;
    }
    // Only an own computeSize is worth restoring; most widgets inherit theirs.
    widget.__h3lqOrigComputeSize = Object.hasOwn(widget, "computeSize")
        ? widget.computeSize
        : undefined;
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    widget.__h3lqHidden = true;
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function syncSceneWidgets(node) {
    const show = isSceneChunkMode(getWidget(node, "chunk_mode")?.value);
    for (const name of SCENE_ONLY_WIDGETS) {
        setWidgetVisible(getWidget(node, name), show);
    }
    // Shrink / grow the node so hidden widgets don't leave empty space.
    const size = node.computeSize?.() || node.size;
    if (size) {
        node.setSize?.([node.size[0], size[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function activeLoadEntries(node) {
    const source = String(getWidget(node, "load_source")?.value || LOAD_SOURCE_SELECTION);
    if (source === LOAD_SOURCE_SELECTION) {
        return parseSelection(getWidget(node, "selection_json")?.value).packages.map((pkg) => ({
            package: pkg,
        }));
    }
    return parseEdit(getWidget(node, "edit_json")?.value).clips;
}

function updateSummary(node) {
    const source = String(getWidget(node, "load_source")?.value || LOAD_SOURCE_SELECTION);
    const chunkMode = String(getWidget(node, "chunk_mode")?.value || CHUNK_MODE_DIRECT);
    const entries = activeLoadEntries(node);
    const index = Number(getWidget(node, "index")?.value || 0);
    const n = entries.length;
    let text = `Source: ${source}\nChunk: ${chunkMode}`;
    text +=
        isSceneChunkMode(chunkMode)
            ? "\nScenes: list fan-out (preview required)"
            : "\nDirect: 1-item list";
    text += `\nClips: ${n}`;
    if (n > 0) {
        const i = ((index % n) + n) % n;
        const entry = entries[i];
        text += `\nindex ${i}/${n} → ${entry.package}`;
        if (source === LOAD_SOURCE_TIMELINE_CROPS) {
            text += ` [${entry.in_frame}–${entry.out_frame}f)`;
        }
    } else {
        text += source === LOAD_SOURCE_SELECTION
            ? "\n(open Browse Saved Packages)"
            : "\n(add clips on the Timeline tab)";
    }
    if (node.__h3lqSummary) {
        node.__h3lqSummary.textContent = text;
    }
    if (node.__h3lqCaveat) {
        const show = isSceneChunkMode(chunkMode);
        node.__h3lqCaveat.hidden = !show;
        node.__h3lqCaveat.textContent = show ? SCENE_CAVEAT : "";
    }
    return text;
}

function removeControlWidget(node, hostName) {
    const widgets = node.widgets || [];
    for (let i = widgets.length - 1; i >= 0; i--) {
        if (String(widgets[i].name || "").endsWith("control_after_generate")) {
            widgets.splice(i, 1);
        }
    }
    const host = getWidget(node, hostName);
    if (host) {
        host.linkedWidgets = [];
    }
}

function setupIndexAdvance(node) {
    const indexWidget = getWidget(node, "index");
    if (!indexWidget || indexWidget.__h3lqAdvance) {
        return;
    }
    indexWidget.afterQueued = () => {
        if (String(getWidget(node, "index_mode")?.value || "increment") !== "increment") {
            return;
        }
        const n = activeLoadEntries(node).length;
        const current = Number(indexWidget.value || 0);
        const next = n > 0 ? (current + 1) % n : current + 1;
        setWidgetValue(node, "index", next);
        updateSummary(node);
    };
    indexWidget.__h3lqAdvance = true;
}

function showModal(node) {
    ensureCss();
    node.__h3lqCloseModal?.();

    const initial = parseSelection(getWidget(node, "selection_json")?.value);
    const initialEdit = parseEdit(getWidget(node, "edit_json")?.value);
    let rootRel = initial.root || DEFAULT_ROOT;
    let pathRel = "";
    if (initial.packages[0] && rootRel === DEFAULT_ROOT) {
        const parts = initial.packages[0].split("/");
        if (parts.length > 1) {
            pathRel = parts.slice(0, -1).join("/");
        }
    }

    /** @type {Set<string>} gallery staging (not yet on list) */
    const staged = new Set();
    /** @type {string[]} ordered selected list */
    let listed = [...initial.packages];
    /** @type {Map<string, object>} package cache for thumbs/preview/meta snippets */
    const packageCache = new Map();

    let packages = [];
    let folders = [];
    let focusedPackage = listed[0] || null;
    let lastClicked = -1;
    let closed = false;
    let hoverVideo = null;
    let activeTab = "browse";

    const overlay = createEl("div", "h3lq-overlay");
    const modal = createEl("div", "h3lq-modal");
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Saved Packages Browser");

    const header = createEl("div", "h3lq-modal__header");
    header.append(createEl("h2", "h3lq-modal__title", "Saved Packages"));

    const rootInput = document.createElement("input");
    rootInput.className = "h3lq-path";
    rootInput.type = "text";
    rootInput.placeholder = "Project root under output (default h3_lq_stash)";
    rootInput.value = pathRel ? `${rootRel}/${pathRel}` : rootRel;

    const upBtn = createEl("button", "h3lq-btn", "Up");
    upBtn.type = "button";
    const refreshBtn = createEl("button", "h3lq-btn", "Refresh");
    refreshBtn.type = "button";
    const applyBtn = createEl("button", "h3lq-btn h3lq-btn--primary", "Apply selection");
    applyBtn.type = "button";
    const cancelBtn = createEl("button", "h3lq-btn", "Cancel");
    cancelBtn.type = "button";
    header.append(rootInput, upBtn, refreshBtn, applyBtn, cancelBtn);

    const tabs = createEl("div", "h3lq-tabs");
    const browseTabBtn = createEl("button", "h3lq-tab is-active", "Browse");
    browseTabBtn.type = "button";
    const timelineTabBtn = createEl("button", "h3lq-tab", "Timeline");
    timelineTabBtn.type = "button";
    tabs.append(browseTabBtn, timelineTabBtn);

    const status = createEl("div", "h3lq-status", "Loading…");

    // —— Browse tab ——
    const browseBody = createEl("div", "h3lq-body h3lq-body--browse");

    const folderPane = createEl("div", "h3lq-pane");
    folderPane.append(createEl("div", "h3lq-pane__label", "Folders"));
    const folderList = createEl("div", "h3lq-folder-list");
    folderPane.append(folderList);

    const galleryPane = createEl("div", "h3lq-pane");
    const galleryHead = createEl("div", "h3lq-pane__head");
    galleryHead.append(createEl("div", "h3lq-pane__label", "Packages"));
    const galleryTools = createEl("div", "h3lq-toolbar");
    const addToListBtn = createEl("button", "h3lq-btn h3lq-btn--primary", "Add to list");
    addToListBtn.type = "button";
    const selectAllBtn = createEl("button", "h3lq-btn", "Select all");
    selectAllBtn.type = "button";
    const clearStagedBtn = createEl("button", "h3lq-btn", "Clear staged");
    clearStagedBtn.type = "button";
    galleryTools.append(addToListBtn, selectAllBtn, clearStagedBtn);
    galleryHead.append(galleryTools);
    const gallery = createEl("div", "h3lq-gallery");
    galleryPane.append(galleryHead, gallery);

    const listPane = createEl("div", "h3lq-pane");
    const listHead = createEl("div", "h3lq-pane__head");
    listHead.append(createEl("div", "h3lq-pane__label", "Selected list"));
    const clearListBtn = createEl("button", "h3lq-btn", "Clear all");
    clearListBtn.type = "button";
    listHead.append(clearListBtn);
    const selectedList = createEl("div", "h3lq-selected-list");
    listPane.append(listHead, selectedList);

    const metaPane = createEl("div", "h3lq-pane");
    metaPane.append(createEl("div", "h3lq-pane__label", "Details"));
    const detailsVideoWrap = createEl("div", "h3lq-details-video-wrap");
    const detailsVideo = document.createElement("video");
    detailsVideo.className = "h3lq-details-video";
    detailsVideo.controls = true;
    detailsVideo.muted = true;
    detailsVideo.loop = true;
    detailsVideo.playsInline = true;
    detailsVideo.preload = "metadata";
    const detailsPlaceholder = createEl("div", "h3lq-details-placeholder", "No preview for this package");
    detailsVideoWrap.append(detailsVideo, detailsPlaceholder);
    const metaBox = createEl("div", "h3lq-meta", "Select a package");
    metaPane.append(detailsVideoWrap, metaBox);

    browseBody.append(folderPane, galleryPane, listPane, metaPane);

    // —— Timeline tab ——
    const timelineBody = createEl("div", "h3lq-body h3lq-body--timeline");
    timelineBody.hidden = true;

    const timeline = createTimelineTab({
        initialEdit,
        getListed: () => listed,
        getPackageInfo: (pkg) => packageCache.get(pkg) || null,
        showHover: (card, pkg, ev) => showHover(card, pkg, ev),
        destroyHover: () => destroyHover(),
        postJson,
        onEditChange: (edit) => {
            setWidgetValue(node, "edit_json", JSON.stringify(edit));
        },
    });
    timelineBody.append(timeline.root);

    modal.append(header, tabs, status, browseBody, timelineBody);
    overlay.append(modal);
    document.body.append(overlay);

    function destroyHover() {
        if (hoverVideo) {
            hoverVideo.pause?.();
            hoverVideo.removeAttribute("src");
            hoverVideo.load?.();
            hoverVideo.remove();
            hoverVideo = null;
        }
    }

    function close() {
        if (closed) {
            return;
        }
        closed = true;
        destroyHover();
        timeline.destroy();
        detailsVideo.removeAttribute("src");
        overlay.remove();
        node.__h3lqCloseModal = null;
        window.removeEventListener("keydown", onKey);
    }
    node.__h3lqCloseModal = close;

    function onKey(ev) {
        if (ev.key === "Escape") {
            close();
        }
    }
    window.addEventListener("keydown", onKey);

    function setTab(name) {
        activeTab = name;
        browseTabBtn.classList.toggle("is-active", name === "browse");
        timelineTabBtn.classList.toggle("is-active", name === "timeline");
        browseBody.hidden = name !== "browse";
        timelineBody.hidden = name !== "timeline";
        // Path controls only meaningful on Browse
        rootInput.disabled = name !== "browse";
        upBtn.disabled = name !== "browse";
        refreshBtn.disabled = name !== "browse";
        if (name === "timeline") {
            destroyHover();
            timeline.refresh();
        }
    }
    browseTabBtn.addEventListener("click", () => setTab("browse"));
    timelineTabBtn.addEventListener("click", () => setTab("timeline"));

    function parseHeaderPath() {
        const raw = String(rootInput.value || "")
            .trim()
            .replace(/\\/g, "/")
            .replace(/^\/+|\/+$/g, "");
        if (!raw) {
            rootRel = DEFAULT_ROOT;
            pathRel = "";
            return;
        }
        const parts = raw.split("/").filter(Boolean);
        if (parts[0] === DEFAULT_ROOT || parts.length === 1) {
            rootRel = parts[0] || DEFAULT_ROOT;
            pathRel = parts.slice(1).join("/");
        } else if (parts[0] === "output" && parts[1]) {
            rootRel = parts[1];
            pathRel = parts.slice(2).join("/");
        } else {
            rootRel = parts[0];
            pathRel = parts.slice(1).join("/");
        }
    }

    function syncHeaderInput() {
        rootInput.value = pathRel ? `${rootRel}/${pathRel}` : rootRel;
    }

    function rememberPackages(list) {
        for (const pkg of list || []) {
            packageCache.set(pkg.package, pkg);
        }
    }

    function setDetailsPreview(pkg) {
        const info = pkg ? packageCache.get(pkg) : null;
        if (info?.has_preview && info.preview_url) {
            detailsPlaceholder.hidden = true;
            detailsVideo.hidden = false;
            if (detailsVideo.src !== new URL(info.preview_url, location.origin).href) {
                detailsVideo.src = info.preview_url;
            }
            detailsVideo.play?.().catch(() => {});
        } else {
            detailsVideo.pause?.();
            detailsVideo.removeAttribute("src");
            detailsVideo.hidden = true;
            detailsPlaceholder.hidden = false;
            detailsPlaceholder.textContent = pkg
                ? "No preview for this package"
                : "Select a package";
        }
    }

    async function loadMeta(pkg) {
        if (!pkg) {
            metaBox.replaceChildren(createEl("div", null, "Select a package"));
            setDetailsPreview(null);
            return;
        }
        setDetailsPreview(pkg);
        metaBox.textContent = "Loading…";
        try {
            const data = await getJson(`/h3_lq_stash/meta?package=${encodeURIComponent(pkg)}`);
            const m = data.meta || {};
            // Enrich cache with frame_count / fps for timeline
            const prev = packageCache.get(pkg) || { package: pkg };
            packageCache.set(pkg, {
                ...prev,
                frame_count: m.frame_count,
                fps: m.fps,
                has_preview: Boolean(m.has_preview) || prev.has_preview,
                preview_url:
                    prev.preview_url ||
                    (m.has_preview || m.preview
                        ? `/h3_lq_stash/preview?package=${encodeURIComponent(pkg)}`
                        : null),
                thumb_url: prev.thumb_url || `/h3_lq_stash/thumb?package=${encodeURIComponent(pkg)}`,
            });
            setDetailsPreview(pkg);
            metaBox.replaceChildren();
            metaBox.append(
                createEl("div", "h3lq-meta__title", m.package_rel || m.package_name || pkg)
            );
            const lines = [
                `created_at: ${m.created_at || "—"}`,
                `video_shape: ${JSON.stringify(m.video_shape || null)}`,
                `audio_shape: ${JSON.stringify(m.audio_shape || null)}`,
                `has_preview: ${Boolean(m.has_preview)}`,
                `fps: ${m.fps ?? "—"}  frames: ${m.frame_count ?? "—"}`,
                `seed: ${m.seed ?? "—"}`,
                "",
                "note:",
                m.note || "(none)",
                "",
                "prompt:",
                m.prompt || "(none)",
            ];
            metaBox.append(createEl("div", null, lines.join("\n")));
            if (activeTab === "timeline") {
                timeline.refresh();
            }
        } catch (err) {
            metaBox.textContent = String(err.message || err);
        }
    }

    function renderFolders() {
        folderList.replaceChildren();
        if (!folders.length) {
            folderList.append(createEl("div", "h3lq-empty", "No subfolders"));
            return;
        }
        for (const folder of folders) {
            const btn = createEl("button", "h3lq-folder", folder.name);
            btn.type = "button";
            btn.addEventListener("click", () => {
                pathRel = folder.path;
                syncHeaderInput();
                refresh();
            });
            folderList.append(btn);
        }
    }

    function showHover(card, pkg, ev) {
        destroyHover();
        const item = packageCache.get(pkg) || packages.find((p) => p.package === pkg);
        if (!item?.has_preview || !item.preview_url) {
            return;
        }
        const video = document.createElement("video");
        video.className = "h3lq-hover-video";
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.autoplay = true;
        video.src = item.preview_url;
        document.body.append(video);
        hoverVideo = video;
        const move = (e) => {
            const x = Math.min(e.clientX + 16, window.innerWidth - video.offsetWidth - 8);
            const y = Math.min(e.clientY + 16, window.innerHeight - video.offsetHeight - 8);
            video.style.left = `${Math.max(8, x)}px`;
            video.style.top = `${Math.max(8, y)}px`;
        };
        move(ev);
        card.__h3lqMove = move;
        card.addEventListener("mousemove", move);
        video.play?.().catch(() => {});
    }

    function renderGallery() {
        gallery.replaceChildren();
        if (!packages.length) {
            gallery.append(createEl("div", "h3lq-empty", "No packages in this folder"));
            return;
        }
        packages.forEach((pkg, idx) => {
            const isStaged = staged.has(pkg.package);
            const onList = listed.includes(pkg.package);
            const card = createEl(
                "div",
                `h3lq-card${isStaged ? " is-staged" : ""}${onList ? " is-on-list" : ""}${
                    focusedPackage === pkg.package ? " is-focused" : ""
                }`
            );
            if (onList) {
                card.append(
                    createEl("div", "h3lq-card__badge", String(listed.indexOf(pkg.package) + 1))
                );
            }
            if (pkg.has_thumb) {
                const img = document.createElement("img");
                img.className = "h3lq-card__thumb";
                img.loading = "lazy";
                img.src = pkg.thumb_url;
                img.alt = pkg.name;
                card.append(img);
            } else {
                card.append(createEl("div", "h3lq-card__placeholder", "No preview"));
            }
            card.append(createEl("div", "h3lq-card__name", pkg.name));
            card.append(createEl("div", "h3lq-card__note", pkg.note || pkg.prompt_snippet || " "));

            card.addEventListener("click", (ev) => {
                focusedPackage = pkg.package;
                if (ev.shiftKey && lastClicked >= 0) {
                    const a = Math.min(lastClicked, idx);
                    const b = Math.max(lastClicked, idx);
                    const turningOn = !staged.has(pkg.package);
                    for (let i = a; i <= b; i++) {
                        const p = packages[i].package;
                        if (turningOn) {
                            staged.add(p);
                        } else {
                            staged.delete(p);
                        }
                    }
                } else {
                    if (staged.has(pkg.package)) {
                        staged.delete(pkg.package);
                    } else {
                        staged.add(pkg.package);
                    }
                    lastClicked = idx;
                }
                status.textContent = `Staged ${staged.size} · listed ${listed.length} · ${packages.length} in folder`;
                renderGallery();
                loadMeta(focusedPackage);
            });

            card.addEventListener("mouseenter", (ev) => showHover(card, pkg.package, ev));
            card.addEventListener("mouseleave", () => {
                if (card.__h3lqMove) {
                    card.removeEventListener("mousemove", card.__h3lqMove);
                    card.__h3lqMove = null;
                }
                destroyHover();
            });

            gallery.append(card);
        });
    }

    function renderSelectedList() {
        selectedList.replaceChildren();
        if (!listed.length) {
            selectedList.append(createEl("div", "h3lq-empty", "Empty — stage cards and Add to list"));
            return;
        }
        listed.forEach((pkg, idx) => {
            const info = packageCache.get(pkg);
            const row = createEl(
                "div",
                `h3lq-selected-row${focusedPackage === pkg ? " is-focused" : ""}`
            );
            row.draggable = true;
            row.dataset.package = pkg;
            row.dataset.index = String(idx);

            const thumb = document.createElement("img");
            thumb.className = "h3lq-selected-row__thumb";
            thumb.src = info?.thumb_url || `/h3_lq_stash/thumb?package=${encodeURIComponent(pkg)}`;
            thumb.alt = "";
            row.append(thumb);

            const name = createEl("div", "h3lq-selected-row__name", pkg);
            name.title = pkg;
            row.append(name);

            const up = createEl("button", "h3lq-btn h3lq-btn--tiny", "↑");
            up.type = "button";
            up.title = "Move up";
            const down = createEl("button", "h3lq-btn h3lq-btn--tiny", "↓");
            down.type = "button";
            down.title = "Move down";
            const remove = createEl("button", "h3lq-btn h3lq-btn--tiny h3lq-btn--danger", "×");
            remove.type = "button";
            remove.title = "Remove";

            up.addEventListener("click", (ev) => {
                ev.stopPropagation();
                if (idx <= 0) {
                    return;
                }
                const tmp = listed[idx - 1];
                listed[idx - 1] = listed[idx];
                listed[idx] = tmp;
                renderSelectedList();
                renderGallery();
            });
            down.addEventListener("click", (ev) => {
                ev.stopPropagation();
                if (idx >= listed.length - 1) {
                    return;
                }
                const tmp = listed[idx + 1];
                listed[idx + 1] = listed[idx];
                listed[idx] = tmp;
                renderSelectedList();
                renderGallery();
            });
            remove.addEventListener("click", (ev) => {
                ev.stopPropagation();
                listed = listed.filter((p) => p !== pkg);
                if (focusedPackage === pkg) {
                    focusedPackage = listed[0] || null;
                    loadMeta(focusedPackage);
                }
                renderSelectedList();
                renderGallery();
                status.textContent = `Listed ${listed.length}`;
                timeline.refresh();
            });

            row.append(up, down, remove);

            row.addEventListener("click", () => {
                focusedPackage = pkg;
                renderSelectedList();
                renderGallery();
                loadMeta(pkg);
            });

            row.addEventListener("dragstart", (ev) => {
                ev.dataTransfer?.setData("text/h3lq-list-index", String(idx));
                ev.dataTransfer.effectAllowed = "move";
            });
            row.addEventListener("dragover", (ev) => {
                ev.preventDefault();
                ev.dataTransfer.dropEffect = "move";
            });
            row.addEventListener("drop", (ev) => {
                ev.preventDefault();
                const from = Number(ev.dataTransfer?.getData("text/h3lq-list-index"));
                if (Number.isNaN(from) || from === idx) {
                    return;
                }
                const [moved] = listed.splice(from, 1);
                listed.splice(idx, 0, moved);
                renderSelectedList();
                renderGallery();
                timeline.refresh();
            });

            selectedList.append(row);
        });
    }

    addToListBtn.addEventListener("click", () => {
        let added = 0;
        for (const pkg of staged) {
            if (!listed.includes(pkg)) {
                listed.push(pkg);
                added += 1;
            }
        }
        staged.clear();
        status.textContent = `Added ${added} · listed ${listed.length}`;
        renderGallery();
        renderSelectedList();
        timeline.refresh();
    });

    selectAllBtn.addEventListener("click", () => {
        for (const pkg of packages) {
            staged.add(pkg.package);
        }
        renderGallery();
        status.textContent = `Staged ${staged.size}`;
    });

    clearStagedBtn.addEventListener("click", () => {
        staged.clear();
        renderGallery();
        status.textContent = `Staged 0 · listed ${listed.length}`;
    });

    clearListBtn.addEventListener("click", () => {
        if (!listed.length) {
            return;
        }
        if (!window.confirm(`Clear all ${listed.length} package(s) from the selected list?`)) {
            return;
        }
        listed = [];
        focusedPackage = null;
        renderSelectedList();
        renderGallery();
        loadMeta(null);
        timeline.refresh();
        status.textContent = "List cleared";
    });

    async function refresh() {
        parseHeaderPath();
        syncHeaderInput();
        status.textContent = "Loading…";
        destroyHover();
        try {
            const q = `root=${encodeURIComponent(rootRel)}&path=${encodeURIComponent(pathRel)}`;
            const [tree, list] = await Promise.all([
                getJson(`/h3_lq_stash/tree?${q}`),
                getJson(`/h3_lq_stash/list?${q}`),
            ]);
            folders = tree.folders || [];
            packages = list.packages || [];
            rememberPackages(packages);
            // Also ensure listed packages have cache entries
            for (const pkg of listed) {
                if (!packageCache.has(pkg)) {
                    packageCache.set(pkg, {
                        package: pkg,
                        name: pkg.split("/").pop(),
                        has_preview: true,
                        has_thumb: true,
                        thumb_url: `/h3_lq_stash/thumb?package=${encodeURIComponent(pkg)}`,
                        preview_url: `/h3_lq_stash/preview?package=${encodeURIComponent(pkg)}`,
                    });
                }
            }
            status.textContent = `${packages.length} packages · ${folders.length} folders · staged ${staged.size} · listed ${listed.length}`;
            renderFolders();
            renderGallery();
            renderSelectedList();
            if (focusedPackage) {
                loadMeta(focusedPackage);
            }
            if (activeTab === "timeline") {
                timeline.refresh();
            }
        } catch (err) {
            status.textContent = String(err.message || err);
            folders = [];
            packages = [];
            renderFolders();
            renderGallery();
            renderSelectedList();
        }
    }

    upBtn.addEventListener("click", () => {
        parseHeaderPath();
        if (pathRel) {
            const parts = pathRel.split("/").filter(Boolean);
            parts.pop();
            pathRel = parts.join("/");
        } else if (rootRel !== DEFAULT_ROOT) {
            const parts = rootRel.split("/").filter(Boolean);
            if (parts.length > 1) {
                pathRel = "";
                rootRel = parts.slice(0, -1).join("/");
            }
        }
        syncHeaderInput();
        refresh();
    });

    refreshBtn.addEventListener("click", () => refresh());
    rootInput.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
            refresh();
        }
    });

    cancelBtn.addEventListener("click", () => close());
    overlay.addEventListener("click", (ev) => {
        if (ev.target === overlay) {
            close();
        }
    });

    applyBtn.addEventListener("click", () => {
        parseHeaderPath();
        const payload = {
            version: 1,
            root: rootRel,
            packages: [...listed],
        };
        setWidgetValue(node, "selection_json", JSON.stringify(payload));
        // Persist current timeline as well
        setWidgetValue(node, "edit_json", JSON.stringify(timeline.serialize()));
        const indexWidget = getWidget(node, "index");
        const activeCount = activeLoadEntries(node).length;
        if (indexWidget && activeCount) {
            const cur = Number(indexWidget.value || 0);
            if (cur >= activeCount) {
                setWidgetValue(node, "index", 0);
            }
        }
        updateSummary(node);
        close();
    });

    // Seed cache stubs for initial listed so timeline bin works before first list fetch
    for (const pkg of listed) {
        packageCache.set(pkg, {
            package: pkg,
            name: pkg.split("/").pop(),
            has_preview: true,
            has_thumb: true,
            thumb_url: `/h3_lq_stash/thumb?package=${encodeURIComponent(pkg)}`,
            preview_url: `/h3_lq_stash/preview?package=${encodeURIComponent(pkg)}`,
        });
    }
    renderSelectedList();
    refresh();
}

function setupNode(node) {
    ensureCss();
    hideSelectionWidget(node);
    removeControlWidget(node, "index");
    setupIndexAdvance(node);

    const wrap = document.createElement("div");
    const summary = createEl("div", "h3lq-node-summary", "");
    const caveat = createEl("div", "h3lq-node-caveat", "");
    caveat.hidden = true;
    const openBtn = createEl("button", "h3lq-btn", "Browse Saved Packages");
    openBtn.type = "button";
    openBtn.onclick = (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        showModal(node);
    };
    wrap.append(summary, caveat, openBtn);
    node.__h3lqSummary = summary;
    node.__h3lqCaveat = caveat;

    node.addDOMWidget("h3lq_ui", "h3lq_ui", wrap, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => (isSceneChunkMode(getWidget(node, "chunk_mode")?.value) ? 110 : 80),
        getValue() {
            return "";
        },
        setValue() {},
    });

    const bump = () => {
        removeControlWidget(node, "index");
        hideSelectionWidget(node);
        syncSceneWidgets(node);
        updateSummary(node);
    };
    for (const name of [
        "selection_json",
        "edit_json",
        "load_source",
        "chunk_mode",
        "index",
        "index_mode",
    ]) {
        const widget = getWidget(node, name);
        if (!widget || widget.__h3lqWrapped) {
            continue;
        }
        const prev = widget.callback;
        widget.callback = function () {
            const result = prev?.apply(this, arguments);
            bump();
            return result;
        };
        widget.__h3lqWrapped = true;
    }
    bump();

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        hideSelectionWidget(this);
        queueMicrotask(() => bump());
        return result;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        this.__h3lqCloseModal?.();
        return onRemoved?.apply(this, arguments);
    };
}

function uniquePackages(list) {
    const seen = new Set();
    const out = [];
    for (const raw of list || []) {
        const pkg = String(raw || "").replace(/\\/g, "/");
        if (!pkg || seen.has(pkg)) {
            continue;
        }
        seen.add(pkg);
        out.push(pkg);
    }
    return out;
}

function connectedLoadNode(node) {
    const input = (node.inputs || []).find((slot) => slot.name === "package_pipe");
    if (!input || input.link == null) {
        return null;
    }
    const link = app.graph?.links?.[input.link];
    if (!link) {
        return null;
    }
    const origin = app.graph.getNodeById(link.origin_id);
    if (!origin) {
        return null;
    }
    if (origin.comfyClass === NODE_NAME || origin.type === NODE_NAME) {
        return origin;
    }
    return origin;
}

function packagesFromLoadNode(loadNode) {
    if (!loadNode) {
        return [];
    }
    const source = String(getWidget(loadNode, "load_source")?.value || LOAD_SOURCE_SELECTION);
    if (source === LOAD_SOURCE_SELECTION) {
        return uniquePackages(parseSelection(getWidget(loadNode, "selection_json")?.value).packages);
    }
    return uniquePackages(parseEdit(getWidget(loadNode, "edit_json")?.value).clips.map((c) => c.package));
}

function loadNodeEdit(loadNode) {
    return parseEdit(getWidget(loadNode, "edit_json")?.value);
}

function collectBinPackages(loadNode) {
    const fromEdit = uniquePackages(loadNodeEdit(loadNode).clips.map((c) => c.package));
    return fromEdit.length ? fromEdit : packagesFromLoadNode(loadNode);
}

function collectPreviewSrc(meta) {
    if (!meta?.filename) {
        return "";
    }
    const params = new URLSearchParams({
        filename: meta.filename,
        subfolder: meta.subfolder || "",
        type: meta.type || "output",
        rand: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function collectNativeWidgetsHeight(node) {
    const rowH = (window.LiteGraph && window.LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    let height = 0;
    for (const widget of node.widgets || []) {
        if (widget === node.__h3lqCollectWidget || widget.name === COLLECT_WIDGET_NAME) {
            continue;
        }
        if (widget.hidden || widget.__h3lqHidden) {
            continue;
        }
        let widgetHeight = rowH;
        if (typeof widget.computeSize === "function") {
            const computed = widget.computeSize(node.size?.[0] || COLLECT_MIN_W);
            widgetHeight = computed && computed[1] > 0 ? computed[1] : 0;
        }
        if (widgetHeight > 0) {
            height += widgetHeight + 4;
        }
    }
    return height;
}

function collectPreviewWidgetSize(node, width) {
    const nodeHeight = Number(node.size?.[1]) || COLLECT_DEFAULT_H;
    return [
        Math.max(Number(width) || COLLECT_MIN_W, COLLECT_MIN_W),
        Math.max(
            COLLECT_PREVIEW_MIN_H,
            nodeHeight - COLLECT_VERTICAL_CHROME - collectNativeWidgetsHeight(node)
        ),
    ];
}

function syncCollectMute(node) {
    const video = node.__h3lqCollectVideo;
    if (!video) {
        return;
    }
    video.muted = !node.__h3lqCollectHovering;
}

function applyCollectPreview(node, meta) {
    const wrap = node.__h3lqCollectPreview;
    const video = node.__h3lqCollectVideo;
    const badge = node.__h3lqCollectBadge;
    if (!wrap || !video || !meta?.filename) {
        return;
    }
    node.properties = node.properties || {};
    node.properties.h3CollectPreview = {
        filename: meta.filename,
        subfolder: meta.subfolder || "",
        type: meta.type || "output",
        has_audio: Boolean(meta.has_audio),
        width: Number(meta.width) || 0,
        height: Number(meta.height) || 0,
        frame_rate: Number(meta.frame_rate) || 24,
    };
    const src = collectPreviewSrc(meta);
    wrap.classList.add("has-video");
    video.muted = true;
    video.src = src;
    video.load();
    const playResult = video.play?.();
    if (playResult?.then) {
        playResult.then(() => syncCollectMute(node)).catch(() => syncCollectMute(node));
    } else {
        syncCollectMute(node);
    }
    if (badge) {
        const rel = [meta.subfolder, meta.filename].filter(Boolean).join("/");
        badge.textContent = rel;
        badge.hidden = false;
        badge.title = rel;
    }
    node.setDirtyCanvas?.(true, true);
}

function pickVariant(pkg, variants, opts = {}) {
    const includeLq = Boolean(opts.includeLq);
    const current = opts.current == null ? null : String(opts.current);
    return new Promise((resolve) => {
        const shield = createEl("div", "h3lq-picker-shield");
        const overlay = createEl("div", "h3lq-picker");
        overlay.append(
            createEl("div", "h3lq-picker__title", `Choose version — ${pkg.split("/").pop()}`)
        );
        const list = createEl("div", "h3lq-picker__list");
        const close = (value = null) => {
            shield.remove();
            overlay.remove();
            window.removeEventListener("keydown", onKey, true);
            resolve(value);
        };
        if (includeLq) {
            const btn = createEl("button", "h3lq-picker__item is-lq", "");
            btn.type = "button";
            if (!current) {
                btn.classList.add("is-current");
            }
            btn.append(
                createEl("div", "h3lq-picker__name", "LQ preview"),
                createEl("div", "h3lq-picker__meta", "preview.mp4")
            );
            btn.addEventListener("click", () => close(""));
            list.append(btn);
        }
        const sorted = [...variants].sort((a, b) => b.index - a.index);
        for (const variant of sorted) {
            const btn = createEl("button", "h3lq-picker__item is-hq", "");
            btn.type = "button";
            if (current && variant.name === current) {
                btn.classList.add("is-current");
            }
            btn.append(
                createEl(
                    "div",
                    "h3lq-picker__name",
                    `HQ ${String(variant.index).padStart(3, "0")}`
                ),
                createEl(
                    "div",
                    "h3lq-picker__meta",
                    variant.frame_count ? `${variant.frame_count}f` : variant.name
                )
            );
            btn.addEventListener("click", () => close(variant.name));
            list.append(btn);
        }
        const cancel = createEl("button", "h3lq-btn", "Cancel");
        cancel.type = "button";
        cancel.addEventListener("click", () => close(null));
        overlay.append(list, cancel);
        const onKey = (ev) => {
            if (ev.key === "Escape") {
                ev.preventDefault();
                ev.stopPropagation();
                ev.stopImmediatePropagation();
                close(null);
            }
        };
        window.addEventListener("keydown", onKey, true);
        shield.addEventListener("click", () => close(null));
        document.body.append(shield, overlay);
        if (opts.clientX != null && opts.clientY != null) {
            overlay.classList.add("h3lq-picker--menu");
            const rect = overlay.getBoundingClientRect();
            const x = Math.min(opts.clientX, window.innerWidth - rect.width - 8);
            const y = Math.min(opts.clientY, window.innerHeight - rect.height - 8);
            overlay.style.left = `${Math.max(8, x)}px`;
            overlay.style.top = `${Math.max(8, y)}px`;
        }
    });
}

function showCollectModal(node) {
    ensureCss();
    node.__h3lqCloseModal?.();

    const loadNode = connectedLoadNode(node);
    const initialEdit = loadNodeEdit(loadNode);
    const listed = collectBinPackages(loadNode);
    const packageCache = new Map();

    const overlay = createEl("div", "h3lq-overlay");
    const modal = createEl("div", "h3lq-modal");
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Collect Timeline");

    const header = createEl("div", "h3lq-modal__header");
    header.append(createEl("h2", "h3lq-modal__title", "Collect Timeline"));
    const refreshBtn = createEl("button", "h3lq-btn", "Refresh HQ files");
    refreshBtn.type = "button";
    const closeBtn = createEl("button", "h3lq-btn", "Close");
    closeBtn.type = "button";
    header.append(refreshBtn, closeBtn);
    const status = createEl(
        "div",
        "h3lq-status",
        loadNode
            ? "Same timeline as Load Package. Blue = LQ, gold = HQ. Right-click a clip to swap versions."
            : "Connect Load Package → package_pipe"
    );

    const timelineBody = createEl("div", "h3lq-body h3lq-body--timeline");
    let hoverVideo = null;
    let closed = false;

    function destroyHover() {
        if (hoverVideo) {
            hoverVideo.pause?.();
            hoverVideo.removeAttribute("src");
            hoverVideo.load?.();
            hoverVideo.remove();
            hoverVideo = null;
        }
    }

    function showHover(card, pkg, ev) {
        destroyHover();
        const info = packageCache.get(pkg);
        const url = info?.upscaled?.at(-1)?.url || info?.preview_url;
        if (!url) {
            return;
        }
        const video = document.createElement("video");
        video.className = "h3lq-hover-video";
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.autoplay = true;
        video.src = url;
        document.body.append(video);
        hoverVideo = video;
        const move = (e) => {
            const x = Math.min(e.clientX + 16, window.innerWidth - video.offsetWidth - 8);
            const y = Math.min(e.clientY + 16, window.innerHeight - video.offsetHeight - 8);
            video.style.left = `${Math.max(8, x)}px`;
            video.style.top = `${Math.max(8, y)}px`;
        };
        move(ev);
        card.addEventListener("mousemove", move);
        video.play?.().catch(() => {});
    }

    const timeline = createTimelineTab({
        hqMode: true,
        initialEdit,
        binLabel: "Clip bin (from Load timeline)",
        binEmptyText: loadNode
            ? "This is the Load Package timeline. Add clips there, or drop packages from this bin."
            : "Connect Load Package → package_pipe to sync its timeline.",
        getListed: () => listed,
        getPackageInfo: (pkg) => packageCache.get(pkg) || null,
        showHover,
        destroyHover,
        postJson,
        pickClipVersion: async (item, ev) => {
            const variants = packageCache.get(item.package)?.upscaled || [];
            return pickVariant(item.package, variants, {
                includeLq: true,
                current: item.media || "",
                clientX: ev?.clientX,
                clientY: ev?.clientY,
            });
        },
        onEditChange: (edit) => {
            if (loadNode) {
                setWidgetValue(loadNode, "edit_json", JSON.stringify(edit));
            }
        },
    });
    timelineBody.append(timeline.root);
    modal.append(header, status, timelineBody);
    overlay.append(modal);
    document.body.append(overlay);

    function close() {
        if (closed) {
            return;
        }
        closed = true;
        destroyHover();
        timeline.destroy();
        overlay.remove();
        node.__h3lqCloseModal = null;
        window.removeEventListener("keydown", onKey);
    }
    node.__h3lqCloseModal = close;
    closeBtn.addEventListener("click", () => close());
    overlay.addEventListener("click", (ev) => {
        if (ev.target === overlay) {
            close();
        }
    });
    function onKey(ev) {
        if (ev.key === "Escape" && !document.querySelector(".h3lq-picker")) {
            close();
        }
    }
    window.addEventListener("keydown", onKey);

    async function refreshVariants() {
        status.textContent = listed.length ? "Loading HQ variants…" : status.textContent;
        await Promise.all(
            listed.map(async (pkg) => {
                try {
                    const data = await getJson(`/h3_lq_stash/variants?package=${encodeURIComponent(pkg)}`);
                    packageCache.set(pkg, {
                        package: pkg,
                        name: data.name || pkg.split("/").pop(),
                        has_preview: Boolean(data.has_preview || (data.upscaled || []).length),
                        has_thumb: Boolean(data.has_thumb),
                        thumb_url: data.thumb_url,
                        preview_url: data.preview_url,
                        frame_count: data.frame_count,
                        fps: data.fps,
                        upscaled: data.upscaled || [],
                    });
                } catch (err) {
                    packageCache.set(pkg, { package: pkg, upscaled: [], has_preview: false });
                }
            })
        );
        const hqCount = listed.filter((pkg) => (packageCache.get(pkg)?.upscaled || []).length).length;
        status.textContent = listed.length
            ? `${hqCount}/${listed.length} packages have HQ renders. Right-click clips to swap LQ/HQ.`
            : "Connect Load Package → package_pipe";
        timeline.refresh();
    }
    refreshBtn.addEventListener("click", () => refreshVariants());
    refreshVariants();
}

function setupCollectNode(node) {
    ensureCss();
    hideMultilineWidget(node, "hq_edit_json");

    const wrap = document.createElement("div");
    wrap.className = "h3lq-collect-ui";
    const previewWrap = createEl("div", "h3lq-collect-preview");
    const video = document.createElement("video");
    video.className = "h3lq-collect-preview__video";
    video.controls = false;
    video.loop = true;
    video.muted = true;
    video.autoplay = true;
    video.playsInline = true;
    video.preload = "metadata";
    const placeholder = createEl(
        "div",
        "h3lq-collect-preview__placeholder",
        "Queue Collect to preview the HQ clip"
    );
    const badge = createEl("div", "h3lq-collect-badge", "");
    badge.hidden = true;
    const openBtn = createEl("button", "h3lq-btn h3lq-btn--tiny h3lq-collect-open", "Open Timeline");
    openBtn.type = "button";
    openBtn.title = "Opens the Load Package timeline. Right-click clips to swap LQ / HQ versions.";
    openBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    openBtn.onclick = (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        showCollectModal(node);
    };
    video.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (video.paused) {
            video.play?.().catch(() => {});
        } else {
            video.pause();
        }
    });
    video.addEventListener("error", () => {
        previewWrap.classList.remove("has-video");
        placeholder.textContent = "Inline preview failed. Queue Collect again.";
        badge.hidden = true;
    });
    previewWrap.addEventListener("pointerenter", () => {
        node.__h3lqCollectHovering = true;
        syncCollectMute(node);
    });
    previewWrap.addEventListener("pointerleave", () => {
        node.__h3lqCollectHovering = false;
        syncCollectMute(node);
    });
    previewWrap.append(video, placeholder, badge, openBtn);
    wrap.append(previewWrap);
    node.__h3lqCollectPreview = previewWrap;
    node.__h3lqCollectVideo = video;
    node.__h3lqCollectBadge = badge;
    wrap.addEventListener(
        "wheel",
        (e) => {
            const cv = app.canvas?.canvas;
            if (!cv) {
                return;
            }
            e.preventDefault();
            cv.dispatchEvent(
                new WheelEvent("wheel", {
                    deltaX: e.deltaX,
                    deltaY: e.deltaY,
                    deltaZ: e.deltaZ,
                    deltaMode: e.deltaMode,
                    clientX: e.clientX,
                    clientY: e.clientY,
                    bubbles: true,
                    cancelable: true,
                })
            );
        },
        { passive: false }
    );
    const widget = node.addDOMWidget(COLLECT_WIDGET_NAME, COLLECT_WIDGET_NAME, wrap, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => collectPreviewWidgetSize(node, node.size?.[0])[1],
        getValue() {
            return "";
        },
        setValue() {},
    });
    widget.computeSize = function (width) {
        return collectPreviewWidgetSize(node, width);
    };
    node.__h3lqCollectWidget = widget;
    if ((node.size?.[0] || 0) < COLLECT_MIN_W || (node.size?.[1] || 0) < COLLECT_DEFAULT_H) {
        node.setSize?.([
            Math.max(node.size?.[0] || 0, COLLECT_MIN_W),
            Math.max(node.size?.[1] || 0, COLLECT_DEFAULT_H),
        ]);
    }

    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        const result = onExecuted?.apply(this, arguments);
        const list = output?.h3_collect_preview || output?.gifs;
        const meta = Array.isArray(list) ? list[0] : null;
        applyCollectPreview(this, meta);
        return result;
    };

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        hideMultilineWidget(this, "hq_edit_json");
        const saved = this.properties?.h3CollectPreview;
        if (saved?.filename) {
            applyCollectPreview(this, saved);
        }
        return result;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        this.__h3lqCloseModal?.();
        const preview = this.__h3lqCollectVideo;
        if (preview) {
            preview.pause?.();
            preview.removeAttribute("src");
            preview.load?.();
        }
        return onRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "ComfyUI.MiniMaxH3.PackageBrowser",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === SAVE_NODE_NAME) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                removeControlWidget(this, "seed");
                return result;
            };
            return;
        }
        if (nodeData?.name === COLLECT_NODE_NAME) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                setupCollectNode(this);
                return result;
            };
            return;
        }
        if (nodeData?.name !== NODE_NAME) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return result;
        };
    },
});
