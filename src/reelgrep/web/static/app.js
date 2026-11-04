// reelgrep web ui - vanilla js single-page app
// loads videos from /api, renders tabs, lightbox, global subtitle search.

const FRAMES_PAGE_SIZE = 50;
const GLOBAL_SEARCH_CONCURRENCY = 4;

const state = {
  version: "?",
  videos: [],
  currentVideoHash: null,
  currentVideo: null,
  currentTab: "overview",
  framesPage: 0,
  framesTotal: 0,
  framesCache: [],
  subtitleQuery: "",
  subtitleCues: [],
  selectedCueId: null,
  exportsKind: "",
  exportsCache: [],
  searchesCache: [],
  showFullPath: false,
  facesFilter: "all",
  facesClusters: [],
  facesCurrentClusterId: null,
};

// ---------- helpers ----------

function $(id) {
  return document.getElementById(id);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") {
      for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
    } else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v === true) node.setAttribute(k, "");
    else if (v === false || v == null) {
      // skip
    } else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function basename(p) {
  if (!p) return "";
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i >= 0 ? p.slice(i + 1) : p;
}

function truncateMiddle(s, max = 70) {
  if (!s || s.length <= max) return s || "";
  const head = Math.ceil((max - 3) / 2);
  const tail = Math.floor((max - 3) / 2);
  return s.slice(0, head) + "..." + s.slice(s.length - tail);
}

function formatMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "--:--:--.---";
  const total = Math.max(0, Math.floor(ms));
  const h = Math.floor(total / 3600000);
  const m = Math.floor((total % 3600000) / 60000);
  const s = Math.floor((total % 60000) / 1000);
  const mmm = total % 1000;
  return (
    String(h).padStart(2, "0") +
    ":" +
    String(m).padStart(2, "0") +
    ":" +
    String(s).padStart(2, "0") +
    "." +
    String(mmm).padStart(3, "0")
  );
}

function formatDuration(ms) {
  if (ms == null) return "?";
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatBytes(n) {
  if (n == null) return "?";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + " " + units[i];
}

function formatDate(iso) {
  if (!iso) return "?";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(d);
  } catch (_) {
    return iso;
  }
}

function fileUrl(path) {
  if (!path) return "";
  return "/file?path=" + encodeURIComponent(path);
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightText(text, query) {
  if (!text) return document.createTextNode("");
  if (!query) return document.createTextNode(text);
  const re = new RegExp("(" + escapeRegex(query) + ")", "ig");
  const parts = text.split(re);
  const frag = document.createDocumentFragment();
  for (const p of parts) {
    if (re.test(p) && p.toLowerCase() === query.toLowerCase()) {
      frag.appendChild(el("mark", {}, [p]));
    } else {
      frag.appendChild(document.createTextNode(p));
    }
    re.lastIndex = 0;
  }
  return frag;
}

let statusTimer = null;
function setStatus(msg, kind) {
  const bar = $("statusbar");
  if (!bar) return;
  bar.textContent = msg || "";
  bar.classList.remove("error", "ok");
  if (kind === "error") bar.classList.add("error");
  if (kind === "ok") bar.classList.add("ok");
  if (msg) bar.classList.add("visible");
  else bar.classList.remove("visible");
  if (statusTimer) clearTimeout(statusTimer);
  if (msg) {
    statusTimer = setTimeout(() => bar.classList.remove("visible"), kind === "error" ? 6000 : 3000);
  }
}

// ---------- api ----------

async function api(path, params) {
  let url = path;
  if (params) {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") sp.set(k, v);
    }
    const qs = sp.toString();
    if (qs) url += (url.includes("?") ? "&" : "?") + qs;
  }
  try {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} ${body.slice(0, 200)}`);
    }
    return await res.json();
  } catch (err) {
    setStatus("API error: " + err.message, "error");
    throw err;
  }
}

// ---------- bootstrap ----------

async function init() {
  bindGlobalUI();
  try {
    const health = await api("/api/health");
    state.version = health.version || "?";
    $("version").textContent = "v" + state.version;
  } catch (_) {
    // already surfaced
  }
  await loadVideos();
}

async function loadVideos() {
  try {
    const data = await api("/api/videos");
    state.videos = data.videos || [];
    renderVideoList();
  } catch (_) {
    const list = $("video-list");
    clear(list);
    list.appendChild(el("li", { class: "video-list-empty" }, ["Failed to load videos."]));
  }
}

// ---------- sidebar ----------

function renderVideoList(matchMap) {
  const list = $("video-list");
  clear(list);
  if (!state.videos.length) {
    list.appendChild(el("li", { class: "video-list-empty" }, ["No videos ingested."]));
    return;
  }
  for (const v of state.videos) {
    const isActive = v.file_hash === state.currentVideoHash;
    const card = el("li", {
      class: "video-card" + (isActive ? " active" : ""),
      dataset: { hash: v.file_hash },
      onclick: () => selectVideo(v.file_hash),
    });
    card.appendChild(el("div", { class: "video-card-title" }, [basename(v.path) || v.path]));
    card.appendChild(
      el("div", { class: "video-card-sub" }, [
        `${formatDuration(v.duration_ms)} - ${v.subtitle_cues_count || 0} cues`,
      ]),
    );
    if (matchMap && matchMap[v.file_hash] != null) {
      const n = matchMap[v.file_hash];
      const label = n === -1 ? "search failed" : `${n} matches`;
      card.appendChild(el("div", { class: "video-card-matches" }, [label]));
    }
    list.appendChild(card);
  }
}

async function sidebarSearchAll(query) {
  const status = $("sidebar-search-status");
  if (!query) {
    status.textContent = "";
    renderVideoList();
    return;
  }
  status.textContent = `searching ${state.videos.length} videos...`;
  const matchMap = {};
  await concurrentFanout(
    state.videos,
    GLOBAL_SEARCH_CONCURRENCY,
    async (v) => {
      try {
        const r = await api(`/api/videos/${encodeURIComponent(v.file_hash)}/subtitles`, {
          q: query,
          limit: 1,
        });
        matchMap[v.file_hash] = r.total ?? (r.cues ? r.cues.length : 0);
      } catch (_) {
        matchMap[v.file_hash] = -1;
      }
      renderVideoList(matchMap);
    },
  );
  const hits = Object.values(matchMap).filter((n) => n > 0).length;
  status.textContent = `${hits} videos with matches`;
}

async function concurrentFanout(items, limit, worker) {
  const queue = items.slice();
  const runners = [];
  for (let i = 0; i < Math.min(limit, queue.length); i++) {
    runners.push(
      (async () => {
        while (queue.length) {
          const item = queue.shift();
          await worker(item);
        }
      })(),
    );
  }
  await Promise.all(runners);
}

// ---------- video selection ----------

async function selectVideo(hash, opts = {}) {
  state.currentVideoHash = hash;
  state.framesPage = 0;
  state.framesCache = [];
  state.subtitleCues = [];
  state.selectedCueId = null;
  state.exportsCache = [];
  state.searchesCache = [];
  state.showFullPath = false;
  $("empty-state").classList.add("hidden");
  $("global-results").classList.add("hidden");
  $("faces-view").classList.add("hidden");
  $("video-detail").classList.remove("hidden");
  renderVideoList();
  try {
    const v = await api(`/api/videos/${encodeURIComponent(hash)}`);
    state.currentVideo = v;
    renderVideoHeader();
    const tab = opts.tab || "overview";
    selectTab(tab);
    if (opts.subtitleQuery) {
      $("subtitle-search-input").value = opts.subtitleQuery;
      state.subtitleQuery = opts.subtitleQuery;
      if (tab === "subtitles") loadSubtitles();
    }
  } catch (_) {
    // surfaced
  }
}

function renderVideoHeader() {
  const v = state.currentVideo;
  if (!v) return;
  $("video-title").textContent = basename(v.path) || v.path;
  const display = $("video-path-display");
  display.textContent = state.showFullPath ? v.path : truncateMiddle(v.path, 80);
  $("video-path-toggle").textContent = state.showFullPath ? "hide full path" : "show full path";

  const meta = $("video-meta");
  clear(meta);
  const parts = [
    `hash: ${v.file_hash.slice(0, 12)}...`,
    formatDuration(v.duration_ms),
    `${v.width || "?"}x${v.height || "?"}`,
    `@ ${v.fps ? v.fps.toFixed ? v.fps.toFixed(2) : v.fps : "?"} fps`,
    v.container || "",
    v.video_codec || "",
    v.audio_codec || "",
    v.size_bytes ? formatBytes(v.size_bytes) : "",
    `ingested ${formatDate(v.ingested_at)}`,
  ].filter(Boolean);
  meta.textContent = parts.join("  -  ");

  const counts = $("video-counts");
  counts.textContent =
    `${v.subtitle_cues_count || 0} cues  -  ` +
    `${v.frames_count || 0} frames  -  ` +
    `${v.person_searches_count || 0} searches  -  ` +
    `${v.exports_count || 0} exports`;
}

$("video-path-toggle")?.addEventListener("click", () => {
  state.showFullPath = !state.showFullPath;
  renderVideoHeader();
});

// ---------- tabs ----------

function selectTab(tab) {
  state.currentTab = tab;
  for (const btn of document.querySelectorAll(".tab")) {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    panel.classList.add("hidden");
  }
  const panel = document.getElementById("tab-" + tab);
  if (panel) panel.classList.remove("hidden");
  if (tab === "overview") renderOverview();
  if (tab === "frames") loadFrames();
  if (tab === "subtitles") loadSubtitles();
  if (tab === "searches") loadSearches();
  if (tab === "exports") loadExports();
}

// ---------- overview ----------

function renderOverview() {
  const v = state.currentVideo;
  const panel = $("tab-overview");
  clear(panel);
  if (!v) return;
  const summary = el("div", { class: "overview-summary" }, [
    `${basename(v.path)} runs ${formatDuration(v.duration_ms)} at ${v.width || "?"}x${v.height || "?"}. `,
    `${v.subtitle_cues_count || 0} subtitle cues, ${v.frames_count || 0} sampled frames, `,
    `${v.person_searches_count || 0} person searches, ${v.exports_count || 0} exports.`,
  ]);
  panel.appendChild(summary);

  if (v.exports_by_kind && Object.keys(v.exports_by_kind).length) {
    panel.appendChild(el("div", { class: "section-title" }, ["EXPORTS BY KIND"]));
    const wrap = el("div", {});
    for (const [k, n] of Object.entries(v.exports_by_kind)) {
      wrap.appendChild(el("span", { class: "chip chip-static" }, [`${k}: ${n}`]));
    }
    panel.appendChild(wrap);
  }

  panel.appendChild(el("div", { class: "section-title" }, ["PERSON SEARCHES (first 5)"]));
  const searchesWrap = el("div", {});
  panel.appendChild(searchesWrap);
  api("/api/searches")
    .then((data) => {
      const mine = (data.searches || [])
        .filter((s) => s.video_hash === v.file_hash)
        .slice(0, 5);
      if (!mine.length) {
        searchesWrap.appendChild(el("div", { class: "video-meta" }, ["no person searches yet"]));
        return;
      }
      for (const s of mine) {
        const row = el("div", { class: "video-meta" }, [
          `${s.label} - ${s.match_count || 0} matches - threshold ${s.threshold ?? "?"}`,
        ]);
        searchesWrap.appendChild(row);
      }
    })
    .catch(() => {
      /* surfaced */
    });

  panel.appendChild(el("div", { class: "section-title" }, ["ACTIONS"]));
  const actions = el("div", {});
  actions.appendChild(
    el(
      "button",
      {
        type: "button",
        onclick: () => {
          window.open("file://" + v.path, "_blank");
        },
      },
      ["Open in player"],
    ),
  );
  panel.appendChild(actions);
}

// ---------- frames ----------

async function loadFrames() {
  const v = state.currentVideo;
  if (!v) return;
  const grid = $("frames-grid");
  clear(grid);
  grid.appendChild(el("div", { class: "video-meta" }, ["loading frames..."]));
  try {
    const data = await api(`/api/videos/${encodeURIComponent(v.file_hash)}/frames`, {
      limit: FRAMES_PAGE_SIZE,
      offset: state.framesPage * FRAMES_PAGE_SIZE,
    });
    state.framesCache = data.frames || [];
    state.framesTotal = data.total ?? state.framesCache.length;
    renderFrames();
  } catch (_) {
    clear(grid);
    grid.appendChild(el("div", { class: "video-meta" }, ["failed to load frames"]));
  }
}

function renderFrames() {
  const grid = $("frames-grid");
  clear(grid);
  if (!state.framesCache.length) {
    grid.appendChild(el("div", { class: "video-meta" }, ["no frames on this page"]));
  }
  for (const f of state.framesCache) {
    const tile = el(
      "div",
      {
        class: "frame-tile",
        onclick: () => openLightbox(f.path, f.timestamp_ms),
      },
      [
        el("img", { src: fileUrl(f.path), alt: "frame", loading: "lazy" }),
        el("div", { class: "frame-tile-timestamp" }, [formatMs(f.timestamp_ms)]),
      ],
    );
    grid.appendChild(tile);
  }
  const totalPages = Math.max(1, Math.ceil(state.framesTotal / FRAMES_PAGE_SIZE));
  $("frames-pageinfo").textContent =
    `page ${state.framesPage + 1} / ${totalPages} (${state.framesTotal} total)`;
  $("frames-prev").disabled = state.framesPage <= 0;
  $("frames-next").disabled = state.framesPage + 1 >= totalPages;
}

$("frames-prev")?.addEventListener("click", () => {
  if (state.framesPage > 0) {
    state.framesPage--;
    loadFrames();
  }
});
$("frames-next")?.addEventListener("click", () => {
  state.framesPage++;
  loadFrames();
});

// ---------- subtitles ----------

async function loadSubtitles() {
  const v = state.currentVideo;
  if (!v) return;
  const results = $("subtitle-results");
  clear(results);
  results.appendChild(el("div", { class: "video-meta" }, ["loading..."]));
  try {
    const data = await api(`/api/videos/${encodeURIComponent(v.file_hash)}/subtitles`, {
      q: state.subtitleQuery || "",
      limit: 500,
    });
    state.subtitleCues = data.cues || [];
    renderSubtitleResults();
  } catch (_) {
    clear(results);
    results.appendChild(el("div", { class: "video-meta" }, ["failed to load subtitles"]));
  }
}

function renderSubtitleResults() {
  const results = $("subtitle-results");
  clear(results);
  if (!state.subtitleCues.length) {
    results.appendChild(
      el("div", { class: "video-meta" }, [
        state.subtitleQuery ? "no matches" : "no cues for this video",
      ]),
    );
    $("subtitle-preview").classList.add("hidden");
    return;
  }
  for (const c of state.subtitleCues) {
    const row = el(
      "div",
      {
        class: "cue-row" + (c.id === state.selectedCueId ? " selected" : ""),
        dataset: { id: String(c.id) },
        onclick: () => selectCue(c.id),
      },
      [
        el("div", { class: "cue-time" }, [formatMs(c.start_ms)]),
        el("div", { class: "cue-text" }, [highlightText(c.text || "", state.subtitleQuery)]),
        el("div", { class: "cue-source" }, [c.source || ""]),
      ],
    );
    results.appendChild(row);
  }
}

async function selectCue(id) {
  state.selectedCueId = id;
  renderSubtitleResults();
  const cue = state.subtitleCues.find((c) => c.id === id);
  if (!cue) return;
  const preview = $("subtitle-preview");
  clear(preview);
  preview.classList.remove("hidden");
  preview.appendChild(el("div", { class: "video-meta" }, ["finding nearest frame..."]));
  try {
    const v = state.currentVideo;
    const data = await api(`/api/videos/${encodeURIComponent(v.file_hash)}/frames`, {
      limit: 5000,
      offset: 0,
    });
    const frames = data.frames || [];
    if (!frames.length) {
      clear(preview);
      preview.appendChild(el("div", { class: "video-meta" }, ["no frames available"]));
      return;
    }
    let nearest = frames[0];
    let bestDist = Math.abs(nearest.timestamp_ms - cue.start_ms);
    for (const f of frames) {
      const d = Math.abs(f.timestamp_ms - cue.start_ms);
      if (d < bestDist) {
        nearest = f;
        bestDist = d;
      }
    }
    clear(preview);
    preview.appendChild(
      el("img", {
        src: fileUrl(nearest.path),
        alt: "nearest frame",
        onclick: () => openLightbox(nearest.path, nearest.timestamp_ms),
      }),
    );
    preview.appendChild(
      el("div", { class: "subtitle-preview-caption" }, [
        `cue @ ${formatMs(cue.start_ms)} - nearest frame @ ${formatMs(nearest.timestamp_ms)} (delta ${bestDist}ms)`,
      ]),
    );
  } catch (_) {
    clear(preview);
    preview.appendChild(el("div", { class: "video-meta" }, ["failed to load nearest frame"]));
  }
}

$("subtitle-search-form")?.addEventListener("submit", (e) => {
  e.preventDefault();
  state.subtitleQuery = $("subtitle-search-input").value.trim();
  state.selectedCueId = null;
  $("subtitle-preview").classList.add("hidden");
  loadSubtitles();
});
$("subtitle-clear")?.addEventListener("click", () => {
  $("subtitle-search-input").value = "";
  state.subtitleQuery = "";
  state.selectedCueId = null;
  $("subtitle-preview").classList.add("hidden");
  loadSubtitles();
});

// ---------- searches ----------

async function loadSearches() {
  const v = state.currentVideo;
  if (!v) return;
  const panel = $("tab-searches");
  clear(panel);
  panel.appendChild(el("div", { class: "video-meta" }, ["loading searches..."]));
  try {
    const data = await api("/api/searches");
    state.searchesCache = (data.searches || []).filter((s) => s.video_hash === v.file_hash);
    renderSearches();
  } catch (_) {
    clear(panel);
    panel.appendChild(el("div", { class: "video-meta" }, ["failed to load searches"]));
  }
}

function renderSearches() {
  const panel = $("tab-searches");
  clear(panel);
  if (!state.searchesCache.length) {
    panel.appendChild(el("div", { class: "video-meta" }, ["no person searches for this video"]));
    return;
  }
  for (const s of state.searchesCache) {
    const card = el("div", { class: "search-card" });
    const header = el(
      "div",
      {
        class: "search-card-header",
        onclick: () => toggleSearchDetail(s.id, card),
      },
      [
        el("span", { class: "search-card-label" }, [s.label || `search ${s.id}`]),
        el("span", { class: "search-card-meta" }, [
          `${s.match_count || 0} matches - backend ${s.backend || "?"} - threshold ${s.threshold ?? "?"} - ${formatDate(s.created_at)}`,
        ]),
      ],
    );
    card.appendChild(header);
    panel.appendChild(card);
  }
}

async function toggleSearchDetail(id, card) {
  const existing = card.querySelector(".search-card-body");
  if (existing) {
    existing.remove();
    return;
  }
  const body = el("div", { class: "search-card-body" }, [
    el("div", { class: "video-meta" }, ["loading matches..."]),
  ]);
  card.appendChild(body);
  try {
    const data = await api(`/api/searches/${encodeURIComponent(id)}`);
    clear(body);
    const matches = (data.matches || [])
      .slice()
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
    if (data.positive_examples || data.negative_examples) {
      body.appendChild(
        el("div", { class: "video-meta" }, [
          `positives: ${(data.positive_examples || []).length} - negatives: ${(data.negative_examples || []).length}`,
        ]),
      );
    }
    if (!matches.length) {
      body.appendChild(el("div", { class: "video-meta" }, ["no matches"]));
      return;
    }
    const grid = el("div", { class: "match-grid" });
    for (const m of matches) {
      const tile = el("div", { class: "match-tile" });
      tile.appendChild(
        el("img", {
          src: fileUrl(m.frame_path),
          alt: "match",
          loading: "lazy",
          onclick: () => openLightbox(m.frame_path, m.frame_timestamp_ms),
        }),
      );
      const meta = el("div", { class: "match-tile-meta" });
      meta.appendChild(
        el("div", {}, [
          el("span", { class: "match-confidence" }, [
            "confidence: " + (m.confidence != null ? m.confidence.toFixed(3) : "?"),
          ]),
        ]),
      );
      meta.appendChild(el("div", {}, ["t=" + formatMs(m.frame_timestamp_ms)]));
      if (m.bbox) {
        const b = m.bbox;
        let bboxText = "";
        if (Array.isArray(b)) bboxText = "[" + b.map((n) => Number(n).toFixed(2)).join(", ") + "]";
        else bboxText = JSON.stringify(b);
        meta.appendChild(el("div", {}, ["bbox: " + bboxText]));
      }
      if (m.reasoning) {
        meta.appendChild(el("div", {}, [m.reasoning]));
      }
      tile.appendChild(meta);
      grid.appendChild(tile);
    }
    body.appendChild(grid);
  } catch (_) {
    clear(body);
    body.appendChild(el("div", { class: "video-meta" }, ["failed to load matches"]));
  }
}

// ---------- exports ----------

async function loadExports() {
  const v = state.currentVideo;
  if (!v) return;
  const list = $("export-list");
  clear(list);
  list.appendChild(el("li", { class: "video-meta" }, ["loading..."]));
  try {
    const data = await api("/api/exports", { kind: state.exportsKind || "", limit: 500 });
    state.exportsCache = (data.exports || []).filter((x) => x.video_hash === v.file_hash);
    renderExports();
  } catch (_) {
    clear(list);
    list.appendChild(el("li", { class: "video-meta" }, ["failed to load exports"]));
  }
}

function renderExports() {
  const list = $("export-list");
  clear(list);
  if (!state.exportsCache.length) {
    list.appendChild(el("li", { class: "video-meta" }, ["no exports"]));
    return;
  }
  for (const x of state.exportsCache) {
    const range = x.start_ms != null || x.end_ms != null
      ? `${formatMs(x.start_ms || 0)} -> ${formatMs(x.end_ms || 0)}`
      : "";
    const row = el("li", { class: "export-row" });
    row.appendChild(el("span", { class: "export-kind" }, [x.kind || "?"]));
    const nameLink = el("a", { href: fileUrl(x.path), target: "_blank", rel: "noopener" }, [
      basename(x.path),
    ]);
    row.appendChild(nameLink);
    row.appendChild(el("span", { class: "cue-time" }, [range]));
    const right = el("span", {});
    right.appendChild(document.createTextNode(formatDate(x.created_at)));
    if (x.manifest_path) {
      right.appendChild(document.createTextNode("  "));
      right.appendChild(
        el(
          "a",
          {
            href: fileUrl(x.manifest_path),
            target: "_blank",
            rel: "noopener",
            class: "link-button",
          },
          ["[manifest]"],
        ),
      );
    }
    row.appendChild(right);
    list.appendChild(row);
  }
}

document.querySelectorAll("#export-filters .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    state.exportsKind = chip.dataset.kind || "";
    document
      .querySelectorAll("#export-filters .chip")
      .forEach((c) => c.classList.toggle("active", c === chip));
    loadExports();
  });
});

// ---------- faces ----------

function showFaces() {
  $("video-detail").classList.add("hidden");
  $("global-results").classList.add("hidden");
  $("empty-state").classList.add("hidden");
  $("faces-view").classList.remove("hidden");
  hideFacesDetail();
  loadFaces();
}

function hideFaces() {
  $("faces-view").classList.add("hidden");
  if (state.currentVideoHash) {
    $("video-detail").classList.remove("hidden");
  } else {
    $("empty-state").classList.remove("hidden");
  }
}

function hideFacesDetail() {
  const detail = $("faces-detail");
  const grid = $("faces-grid");
  if (detail) detail.classList.add("hidden");
  if (grid) grid.classList.remove("hidden");
  state.facesCurrentClusterId = null;
  const errEl = $("faces-detail-error");
  if (errEl) {
    errEl.textContent = "";
    errEl.classList.add("hidden");
  }
}

async function loadFaces() {
  const grid = $("faces-grid");
  if (!grid) return;
  clear(grid);
  grid.appendChild(el("div", { class: "video-meta" }, ["loading clusters..."]));
  const filter = state.facesFilter || "all";
  const params = filter === "labeled" ? { labeled_only: "true" } : null;
  try {
    const data = await api("/api/faces/clusters", params);
    state.facesClusters = data.clusters || [];
    renderFacesGrid();
  } catch (_) {
    clear(grid);
    grid.appendChild(el("div", { class: "video-meta" }, ["failed to load clusters"]));
  }
}

function renderFacesGrid() {
  const grid = $("faces-grid");
  clear(grid);
  const filter = state.facesFilter || "all";
  const filtered = state.facesClusters.filter((c) => {
    if (filter === "unlabeled") return !c.label;
    return true;
  });
  if (!filtered.length) {
    const empty = el("div", { class: "video-meta" }, [
      "no clusters yet. run ",
      el("code", {}, ["reelgrep faces cluster"]),
      " to build them.",
    ]);
    grid.appendChild(empty);
    return;
  }
  for (const c of filtered) {
    const card = el(
      "button",
      {
        type: "button",
        class: "cluster-card",
        onclick: () => openClusterDetail(c.id),
      },
      [
        el("div", { class: "cluster-id" }, [`#${c.id}`]),
        c.label
          ? el("div", { class: "cluster-label" }, [c.label])
          : el("div", { class: "cluster-label cluster-label-empty" }, ["unlabeled"]),
        el("div", { class: "cluster-size" }, [`${c.size || 0} faces`]),
      ],
    );
    grid.appendChild(card);
  }
}

async function openClusterDetail(clusterId) {
  const grid = $("faces-grid");
  const detail = $("faces-detail");
  const title = $("faces-detail-title");
  const labelInput = $("faces-detail-label");
  const errEl = $("faces-detail-error");
  const memberGrid = $("faces-detail-members");
  if (!detail || !title || !labelInput || !memberGrid) return;
  errEl.textContent = "";
  errEl.classList.add("hidden");
  state.facesCurrentClusterId = clusterId;
  clear(memberGrid);
  memberGrid.appendChild(el("div", { class: "video-meta" }, ["loading cluster..."]));
  detail.classList.remove("hidden");
  grid.classList.add("hidden");
  try {
    const data = await api(`/api/faces/clusters/${encodeURIComponent(clusterId)}`);
    const cluster = data.cluster || {};
    title.textContent = `Cluster #${cluster.id}  -  ${cluster.size || 0} faces`;
    labelInput.value = cluster.label || "";
    clear(memberGrid);
    const members = data.members || [];
    if (!members.length) {
      memberGrid.appendChild(el("div", { class: "video-meta" }, ["no members"]));
      return;
    }
    for (const m of members) {
      const tile = el("div", { class: "member-card" });
      tile.appendChild(
        el("img", {
          src: fileUrl(m.frame_path),
          alt: "face frame",
          loading: "lazy",
          onclick: () => openLightbox(m.frame_path, m.timestamp_ms),
        }),
      );
      const meta = el("div", { class: "member-meta" });
      meta.appendChild(
        el("div", { class: "member-meta-path" }, [
          truncateMiddle(m.video_path || "", 50),
        ]),
      );
      meta.appendChild(
        el("div", { class: "cue-time" }, [formatMs(m.timestamp_ms || 0)]),
      );
      tile.appendChild(meta);
      memberGrid.appendChild(tile);
    }
  } catch (_) {
    clear(memberGrid);
    memberGrid.appendChild(el("div", { class: "video-meta" }, ["failed to load cluster"]));
  }
}

async function saveClusterLabel() {
  const clusterId = state.facesCurrentClusterId;
  if (clusterId == null) return;
  const labelInput = $("faces-detail-label");
  const errEl = $("faces-detail-error");
  if (!labelInput || !errEl) return;
  errEl.textContent = "";
  errEl.classList.add("hidden");
  const trimmed = (labelInput.value || "").trim();
  const body = JSON.stringify({ label: trimmed === "" ? null : trimmed });
  try {
    const res = await fetch(`/api/faces/clusters/${encodeURIComponent(clusterId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const errBody = await res.json();
        detail = errBody.detail || detail;
      } catch (_) {
        /* keep statusText */
      }
      errEl.textContent = String(detail);
      errEl.classList.remove("hidden");
      return;
    }
    setStatus("label saved", "ok");
    await loadFaces();
    hideFacesDetail();
  } catch (err) {
    errEl.textContent = "save failed: " + (err && err.message ? err.message : "network error");
    errEl.classList.remove("hidden");
  }
}

// ---------- global subtitle search ----------

async function runGlobalSearch(query) {
  if (!query) return;
  const view = $("global-results");
  const body = $("global-results-body");
  $("video-detail").classList.add("hidden");
  $("empty-state").classList.add("hidden");
  $("faces-view").classList.add("hidden");
  view.classList.remove("hidden");
  clear(body);
  body.appendChild(el("div", { class: "video-meta" }, [`searching ${state.videos.length} videos for "${query}"...`]));
  const buckets = [];
  await concurrentFanout(
    state.videos,
    GLOBAL_SEARCH_CONCURRENCY,
    async (v) => {
      try {
        const r = await api(`/api/videos/${encodeURIComponent(v.file_hash)}/subtitles`, {
          q: query,
          limit: 50,
        });
        const total = r.total ?? (r.cues ? r.cues.length : 0);
        if (total > 0) {
          buckets.push({ video: v, total, cues: r.cues || [] });
          renderGlobalResults(buckets, query);
        }
      } catch (_) {
        /* skip */
      }
    },
  );
  if (!buckets.length) {
    clear(body);
    body.appendChild(el("div", { class: "video-meta" }, ["no matches across any video"]));
  }
}

function renderGlobalResults(buckets, query) {
  const body = $("global-results-body");
  clear(body);
  const sorted = buckets.slice().sort((a, b) => b.total - a.total);
  for (const b of sorted) {
    const group = el("div", { class: "global-results-group" });
    const title = el(
      "div",
      {
        class: "global-results-group-title",
        onclick: () =>
          selectVideo(b.video.file_hash, { tab: "subtitles", subtitleQuery: query }),
      },
      [basename(b.video.path) || b.video.path],
    );
    title.appendChild(el("span", { class: "global-results-count" }, [`${b.total} matches`]));
    group.appendChild(title);
    for (const c of b.cues.slice(0, 5)) {
      const row = el("div", { class: "global-results-cue" }, [
        el("span", { class: "cue-time" }, [formatMs(c.start_ms)]),
        el("span", {}, [highlightText(c.text || "", query)]),
      ]);
      group.appendChild(row);
    }
    if (b.total > 5) {
      group.appendChild(el("div", { class: "video-meta" }, [`+ ${b.total - 5} more...`]));
    }
    body.appendChild(group);
  }
}

// ---------- lightbox ----------

function openLightbox(path, ts) {
  const lb = $("lightbox");
  $("lightbox-img").src = fileUrl(path);
  $("lightbox-timestamp").textContent = formatMs(ts);
  lb.dataset.timestamp = formatMs(ts);
  lb.classList.remove("hidden");
}

function closeLightbox() {
  $("lightbox").classList.add("hidden");
  $("lightbox-img").src = "";
}

$("lightbox-close")?.addEventListener("click", closeLightbox);
$("lightbox-backdrop")?.addEventListener("click", closeLightbox);
$("lightbox-copy")?.addEventListener("click", async () => {
  const ts = $("lightbox-timestamp").textContent;
  try {
    await navigator.clipboard.writeText(ts);
    setStatus("timestamp copied: " + ts, "ok");
  } catch (_) {
    setStatus("copy failed", "error");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("lightbox").classList.contains("hidden")) {
    closeLightbox();
  }
});

// ---------- global UI bindings ----------

function bindGlobalUI() {
  $("global-search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("global-search-input").value.trim();
    if (q) runGlobalSearch(q);
  });
  $("global-results-close").addEventListener("click", () => {
    $("global-results").classList.add("hidden");
    if (state.currentVideoHash) $("video-detail").classList.remove("hidden");
    else $("empty-state").classList.remove("hidden");
  });
  $("sidebar-search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("sidebar-search-input").value.trim();
    sidebarSearchAll(q);
  });
  for (const btn of document.querySelectorAll(".tab")) {
    btn.addEventListener("click", () => selectTab(btn.dataset.tab));
  }
  for (const btn of document.querySelectorAll(".topbar-nav button[data-quicktab]")) {
    btn.addEventListener("click", () => {
      if (state.currentVideoHash) selectTab(btn.dataset.quicktab);
      else setStatus("pick a video first", "error");
    });
  }
  $("topbar-faces")?.addEventListener("click", showFaces);
  $("faces-view-close")?.addEventListener("click", hideFaces);
  $("faces-refresh")?.addEventListener("click", loadFaces);
  $("faces-filter")?.addEventListener("change", (e) => {
    state.facesFilter = e.target.value || "all";
    loadFaces();
  });
  $("faces-detail-back")?.addEventListener("click", hideFacesDetail);
  $("faces-detail-save")?.addEventListener("click", saveClusterLabel);
  $("faces-detail-label")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      saveClusterLabel();
    }
  });
}

// kick off
init();
