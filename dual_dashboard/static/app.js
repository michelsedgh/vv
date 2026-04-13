const state = {
  ws: null,
  wsPingTimer: null,
  audioCtx: null,
  videoPollTimer: null,
  videoRequestInFlight: false,
  videoObjectUrl: null,
  currentVoiceStep: 0,
  editors: {
    voice: editorState("voice", "/api/voice/config", "/api/voice/config/editor"),
    poguise: editorState("poguise", "/api/poguise/config", "/api/poguise/config/editor"),
    scheduler: editorState("scheduler", "/api/scheduler/config", "/api/scheduler/config/editor"),
  },
  voiceRunning: false,
  poguiseRunning: false,
  shellVisible: {
    voice: true,
    poguise: true,
    scheduler: true,
  },
  lastPoguiseError: "",
  lastPoguiseStatusMessage: "",
  lastPoguiseHistoryKey: "",
};

const speakerLanes = {};
const eventLog = [];
const COLORS = ["#d98d79", "#d7b16d", "#87b4c9", "#a1c289", "#c4a0d8", "#d1a3a9", "#8fb7ab", "#d9c38a"];
const AUDIO_LOOKAHEAD_SEC = 0.12;
const AUDIO_MIN_LEAD_SEC = 0.02;
const AUDIO_MAX_BUFFER_SEC = 0.75;
const AUDIO_PACKET_HEADER_BYTES = 20;
const AUDIO_PACKET_MAGIC = "VAUD";
const UNKNOWN_LANE_STALE_STEPS = 3;
const timeline = {
  steps: [],
  stepDuration: 0.5,
  windowSec: 60,
  knownSpeakers: {},
  canvas: null,
  ctx: null,
};

function editorState(name, path, reloadPath) {
  return {
    name,
    path,
    reloadPath,
    fields: [],
    values: {},
    loadedSnapshot: "",
    hasOverrides: false,
    overrideFile: "",
  };
}

function $(id) {
  return document.getElementById(id);
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function getConfigValue(obj, path) {
  return path.split(".").reduce((cur, key) => (cur == null ? undefined : cur[key]), obj);
}

function setConfigValue(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i];
    if (!cur[key] || typeof cur[key] !== "object" || Array.isArray(cur[key])) {
      cur[key] = {};
    }
    cur = cur[key];
  }
  cur[parts[parts.length - 1]] = value;
}

function snapshotEditor(editor) {
  const flat = {};
  for (const field of editor.fields) {
    flat[field.path] = getConfigValue(editor.values, field.path);
  }
  return JSON.stringify(flat);
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || resp.statusText);
  }
  return data;
}

function setButtonDisabled(id, disabled) {
  const el = $(id);
  if (el) {
    el.disabled = disabled;
  }
}

function syncActionButton(id, running, startLabel, stopLabel) {
  const el = $(id);
  if (!el) return;
  el.textContent = running ? stopLabel : startLabel;
  el.classList.toggle("btn-primary", !running);
  el.classList.toggle("btn-danger", running);
}

function fmtMs(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(0)} ms`;
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(0)}%`;
}

function fmtSeconds(value) {
  if (value == null || Number.isNaN(value)) return "-";
  const sec = Number(value);
  if (sec >= 10) return `${sec.toFixed(0)}s`;
  return `${sec.toFixed(1)}s`;
}

function fmtBytes(value) {
  if (value == null || Number.isNaN(value)) return "-";
  const bytes = Number(value);
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function fmtStamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function normalizeSource(value) {
  if (value === "voice") return "voice";
  if (value === "poguise" || value === "pog") return "poguise";
  return "system";
}

function sourceLabel(value) {
  if (value === "voice") return "Voice";
  if (value === "poguise") return "PO-GUISE";
  return "System";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function logEvent(sourceOrMessage, maybeMessageOrOptions = {}, maybeOptions = undefined) {
  let source = "system";
  let message = sourceOrMessage;
  let options = {};

  if (maybeOptions !== undefined) {
    source = normalizeSource(sourceOrMessage);
    message = maybeMessageOrOptions;
    options = maybeOptions || {};
  } else if (typeof maybeMessageOrOptions === "string") {
    source = normalizeSource(sourceOrMessage);
    message = maybeMessageOrOptions;
  } else if (typeof maybeMessageOrOptions === "boolean") {
    options = { isError: maybeMessageOrOptions };
  } else {
    options = maybeMessageOrOptions || {};
    source = normalizeSource(options.source || "system");
  }

  if (!message) return;
  if (message instanceof Error) {
    message = message.message || String(message);
  }
  message = String(message);
  const meta = options.meta ? String(options.meta) : "";
  const isError = !!options.isError;
  const stamp = new Date().toLocaleTimeString();
  const next = { stamp, source, message, meta, isError };
  const last = eventLog[0];
  if (
    last
    && last.source === next.source
    && last.message === next.message
    && last.meta === next.meta
    && last.isError === next.isError
  ) {
    return;
  }
  eventLog.unshift(next);
  if (eventLog.length > 60) {
    eventLog.pop();
  }
  renderEventLog();
}

function renderEventLog() {
  const root = $("event-log");
  root.innerHTML = "";
  if (!eventLog.length) {
    root.innerHTML = '<div class="event-line">Waiting for runtime events.</div>';
    return;
  }
  for (const item of eventLog) {
    const div = document.createElement("div");
    const sourceClass = item.isError ? "error" : item.source;
    div.className = `event-line source-${sourceClass}${item.isError ? " error" : ""}`;
    div.innerHTML = `
      <span class="event-source ${sourceClass}">${sourceLabel(item.source)}</span>
      <div class="event-body">
        <div class="event-message">${escapeHtml(item.message)}</div>
        <div class="event-meta">${escapeHtml(item.stamp)}${item.meta ? ` • ${escapeHtml(item.meta)}` : ""}</div>
      </div>
    `;
    root.appendChild(div);
  }
}

function renderSystemStatus(system) {
  const row = $("system-status-row");
  const voiceState = system.voice?.running ? "running" : "stopped";
  const pogState = system.poguise?.running ? "running" : "stopped";
  const protectionEnabled = !!system.scheduler?.config?.enabled;
  row.innerHTML = `
    <span class="status-chip" data-state="${voiceState}">Voice <strong>${stateLabel(voiceState)}</strong></span>
    <span class="status-chip" data-state="${pogState}">PO <strong>${stateLabel(pogState)}</strong></span>
    <span class="status-chip" data-state="${protectionEnabled ? "running" : "stopped"}">Protection <strong>${protectionEnabled ? "On" : "Off"}</strong></span>
  `;
}

function stateLabel(value) {
  return value === "running" ? "Live" : value === "warning" ? "Active" : "Idle";
}

function setVoiceToolsVisible(visible) {
  $("voice-tools-panel").classList.toggle("is-hidden", !visible);
}

function applyEditorPayload(editor, payload) {
  editor.fields = payload.fields || [];
  editor.values = deepClone(payload.config || {});
  if (editor.name === "voice") {
    timeline.stepDuration = Number(editor.values?.pixit?.step) || 0.5;
  }
  editor.loadedSnapshot = snapshotEditor(editor);
  editor.hasOverrides = !!payload.has_overrides;
  editor.overrideFile = payload.override_file || "config.overrides";
  renderEditor(editor);
}

function editorIds(name) {
  if (name === "voice") {
    return {
      shell: "voice-settings-shell",
      badge: "voice-config-badge",
      meta: "voice-config-meta",
      grid: "voice-config-fields",
      toggle: "btn-toggle-voice-settings",
    };
  }
  if (name === "poguise") {
    return {
      shell: "pog-settings-shell",
      badge: "pog-config-badge",
      meta: "pog-config-meta",
      grid: "pog-config-fields",
      toggle: "btn-toggle-pog-settings",
    };
  }
  return {
    shell: "scheduler-settings-shell",
    badge: "scheduler-config-badge",
    meta: "scheduler-config-meta",
    grid: "scheduler-config-fields",
    toggle: "btn-toggle-scheduler-settings",
  };
}

function editorRunning(editor) {
  if (editor.name === "voice") return state.voiceRunning;
  if (editor.name === "poguise") return state.poguiseRunning;
  return state.voiceRunning || state.poguiseRunning;
}

function updateEditorBadge(editor) {
  const ids = editorIds(editor.name);
  const badge = $(ids.badge);
  const meta = $(ids.meta);
  const dirty = editor.loadedSnapshot && snapshotEditor(editor) !== editor.loadedSnapshot;
  if (!editor.fields.length) {
    badge.textContent = "Unavailable";
    return;
  }
  if (dirty) {
    badge.textContent = "Unsaved changes";
  } else if (editor.hasOverrides) {
    badge.textContent = "Overrides saved";
  } else {
    badge.textContent = "Defaults";
  }
  if (editor.hasOverrides) {
    meta.textContent = `Saved values live in ${editor.overrideFile}.`;
  } else {
    meta.textContent = `No saved overrides yet.`;
  }
}

function renderEditor(editor) {
  const ids = editorIds(editor.name);
  const grid = $(ids.grid);
  grid.innerHTML = "";
  if (!editor.fields.length) {
    grid.innerHTML = '<div class="field-card">No editable settings metadata loaded.</div>';
    updateEditorBadge(editor);
    return;
  }

  for (const field of editor.fields) {
    const card = document.createElement("article");
    card.className = "field-card";
    const currentValue = getConfigValue(editor.values, field.path);
    card.innerHTML = `
      <div class="field-head">
        <label>${field.label}</label>
        <span class="badge">${field.section}</span>
      </div>
      <div class="field-description">${field.description}</div>
      <div class="field-path">${field.path}</div>
    `;

    let input;
    if (field.type === "boolean") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!currentValue;
    } else if (field.type === "select") {
      input = document.createElement("select");
      for (const optionValue of field.options || []) {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = field.option_labels?.[optionValue] || optionValue;
        if (String(optionValue) === String(currentValue)) option.selected = true;
        input.appendChild(option);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.value = currentValue;
      if (field.min != null) input.min = field.min;
      if (field.max != null) input.max = field.max;
      if (field.step != null) input.step = field.step;
    }

    input.className = "field-input";
    input.disabled = editorRunning(editor);
    input.addEventListener("change", () => {
      let value;
      if (field.type === "boolean") {
        value = input.checked;
      } else if (field.type === "integer") {
        value = parseInt(input.value, 10);
        if (Number.isNaN(value)) return;
      } else if (field.type === "number") {
        value = parseFloat(input.value);
        if (Number.isNaN(value)) return;
      } else {
        value = input.value;
      }
      setConfigValue(editor.values, field.path, value);
      updateEditorBadge(editor);
    });
    card.appendChild(input);
    grid.appendChild(card);
  }
  updateEditorBadge(editor);
}

function refreshEditorInteractivity(editor) {
  const ids = editorIds(editor.name);
  const disabled = editorRunning(editor);
  const shell = $(ids.shell);
  for (const input of shell.querySelectorAll("input, select")) {
    input.disabled = disabled;
  }
  if (editor.name === "voice") {
    setButtonDisabled("btn-save-voice", disabled);
    setButtonDisabled("btn-reset-voice", disabled);
  } else if (editor.name === "poguise") {
    setButtonDisabled("btn-save-pog", disabled);
    setButtonDisabled("btn-reset-pog", disabled);
  } else {
    setButtonDisabled("btn-save-scheduler", disabled);
    setButtonDisabled("btn-reset-scheduler", disabled);
  }
  updateEditorBadge(editor);
}

function collectEditorValues(editor) {
  const values = {};
  for (const field of editor.fields) {
    values[field.path] = getConfigValue(editor.values, field.path);
  }
  return values;
}

async function saveEditor(editor, { silent = false } = {}) {
  if (editorRunning(editor)) {
    logEvent("system", `Stop ${editor.name} before changing its settings.`, { isError: true });
    throw new Error(`Stop ${editor.name} before changing its settings.`);
  }
  const payload = await fetchJson(editor.path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values: collectEditorValues(editor) }),
  });
  applyEditorPayload(editor, payload);
  if (!silent) {
    logEvent("system", `${editor.name} settings saved.`);
  }
}

async function resetEditor(editor) {
  if (editorRunning(editor)) {
    logEvent("system", `Stop ${editor.name} before resetting its settings.`, { isError: true });
    return;
  }
  const payload = await fetchJson(editor.path, { method: "DELETE" });
  applyEditorPayload(editor, payload);
  logEvent("system", `${editor.name} settings reset.`);
}

function setShellVisibility(name, visible) {
  state.shellVisible[name] = visible;
  const ids = editorIds(name);
  $(ids.shell).classList.toggle("is-hidden", !visible);
  $(ids.toggle).textContent = visible ? "Hide Settings" : "Show Settings";
}

function toggleShell(name) {
  setShellVisibility(name, !state.shellVisible[name]);
}

async function ensureAudioReady() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  }
  if (state.audioCtx.state !== "running") {
    try {
      await state.audioCtx.resume();
    } catch (err) {
      logEvent("system", `Audio resume failed: ${err}`, { isError: true });
    }
  }
  for (const lane of Object.values(speakerLanes)) {
    ensureLaneAudioNode(lane);
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  state.ws.binaryType = "arraybuffer";

  state.ws.onopen = () => {
    logEvent("voice", "Voice websocket connected");
    if (state.wsPingTimer) clearInterval(state.wsPingTimer);
    state.wsPingTimer = setInterval(() => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send("ping");
      }
    }, 15000);
  };

  state.ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        handleVoiceMessage(JSON.parse(event.data));
      } catch (err) {
        console.error(err);
      }
      return;
    }
    handleAudioPacket(event.data);
  };

  state.ws.onerror = () => {
    logEvent("voice", "Voice websocket error", { isError: true });
  };

  state.ws.onclose = (event) => {
    if (state.wsPingTimer) {
      clearInterval(state.wsPingTimer);
      state.wsPingTimer = null;
    }
    logEvent("voice", `Voice websocket closed (${event.code})`);
    setTimeout(connectWS, 3000);
  };
}

function handleVoiceMessage(msg) {
  if (msg.type === "diarization") {
    state.currentVoiceStep = msg.step || 0;
    $("voice-step").textContent = msg.step || 0;
    $("voice-infer").textContent = fmtMs(msg.infer_ms);
    $("runtime-voice-step").textContent = msg.step || 0;
    $("runtime-voice-infer").textContent = fmtMs(msg.infer_ms);
    renderDiarizationTable(msg.speakers || []);
    updateTimeline(msg);
    const activeIds = new Set();
    for (const sp of msg.speakers || []) {
      activeIds.add(sp.id);
      updateSpeakerLane(sp);
    }
    for (const [id, lane] of Object.entries(speakerLanes)) {
      if (!activeIds.has(Number(id))) {
        lane.element.classList.remove("active");
        lane.element.querySelector(".glow-dot").className = "glow-dot off";
      }
    }
    pruneStaleUnknownLanes();
    if ((msg.speakers || []).length) {
      $("no-speakers-msg").style.display = "none";
    }
    return;
  }
  if (msg.type === "status") {
    logEvent("voice", msg.message);
    return;
  }
  if (msg.type === "error") {
    logEvent("voice", msg.message, { isError: true });
    return;
  }
  if (msg.type === "enrollment") {
    $("enroll-status").textContent = msg.message || msg.status;
    logEvent("voice", msg.message || msg.status);
    if (msg.status === "done" || msg.status === "error") {
      setButtonDisabled("btn-enroll", false);
      refreshSpeakers();
    }
    return;
  }
  if (msg.type === "diag") {
    $("diag-status").textContent = msg.message || msg.status;
    logEvent("voice", msg.message || msg.status, { isError: msg.status === "error" });
    if (msg.status === "done" || msg.status === "error") {
      setButtonDisabled("btn-diag", false);
      refreshDiagClips();
    }
  }
}

function createSpeakerLane(sp) {
  const id = sp.id;
  const color = COLORS[id % COLORS.length];
  const initial = sp.label.charAt(0).toUpperCase();
  const defaultMuted = !sp.enrolled;

  const div = document.createElement("div");
  div.className = "speaker-lane";
  div.id = `lane-${id}`;
  div.innerHTML = `
    <div class="glow-dot off"></div>
    <div class="avatar" style="background:${color}22; color:${color}; border:1px solid ${color}55">${initial}</div>
    <div class="speaker-main">
      <div class="speaker-topline">
        <span class="speaker-name">${sp.label}</span>
        <span class="speaker-badge ${sp.enrolled ? "enrolled" : "unknown"}">${sp.enrolled ? "enrolled" : "unknown"}</span>
        <span class="speaker-metrics"><span class="activity-pct">0% diar</span></span>
      </div>
      <canvas class="waveform" height="42"></canvas>
    </div>
    <div class="speaker-controls">
      <input type="range" min="0" max="100" value="80">
      <button class="mini-btn">${defaultMuted ? "unmute" : "mute"}</button>
    </div>
  `;

  $("speaker-lanes").appendChild(div);
  const canvas = div.querySelector("canvas");
  canvas.width = (canvas.offsetWidth || 420) * 2;
  canvas.height = 84;

  const lane = {
    element: div,
    canvas,
    canvasCtx: canvas.getContext("2d"),
    waveformBuf: new Float32Array(0),
    color,
    gainNode: null,
    muted: defaultMuted,
    autoMuted: defaultMuted,
    enrolled: !!sp.enrolled,
    nextPlayTime: 0,
    lastPacketStep: null,
    lastPacketStart: 0,
    lastSeenStep: state.currentVoiceStep,
  };

  div.querySelector('input[type="range"]').addEventListener("input", (event) => {
    setVolume(id, event.target.value);
  });
  div.querySelector(".mini-btn").addEventListener("click", () => toggleMute(id));
  speakerLanes[id] = lane;
  ensureLaneAudioNode(lane);
  refreshLaneMuteUI(lane);
  return lane;
}

function ensureLaneAudioNode(lane) {
  if (!state.audioCtx || lane.gainNode) return;
  const slider = lane.element.querySelector('input[type="range"]');
  lane.gainNode = state.audioCtx.createGain();
  lane.gainNode.gain.value = lane.muted ? 0 : Number(slider.value) / 100;
  lane.gainNode.connect(state.audioCtx.destination);
}

function refreshLaneMuteUI(lane) {
  ensureLaneAudioNode(lane);
  const button = lane.element.querySelector(".mini-btn");
  button.textContent = lane.muted ? "unmute" : "mute";
  button.style.color = lane.muted ? "#ffb0a8" : "";
  if (lane.gainNode) {
    const slider = lane.element.querySelector('input[type="range"]');
    lane.gainNode.gain.value = lane.muted ? 0 : Number(slider.value) / 100;
  }
}

function updateSpeakerLane(sp) {
  let lane = speakerLanes[sp.id];
  if (!lane) {
    lane = createSpeakerLane(sp);
  }
  lane.enrolled = !!sp.enrolled;
  lane.lastSeenStep = state.currentVoiceStep;
  const badge = lane.element.querySelector(".speaker-badge");
  badge.textContent = sp.enrolled ? "enrolled" : "unknown";
  badge.className = `speaker-badge ${sp.enrolled ? "enrolled" : "unknown"}`;
  if (sp.enrolled && lane.autoMuted) {
    lane.muted = false;
    lane.autoMuted = false;
    refreshLaneMuteUI(lane);
  }

  const audioActive = sp.audio_active == null ? !!sp.active : !!sp.audio_active;
  lane.element.classList.toggle("active", audioActive);
  lane.element.querySelector(".glow-dot").className = `glow-dot ${audioActive ? "on" : "off"}`;

  const diarPct = Math.round((sp.activity || 0) * 100);
  let line = `${diarPct}% diar`;
  if (sp.audio_peak != null) {
    line += ` · ${Math.round(sp.audio_peak * 100)}% aud`;
  }
  if (sp.enrolled && sp.identity_similarity != null) {
    line += ` · ${Math.round(sp.identity_similarity * 100)}% ID`;
  }
  lane.element.querySelector(".activity-pct").textContent = line;
}

function pruneStaleUnknownLanes() {
  for (const [id, lane] of Object.entries(speakerLanes)) {
    if (lane.enrolled) continue;
    if (state.currentVoiceStep - (lane.lastSeenStep || 0) < UNKNOWN_LANE_STALE_STEPS) continue;
    if (lane.gainNode) lane.gainNode.disconnect();
    lane.element.remove();
    delete speakerLanes[id];
  }
}

function resetVoiceStage() {
  for (const [id, lane] of Object.entries(speakerLanes)) {
    if (lane.gainNode) {
      try {
        lane.gainNode.disconnect();
      } catch (err) {
        console.warn(err);
      }
    }
    lane.element.remove();
    delete speakerLanes[id];
  }
  $("speaker-lanes").innerHTML = '<div class="empty-card" id="no-speakers-msg">No speaker lanes yet. Start Voice to begin streaming.</div>';
  state.currentVoiceStep = 0;
  $("voice-step").textContent = "0";
  $("voice-infer").textContent = "-";
  $("runtime-voice-step").textContent = "0";
  $("runtime-voice-infer").textContent = "-";
  resetDiarizationView();
}

function renderDiarizationTable(speakers = []) {
  const body = $("diarization-current-body");
  body.innerHTML = "";
  if (!speakers.length) {
    body.innerHTML = '<tr><td colspan="6" class="diarization-empty">No diarization data yet.</td></tr>';
    return;
  }

  const sorted = [...speakers].sort((a, b) => {
    if (!!a.enrolled !== !!b.enrolled) return a.enrolled ? -1 : 1;
    const aActive = !!(a.audio_active || a.diar_active || a.active);
    const bActive = !!(b.audio_active || b.diar_active || b.active);
    if (aActive !== bActive) return aActive ? -1 : 1;
    return String(a.label).localeCompare(String(b.label));
  });

  for (const sp of sorted) {
    const row = document.createElement("tr");
    const color = COLORS[sp.id % COLORS.length];
    const diarPct = Math.max(0, Math.min(100, Math.round((sp.activity || 0) * 100)));
    const audioPct = sp.audio_peak == null ? "-" : `${Math.max(0, Math.min(100, Math.round(sp.audio_peak * 100)))}%`;
    const idPct = sp.identity_similarity == null ? "-" : `${Math.max(0, Math.min(100, Math.round(sp.identity_similarity * 100)))}%`;
    const active = !!(sp.audio_active || sp.diar_active || sp.active);
    row.innerHTML = `
      <td>
        <span class="speaker-name-cell">
          <span class="speaker-dot" style="background:${color}"></span>
          <span>${escapeHtml(sp.label)}</span>
        </span>
      </td>
      <td><span class="speaker-kind ${sp.enrolled ? "enrolled" : "unknown"}">${sp.enrolled ? "Enrolled" : "Unknown"}</span></td>
      <td>${diarPct}%</td>
      <td>${audioPct}</td>
      <td>${idPct}</td>
      <td><span class="speaker-state ${active ? "live" : "idle"}">${active ? "Live" : "Idle"}</span></td>
    `;
    body.appendChild(row);
  }
}

function initTimeline() {
  const canvas = $("timeline-canvas");
  if (!canvas) return;
  const container = canvas.parentElement;
  canvas.width = Math.max(1, container.offsetWidth * 2);
  canvas.height = Math.max(1, container.offsetHeight * 2);
  timeline.canvas = canvas;
  timeline.ctx = canvas.getContext("2d");
  timeline.windowSec = parseInt($("timeline-window").value, 10);
  drawTimeline();
}

function timelineWindowChanged() {
  timeline.windowSec = parseInt($("timeline-window").value, 10);
  drawTimeline();
}

function updateTimeline(msg) {
  const t = (msg.step || 0) * timeline.stepDuration;
  const activeSpeakers = [];
  for (const sp of msg.speakers || []) {
    const diarActive = sp.diar_active == null ? (sp.activity || 0) > 0.3 : !!sp.diar_active;
    const audioActive = !!sp.audio_active;
    if (audioActive || diarActive) {
      activeSpeakers.push({
        id: sp.id,
        label: sp.label,
        activity: sp.activity || 0,
        enrolled: !!sp.enrolled,
        audio_active: audioActive,
      });
      if (!timeline.knownSpeakers[sp.id]) {
        timeline.knownSpeakers[sp.id] = {
          label: sp.label,
          color: COLORS[sp.id % COLORS.length],
          enrolled: !!sp.enrolled,
        };
        updateTimelineLegend();
      } else {
        timeline.knownSpeakers[sp.id].label = sp.label;
        timeline.knownSpeakers[sp.id].enrolled = !!sp.enrolled;
      }
    }
  }
  timeline.steps.push({ t, speakers: activeSpeakers });
  const maxKeep = Math.floor(300 / timeline.stepDuration);
  if (timeline.steps.length > maxKeep) {
    timeline.steps = timeline.steps.slice(timeline.steps.length - maxKeep);
  }
  updateTimelineLegend();
  drawTimeline();
}

function updateTimelineLegend() {
  const el = $("timeline-legend");
  el.innerHTML = "";
  const sorted = Object.entries(timeline.knownSpeakers).sort((a, b) => Number(a[0]) - Number(b[0]));
  for (const [id, info] of sorted) {
    const item = document.createElement("div");
    item.className = "timeline-legend-item";
    item.innerHTML = `
      <span class="timeline-legend-dot" style="background:${info.color}"></span>
      <span>${escapeHtml(info.label)}${info.enrolled ? " \u2713" : ""}</span>
    `;
    el.appendChild(item);
  }
}

function formatTimelineTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function drawTimeline() {
  const ctx = timeline.ctx;
  const canvas = timeline.canvas;
  if (!ctx || !canvas) return;

  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 8, bottom: 8, left: 4, right: 4 };
  const drawW = W - pad.left - pad.right;
  const drawH = H - pad.top - pad.bottom;

  ctx.fillStyle = "#10161a";
  ctx.fillRect(0, 0, W, H);

  if (!timeline.steps.length) {
    ctx.fillStyle = "#7a878f";
    ctx.font = `${Math.round(H * 0.12)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("Waiting for diarization data...", W / 2, H / 2);
    return;
  }

  const lastT = timeline.steps[timeline.steps.length - 1].t;
  const tEnd = Math.max(lastT, timeline.windowSec);
  const tStart = tEnd - timeline.windowSec;
  $("timeline-t-start").textContent = formatTimelineTime(Math.max(0, tStart));
  $("timeline-t-end").textContent = formatTimelineTime(tEnd);

  const speakerOrder = [];
  const seen = new Set();
  for (const [id, info] of Object.entries(timeline.knownSpeakers)) {
    if (info.enrolled && !seen.has(id)) {
      speakerOrder.push(id);
      seen.add(id);
    }
  }
  for (const [id] of Object.entries(timeline.knownSpeakers)) {
    if (!seen.has(id)) {
      speakerOrder.push(id);
      seen.add(id);
    }
  }
  if (!speakerOrder.length) return;

  const rowH = Math.min(drawH / speakerOrder.length, 28);
  const rowGap = Math.max(2, Math.round(rowH * 0.15));
  const barH = rowH - rowGap;
  const totalH = speakerOrder.length * rowH;
  const yOffset = pad.top + (drawH - totalH) / 2;

  ctx.font = `bold ${Math.round(barH * 0.65)}px sans-serif`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let r = 0; r < speakerOrder.length; r += 1) {
    const id = speakerOrder[r];
    const info = timeline.knownSpeakers[id];
    const y = yOffset + r * rowH;
    ctx.fillStyle = `${info.color}18`;
    ctx.fillRect(pad.left, y, drawW, barH);
    ctx.fillStyle = `${info.color}`;
    ctx.fillText(info.label, pad.left + 92, y + barH / 2);
  }

  const barLeft = pad.left + 98;
  const barW = Math.max(1, drawW - 98);
  for (const step of timeline.steps) {
    if (step.t < tStart || step.t > tEnd) continue;
    const xFrac = (step.t - tStart) / timeline.windowSec;
    const x = barLeft + xFrac * barW;
    const segW = Math.max(2, (timeline.stepDuration / timeline.windowSec) * barW);
    for (const sp of step.speakers) {
      const rowIdx = speakerOrder.indexOf(String(sp.id));
      if (rowIdx < 0) continue;
      const y = yOffset + rowIdx * rowH;
      const info = timeline.knownSpeakers[sp.id];
      const alpha = Math.min(1, 0.4 + (sp.activity || 0) * 0.6);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = info.color;
      ctx.fillRect(Math.round(x), Math.round(y), Math.ceil(segW), barH);
    }
  }
  ctx.globalAlpha = 1;

  for (const step of timeline.steps) {
    if (step.t < tStart || step.t > tEnd || step.speakers.length < 2) continue;
    const xFrac = (step.t - tStart) / timeline.windowSec;
    const x = barLeft + xFrac * barW;
    const segW = Math.max(2, (timeline.stepDuration / timeline.windowSec) * barW);
    ctx.globalAlpha = 0.75;
    ctx.fillStyle = "#e6a463";
    ctx.fillRect(Math.round(x), pad.top, Math.ceil(segW), 3);
  }
  ctx.globalAlpha = 1;

  const nowFrac = (lastT - tStart) / timeline.windowSec;
  const nowX = barLeft + nowFrac * barW;
  ctx.strokeStyle = "#f5f3ee";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(nowX, pad.top);
  ctx.lineTo(nowX, H - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
}

function resetDiarizationView() {
  renderDiarizationTable([]);
  timeline.steps = [];
  timeline.knownSpeakers = {};
  updateTimelineLegend();
  drawTimeline();
  $("timeline-t-start").textContent = "0:00";
  $("timeline-t-end").textContent = "0:00";
}

function decodePcm16(i16) {
  const out = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i += 1) {
    out[i] = i16[i] / 32768;
  }
  return out;
}

function appendSpeakerAudio(speakerId, f32, sampleRate, packetStep) {
  const lane = speakerLanes[speakerId];
  if (!lane) return;
  const maxSamples = sampleRate * 3;
  const combined = new Float32Array(lane.waveformBuf.length + f32.length);
  combined.set(lane.waveformBuf);
  combined.set(f32, lane.waveformBuf.length);
  lane.waveformBuf = combined.length > maxSamples
    ? combined.slice(combined.length - maxSamples)
    : combined;
  drawWaveform(lane);
  if (state.audioCtx && !lane.muted) {
    playAudio(f32, sampleRate, lane, packetStep);
  }
}

function handleAudioPacket(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < AUDIO_PACKET_HEADER_BYTES) return;
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== AUDIO_PACKET_MAGIC) return;
  const packetStep = view.getUint32(4, true);
  const speakerId = view.getUint32(8, true);
  const sampleRate = view.getUint32(12, true);
  const numSamples = view.getUint32(16, true);
  if (buffer.byteLength < AUDIO_PACKET_HEADER_BYTES + numSamples * 2) return;
  const i16 = new Int16Array(buffer, AUDIO_PACKET_HEADER_BYTES, numSamples);
  appendSpeakerAudio(speakerId, decodePcm16(i16), sampleRate, packetStep);
}

function drawWaveform(lane) {
  const ctx = lane.canvasCtx;
  const w = lane.canvas.width;
  const h = lane.canvas.height;
  const data = lane.waveformBuf;
  ctx.fillStyle = "rgba(4, 10, 12, 0.95)";
  ctx.fillRect(0, 0, w, h);
  if (!data.length) return;
  ctx.strokeStyle = lane.color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / w));
  const mid = h / 2;
  for (let x = 0; x < w; x += 1) {
    const idx = Math.floor(x * data.length / w);
    let min = 1;
    let max = -1;
    for (let j = 0; j < step && idx + j < data.length; j += 1) {
      const v = data[idx + j];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x, mid + min * mid * 0.9);
    ctx.lineTo(x, mid + max * mid * 0.9);
  }
  ctx.stroke();
}

function playAudio(f32, sampleRate, lane, packetStep) {
  if (!state.audioCtx || !lane.gainNode) return;
  if (state.audioCtx.state !== "running") {
    state.audioCtx.resume().catch(() => {});
    return;
  }
  const buffer = state.audioCtx.createBuffer(1, f32.length, sampleRate);
  buffer.getChannelData(0).set(f32);
  const src = state.audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(lane.gainNode);

  const now = state.audioCtx.currentTime;
  const duration = f32.length / sampleRate;
  const targetStart = now + AUDIO_LOOKAHEAD_SEC;
  let scheduledStart = targetStart;
  const hasPacketStep = Number.isInteger(packetStep);

  if (hasPacketStep && Number.isInteger(lane.lastPacketStep) && packetStep > lane.lastPacketStep) {
    const stepDelta = packetStep - lane.lastPacketStep;
    const expectedStart = lane.lastPacketStart + stepDelta * duration;
    if (expectedStart <= now + AUDIO_MAX_BUFFER_SEC) {
      scheduledStart = Math.max(expectedStart, now + AUDIO_MIN_LEAD_SEC);
    }
  } else if (lane.nextPlayTime >= now + AUDIO_MIN_LEAD_SEC && lane.nextPlayTime <= now + AUDIO_MAX_BUFFER_SEC) {
    scheduledStart = lane.nextPlayTime;
  }

  src.start(scheduledStart);
  lane.nextPlayTime = scheduledStart + duration;
  lane.lastPacketStart = scheduledStart;
  lane.lastPacketStep = hasPacketStep ? packetStep : null;
}

function setVolume(id, value) {
  const lane = speakerLanes[id];
  if (!lane || !lane.gainNode) return;
  lane.gainNode.gain.value = lane.muted ? 0 : Number(value) / 100;
}

function toggleMute(id) {
  const lane = speakerLanes[id];
  if (!lane) return;
  lane.autoMuted = false;
  lane.muted = !lane.muted;
  refreshLaneMuteUI(lane);
}

function renderSpeakers(list) {
  const root = $("enrolled-list");
  root.innerHTML = "";
  if (!list.length) {
    root.innerHTML = '<div class="list-item empty">No enrolled speakers yet.</div>';
    return;
  }
  for (const sp of list) {
    const card = document.createElement("section");
    card.className = "speaker-library-card";

    const head = document.createElement("div");
    head.className = "speaker-library-head";
    head.innerHTML = `
      <div>
        <strong>${escapeHtml(sp.name)}</strong>
        <div class="subtle">
          ${sp.reference_count || 0} refs • ${fmtSeconds(sp.total_reference_sec || 0)} • ${sp.has_embedding ? "embedding ready" : "missing embedding"}
        </div>
      </div>
      <button class="btn btn-ghost btn-small" type="button">Delete Speaker</button>
    `;
    head.querySelector("button").addEventListener("click", () => deleteSpeaker(sp.name));
    card.appendChild(head);

    const clips = document.createElement("div");
    clips.className = "clip-list";
    const refs = sp.references || [];
    if (!refs.length) {
      clips.innerHTML = '<div class="list-item empty">No reference clips yet.</div>';
    } else {
      for (const ref of refs) {
        const row = document.createElement("div");
        row.className = "clip-row";
        row.innerHTML = `
          <div class="clip-info">
            <div class="clip-title">${escapeHtml(ref.label || ref.name)}</div>
            <div class="clip-meta">${escapeHtml(ref.name)} • ${fmtSeconds(ref.duration_sec)} • ${fmtBytes(ref.size)}${fmtStamp(ref.modified_at) ? ` • ${escapeHtml(fmtStamp(ref.modified_at))}` : ""}</div>
          </div>
          <div class="clip-actions">
            <button class="btn btn-ghost btn-small" type="button">Delete</button>
          </div>
        `;
        row.querySelector("button").addEventListener("click", () => deleteReference(sp.name, ref.name));
        clips.appendChild(row);
      }
    }
    card.appendChild(clips);
    root.appendChild(card);
  }
}

function renderDiagClips(list) {
  const root = $("diag-clips");
  root.innerHTML = "";
  if (!list.length) {
    root.innerHTML = '<div class="list-item empty">No diagnostic clips yet.</div>';
    return;
  }
  for (const clip of list) {
    const row = document.createElement("div");
    row.className = "clip-row";
    row.innerHTML = `
      <div class="clip-info">
        <div class="clip-title">${escapeHtml(clip.label || clip.name)}</div>
        <div class="clip-meta">${escapeHtml(clip.name)} • ${fmtSeconds(clip.duration_sec)} • ${fmtBytes(clip.size)}${fmtStamp(clip.modified_at) ? ` • ${escapeHtml(fmtStamp(clip.modified_at))}` : ""}</div>
      </div>
      <div class="clip-actions">
        <button class="btn btn-secondary btn-small" type="button">Add As Ref</button>
        <button class="btn btn-ghost btn-small" type="button">Delete</button>
      </div>
    `;
    const [promoteBtn, deleteBtn] = row.querySelectorAll("button");
    promoteBtn.addEventListener("click", () => addDiagClipAsReference(clip));
    deleteBtn.addEventListener("click", () => deleteDiagClip(clip.name));
    root.appendChild(row);
  }
}

function renderPoguise(status) {
  const running = !!status.running;
  $("pog-fps").textContent = running && status.fps_cam ? `${status.fps_cam.toFixed(1)} fps` : "-";
  $("pog-infer").textContent = running ? fmtMs(status.inference_ms) : "-";
  $("runtime-pog-infer").textContent = running ? fmtMs(status.inference_ms) : "-";
  $("runtime-pog-fps").textContent = running && status.fps_cam ? `${status.fps_cam.toFixed(1)} fps` : "-";
  $("pog-current-action").textContent = running ? (status.current_action || "Waiting...") : "Waiting...";
  $("pog-buffer").textContent = running ? fmtPct(status.buffer_pct) : "-";
  $("pog-effective").textContent = running ? (status.effective_infer_every ?? "-") : "-";
  $("pog-skip-policy").textContent = status.skipped_due_to_policy ?? 0;
  $("pog-skip-gpu").textContent = status.skipped_due_to_busy_gpu ?? 0;
  $("pog-checkpoint").textContent = status.checkpoint || "-";
  $("btn-toggle-heatmap").textContent = status.heatmap_enabled ? "Heatmap On" : "Heatmap";
  if (status.status_message && status.status_message !== state.lastPoguiseStatusMessage) {
    state.lastPoguiseStatusMessage = status.status_message;
    logEvent("poguise", status.status_message, { isError: !!status.error });
  }
  if (status.error && status.error !== state.lastPoguiseError) {
    state.lastPoguiseError = status.error;
    logEvent("poguise", status.error, { isError: true });
  }
  if (!status.error) {
    state.lastPoguiseError = "";
  }

  const predictions = $("pog-predictions");
  predictions.innerHTML = "";
  for (const item of running ? (status.predictions || []) : []) {
    const div = document.createElement("div");
    div.className = "prediction-item";
    div.innerHTML = `
      <div class="prediction-topline">
        <span>${item.label}</span>
        <strong>${item.prob.toFixed(1)}%</strong>
      </div>
      <div class="prediction-bar">
        <div class="prediction-fill" style="width:${Math.max(item.prob, 1)}%"></div>
      </div>
    `;
    predictions.appendChild(div);
  }
  if (!running || !status.predictions || !status.predictions.length) {
    predictions.innerHTML = `<div class="list-item empty">${running ? "Predictions will appear after the buffer fills." : "Start PO-GUISE to see live predictions."}</div>`;
  }

  const history = $("pog-history");
  history.innerHTML = "";
  if (!running || !status.action_history || !status.action_history.length) {
    history.innerHTML = `<div class="list-item empty">${running ? "No committed actions yet." : "PO-GUISE history will appear while live."}</div>`;
    if (!running) {
      state.lastPoguiseHistoryKey = "";
    }
  } else {
    for (const item of status.action_history) {
      const row = document.createElement("div");
      row.className = "list-item";
      row.innerHTML = `<span>${item.label}</span><strong>${item.time}</strong>`;
      history.appendChild(row);
    }
    const latest = status.action_history[0];
    const latestKey = `${latest.time}:${latest.label}`;
    if (latestKey !== state.lastPoguiseHistoryKey) {
      state.lastPoguiseHistoryKey = latestKey;
      logEvent("poguise", `Action committed: ${latest.label}`, { meta: latest.time });
    }
  }
}

function renderScheduler(snapshot) {
  $("sched-last-voice").textContent = snapshot.last_voice_ms == null ? "-" : fmtMs(snapshot.last_voice_ms);
  $("sched-avg-voice").textContent = snapshot.avg_voice_ms == null ? "-" : fmtMs(snapshot.avg_voice_ms);
  $("sched-last-pog").textContent = snapshot.last_poguise_ms == null ? "-" : fmtMs(snapshot.last_poguise_ms);
  $("sched-protection").textContent = snapshot.config?.enabled ? "On" : "Off";
}

function applySystemStatus(data) {
  const wasVoiceRunning = state.voiceRunning;
  const wasPoguiseRunning = state.poguiseRunning;
  const wasAnyRunning = wasVoiceRunning || wasPoguiseRunning;
  state.voiceRunning = !!data.voice.running;
  state.poguiseRunning = !!data.poguise.running;
  const isAnyRunning = state.voiceRunning || state.poguiseRunning;
  syncActionButton("btn-system-action", isAnyRunning, "Start Runtime", "Stop Runtime");
  syncActionButton("btn-voice-action", state.voiceRunning, "Start Voice", "Stop Voice");
  syncActionButton("btn-pog-action", state.poguiseRunning, "Start PO-GUISE", "Stop PO-GUISE");
  renderSystemStatus(data);
  renderPoguise(data.poguise);
  renderScheduler(data.scheduler);
  setVoiceToolsVisible(!state.voiceRunning);
  if (state.voiceRunning && !wasVoiceRunning) {
    setShellVisibility("voice", false);
    logEvent("voice", "Voice is live");
  }
  if (!state.voiceRunning && wasVoiceRunning) {
    setShellVisibility("voice", true);
    resetVoiceStage();
    logEvent("voice", "Voice stopped");
  }
  if (state.poguiseRunning && !wasPoguiseRunning) {
    setShellVisibility("poguise", false);
  }
  if (!state.poguiseRunning && wasPoguiseRunning) {
    setShellVisibility("poguise", true);
    resetPoguiseVideo();
  }
  if (isAnyRunning && !wasAnyRunning) setShellVisibility("scheduler", false);
  if (!isAnyRunning && wasAnyRunning) setShellVisibility("scheduler", true);
  $("voice-step").textContent = state.voiceRunning ? (data.voice.step || 0) : 0;
  $("voice-infer").textContent = state.voiceRunning ? fmtMs(data.voice.infer_ms) : "-";
  $("runtime-voice-step").textContent = state.voiceRunning ? (data.voice.step || 0) : 0;
  $("runtime-voice-infer").textContent = state.voiceRunning ? fmtMs(data.voice.infer_ms) : "-";
  refreshEditorInteractivity(state.editors.voice);
  refreshEditorInteractivity(state.editors.poguise);
  refreshEditorInteractivity(state.editors.scheduler);
  if (state.poguiseRunning) {
    fetchPoguiseFrame().catch(() => {});
  }
}

async function refreshStatuses() {
  try {
    const data = await fetchJson("/api/system/status");
    applySystemStatus(data);
  } catch (err) {
    logEvent("system", `Status refresh failed: ${err}`, { isError: true });
  }
}

async function refreshSpeakers() {
  const data = await fetchJson("/api/voice/speakers");
  renderSpeakers(data);
}

async function refreshDiagClips() {
  const data = await fetchJson("/api/voice/diag/clips");
  renderDiagClips(data);
}

async function startVoice() {
  await ensureAudioReady();
  await saveEditor(state.editors.voice, { silent: true });
  await fetchJson("/api/voice/start", { method: "POST" });
  setShellVisibility("voice", false);
  logEvent("voice", "Voice start requested");
  refreshStatuses();
}

async function stopVoice() {
  await fetchJson("/api/voice/stop", { method: "POST" });
  setShellVisibility("voice", true);
  logEvent("voice", "Voice stop requested");
  refreshStatuses();
}

async function startPoguise() {
  await saveEditor(state.editors.poguise, { silent: true });
  await fetchJson("/api/poguise/start", { method: "POST" });
  setShellVisibility("poguise", false);
  refreshStatuses();
}

async function stopPoguise() {
  await fetchJson("/api/poguise/stop", { method: "POST" });
  setShellVisibility("poguise", true);
  refreshStatuses();
}

async function startAll() {
  await ensureAudioReady();
  await Promise.all([
    saveEditor(state.editors.voice, { silent: true }),
    saveEditor(state.editors.poguise, { silent: true }),
    saveEditor(state.editors.scheduler, { silent: true }),
  ]);
  await fetchJson("/api/system/start", { method: "POST" });
  setShellVisibility("voice", false);
  setShellVisibility("poguise", false);
  setShellVisibility("scheduler", false);
  logEvent("system", "Start runtime requested");
  refreshStatuses();
}

async function stopAll() {
  await fetchJson("/api/system/stop", { method: "POST" });
  setShellVisibility("voice", true);
  setShellVisibility("poguise", true);
  setShellVisibility("scheduler", true);
  logEvent("system", "Stop runtime requested");
  refreshStatuses();
}

async function enrollSpeaker() {
  const name = $("enroll-name").value.trim();
  const referenceLabel = $("enroll-reference-label").value.trim();
  const duration = parseInt($("enroll-duration").value, 10);
  if (!name) {
    logEvent("voice", "Enter a speaker name first.", { isError: true });
    return;
  }
  setButtonDisabled("btn-enroll", true);
  $("enroll-status").textContent = "Starting enrollment...";
  await fetchJson("/api/voice/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, duration, reference_label: referenceLabel || null }),
  });
}

async function recordDiag() {
  const label = $("diag-label").value.trim();
  const duration = parseInt($("diag-duration").value, 10);
  if (!label) {
    logEvent("voice", "Enter a diagnostic clip label first.", { isError: true });
    return;
  }
  setButtonDisabled("btn-diag", true);
  $("diag-status").textContent = "Starting clip recording...";
  await fetchJson("/api/voice/diag/record", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, duration }),
  });
}

async function deleteSpeaker(name) {
  if (!confirm(`Delete speaker "${name}"?`)) return;
  await fetchJson(`/api/voice/speakers/${encodeURIComponent(name)}`, { method: "DELETE" });
  logEvent("voice", `Deleted speaker ${name}`);
  refreshSpeakers();
}

async function deleteReference(speakerName, filename) {
  if (!confirm(`Delete reference "${filename}" from "${speakerName}"?`)) return;
  await fetchJson(
    `/api/voice/speakers/${encodeURIComponent(speakerName)}/references/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );
  logEvent("voice", `Deleted ${filename} from ${speakerName}`);
  refreshSpeakers();
}

async function deleteDiagClip(filename) {
  if (!confirm(`Delete diagnostic clip "${filename}"?`)) return;
  await fetchJson(`/api/voice/diag/clips/${encodeURIComponent(filename)}`, { method: "DELETE" });
  logEvent("voice", `Deleted diagnostic clip ${filename}`);
  refreshDiagClips();
}

async function addDiagClipAsReference(clip) {
  const fallbackSpeaker = $("diag-target-speaker").value.trim() || $("enroll-name").value.trim();
  if (!fallbackSpeaker) {
    logEvent("voice", "Enter a speaker in the diagnostic target field first.", { isError: true });
    return;
  }
  const referenceLabel = $("diag-reference-label").value.trim() || clip.label || "";
  await fetchJson("/api/voice/diag/promote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speaker_name: fallbackSpeaker,
      clip_name: clip.name,
      reference_label: referenceLabel || null,
    }),
  });
  logEvent(
    "voice",
    `Added ${clip.name} to ${fallbackSpeaker}${referenceLabel ? ` as ${referenceLabel}` : ""}`,
  );
  $("diag-target-speaker").value = fallbackSpeaker;
  if (!($("diag-reference-label").value || "").trim()) {
    $("diag-reference-label").value = referenceLabel;
  }
  await refreshSpeakers();
}

async function toggleHeatmap() {
  const data = await fetchJson("/api/poguise/heatmap/toggle", { method: "POST" });
  $("btn-toggle-heatmap").textContent = data.show_heatmap ? "Heatmap On" : "Heatmap";
  logEvent("poguise", data.show_heatmap ? "Heatmap enabled" : "Heatmap disabled");
}

function resetPoguiseVideo() {
  const img = $("poguise-video");
  if (state.videoObjectUrl) {
    URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = null;
  }
  img.removeAttribute("src");
}

async function fetchPoguiseFrame() {
  if (!state.poguiseRunning || state.videoRequestInFlight) return;
  state.videoRequestInFlight = true;
  try {
    const resp = await fetch(`/api/poguise/frame.jpg?ts=${Date.now()}`, { cache: "no-store" });
    if (!resp.ok) return;
    const blob = await resp.blob();
    if (!blob.size) return;
    const nextUrl = URL.createObjectURL(blob);
    const prevUrl = state.videoObjectUrl;
    state.videoObjectUrl = nextUrl;
    $("poguise-video").src = nextUrl;
    if (prevUrl) {
      URL.revokeObjectURL(prevUrl);
    }
  } catch (err) {
    console.debug(err);
  } finally {
    state.videoRequestInFlight = false;
  }
}

function ensurePoguiseVideoPolling() {
  if (state.videoPollTimer) return;
  state.videoPollTimer = setInterval(() => {
    fetchPoguiseFrame().catch(() => {});
  }, 275);
}

function bindUI() {
  $("btn-system-action").addEventListener("click", () => (
    (state.voiceRunning || state.poguiseRunning ? stopAll() : startAll()).catch((err) => logEvent("system", err, { isError: true }))
  ));
  $("btn-voice-action").addEventListener("click", () => (
    (state.voiceRunning ? stopVoice() : startVoice()).catch((err) => logEvent("voice", err, { isError: true }))
  ));
  $("btn-pog-action").addEventListener("click", () => (
    (state.poguiseRunning ? stopPoguise() : startPoguise()).catch((err) => logEvent("poguise", err, { isError: true }))
  ));
  $("btn-enroll").addEventListener("click", () => enrollSpeaker().catch((err) => logEvent("voice", err, { isError: true })));
  $("btn-diag").addEventListener("click", () => recordDiag().catch((err) => logEvent("voice", err, { isError: true })));
  $("btn-toggle-heatmap").addEventListener("click", () => toggleHeatmap().catch((err) => logEvent("poguise", err, { isError: true })));

  $("btn-save-voice").addEventListener("click", () => saveEditor(state.editors.voice).catch((err) => logEvent("voice", err, { isError: true })));
  $("btn-reset-voice").addEventListener("click", () => resetEditor(state.editors.voice).catch((err) => logEvent("voice", err, { isError: true })));
  $("btn-save-pog").addEventListener("click", () => saveEditor(state.editors.poguise).catch((err) => logEvent("poguise", err, { isError: true })));
  $("btn-reset-pog").addEventListener("click", () => resetEditor(state.editors.poguise).catch((err) => logEvent("poguise", err, { isError: true })));
  $("btn-save-scheduler").addEventListener("click", () => saveEditor(state.editors.scheduler).catch((err) => logEvent("system", err, { isError: true })));
  $("btn-reset-scheduler").addEventListener("click", () => resetEditor(state.editors.scheduler).catch((err) => logEvent("system", err, { isError: true })));

  $("btn-toggle-voice-settings").addEventListener("click", () => toggleShell("voice"));
  $("btn-toggle-pog-settings").addEventListener("click", () => toggleShell("poguise"));
  $("btn-toggle-scheduler-settings").addEventListener("click", () => toggleShell("scheduler"));
  $("timeline-window").addEventListener("change", timelineWindowChanged);
}

async function bootstrap() {
  const data = await fetchJson("/api/bootstrap");
  applyEditorPayload(state.editors.voice, data.voice.editor);
  applyEditorPayload(state.editors.poguise, data.poguise.editor);
  applyEditorPayload(state.editors.scheduler, data.scheduler.editor);
  renderSpeakers(data.voice.speakers || []);
  renderDiagClips(data.voice.diag_clips || []);
  applySystemStatus({
    voice: data.voice.status,
    poguise: data.poguise.status,
    scheduler: data.scheduler.status,
  });
  setVoiceToolsVisible(!data.voice.status.running);
  if (data.poguise.status.running) {
    await fetchPoguiseFrame();
  } else {
    resetPoguiseVideo();
  }
}

async function init() {
  bindUI();
  renderEventLog();
  initTimeline();
  connectWS();
  ensurePoguiseVideoPolling();
  await bootstrap();
  setInterval(refreshStatuses, 600);
}

window.addEventListener("load", () => {
  init().catch((err) => {
    logEvent("system", `Bootstrap failed: ${err}`, { isError: true });
  });
});

window.addEventListener("beforeunload", () => {
  if (state.videoPollTimer) {
    clearInterval(state.videoPollTimer);
    state.videoPollTimer = null;
  }
  if (state.videoObjectUrl) {
    URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = null;
  }
});

window.addEventListener("resize", () => {
  initTimeline();
  for (const lane of Object.values(speakerLanes)) {
    lane.canvas.width = (lane.canvas.offsetWidth || 420) * 2;
  }
});
