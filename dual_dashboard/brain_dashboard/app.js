"use strict";

const STAGE_ORDER = ["start", "event_bus", "world_state", "rule_engine", "llm_reasoner", "executor"];
const SYSTEM_ORDER = STAGE_ORDER.slice(1);
const LAYOUT_KEY = "brain-flow-layout-v2";
const PARTICLE_COLORS = ["#d4a449", "#ff6c5d", "#4db8ff", "#f0d389", "#9cdfff"];
const INTENT_OPTIONS = [
  { id: "create_rule", label: "Create Rule", desc: "Speech becomes a new automation." },
  { id: "modify_rule", label: "Modify Rule", desc: "Speech edits an existing automation." },
  { id: "direct_command", label: "Direct Command", desc: "Speech should execute immediately." },
  { id: "confirmation", label: "Confirmation", desc: "Speech answers a pending question." },
  { id: "question", label: "Question", desc: "Speech asks something without acting." },
  { id: "conversation", label: "Conversation", desc: "Speech is ordinary talk and is ignored." },
  { id: "bypassed", label: "Bypassed", desc: "LLM did not participate for this trace." },
];
const EVENT_OPTIONS = [
  { id: "speaker_active", label: "Speaker Active", desc: "Known or unknown speaker became active." },
  { id: "speaker_silent", label: "Speaker Silent", desc: "Speaker stopped being active." },
  { id: "speech_text", label: "Speech Text", desc: "Text transcript entered the brain." },
  { id: "action_detected", label: "Action Detected", desc: "Vision predicted a raw action." },
  { id: "action_changed", label: "Action Changed", desc: "Action became stable after debounce." },
  { id: "person_left", label: "Person Left", desc: "Presence tracker removed someone." },
  { id: "person_entered", label: "Person Entered", desc: "Presence tracker created a person." },
  { id: "system_status", label: "System Status", desc: "Runtime on/off flags changed." },
];

const STAGE_META = {
  start: {
    title: "Start Event",
    kicker: "Start",
    tone: "gold",
    summary: "The trigger that entered the brain.",
  },
  event_bus: {
    title: "Event Bus Routing",
    kicker: "System 1",
    tone: "gold",
    summary: "How the event got routed and fanned out.",
  },
  world_state: {
    title: "Context Builder",
    kicker: "System 2",
    tone: "blue",
    summary: "Which state trackers updated from the event.",
  },
  rule_engine: {
    title: "Smart Home Rules",
    kicker: "System 3",
    tone: "gold",
    summary: "The candidate automations for this event.",
  },
  llm_reasoner: {
    title: "Reasoner",
    kicker: "System 4",
    tone: "blue",
    summary: "Intent or interpretation path, when needed.",
  },
  executor: {
    title: "Action Gate",
    kicker: "System 5",
    tone: "red",
    summary: "Permission path and final output.",
  },
};

const QUICK_ACTIONS = [
  {
    label: "Speaker Active",
    handler: () => injectEvent("speaker_active", { who: "michel", enrolled: true, confidence: 0.98 }),
  },
  {
    label: "Dinner Path",
    handler: async () => {
      await injectEvent("speaker_active", { who: "michel", enrolled: true, confidence: 0.98 });
      await injectEvent("action_changed", { from: "Walk", to: "Eat.Attable", confidence: 0.94 });
    },
  },
  {
    label: "Cooking Path",
    handler: () => injectEvent("action_changed", { from: "Walk", to: "Cook.Stir", confidence: 0.93 }),
  },
  {
    label: "Speech Path",
    handler: () => injectEvent("speech_text", { who: "michel", text: "turn off all lights when I leave" }),
  },
  {
    label: "Leave Path",
    handler: () => injectEvent("person_left", { who: "michel" }),
  },
];

const state = {
  status: null,
  world: null,
  rules: [],
  queue: { pending_count: 0, active_count: 0, queue_depth: 0, items: [] },
  traces: [],
  traceMap: new Map(),
  events: [],
  eventMap: new Map(),
  actions: [],
  pending: [],
  fireHistory: [],
  selectedTraceId: null,
  selectedStageId: "start",
  selectedOptionId: null,
  detailOpen: false,
  autoFollow: true,
  streamConnected: false,
  eventSource: null,
  stagePositions: {},
  stageModels: {},
  stageOrderRendered: false,
  masterPathEl: null,
  progressPathEl: null,
  replay: {
    progress: 0,
    lastBurstIndex: -1,
  },
  particles: [],
  lastParticleAt: 0,
  lastFrameAt: 0,
};

const renderQueueState = {
  frame: 0,
  full: false,
  live: false,
  animate: false,
};

const streamState = {
  progressTimer: 0,
  pendingProgress: null,
};

const STREAM_PROGRESS_RENDER_MS = 140;

function byId(id) {
  return document.getElementById(id);
}

async function copyText(value) {
  const text = String(value ?? "");
  if (!text) return;
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const probe = document.createElement("textarea");
  probe.value = text;
  probe.setAttribute("readonly", "readonly");
  probe.style.position = "absolute";
  probe.style.left = "-9999px";
  document.body.appendChild(probe);
  probe.select();
  document.execCommand("copy");
  document.body.removeChild(probe);
}

function hasActiveTextSelection() {
  const selection = window.getSelection?.();
  return !!selection && !selection.isCollapsed && String(selection).trim().length > 0;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function queueEntry(eventId) {
  if (!eventId) return null;
  return (state.queue?.items || []).find((item) => item.event_id === eventId) || null;
}

function selectedQueueEntry() {
  return queueEntry(state.selectedTraceId)
    || (state.queue?.items || []).find((item) => item.status === "processing")
    || state.queue?.items?.[0]
    || null;
}

function elapsedSeconds(timestamp, endTimestamp = null) {
  if (!timestamp) return "-";
  const end = endTimestamp == null ? (Date.now() / 1000) : Number(endTimestamp);
  return `${Math.max(0, end - Number(timestamp)).toFixed(1)}s`;
}

function relativeSecondsAttrs(startTimestamp, endTimestamp = null) {
  if (!startTimestamp) return "";
  const attrs = [`data-elapsed-seconds-from="${escapeHtml(String(startTimestamp))}"`];
  if (endTimestamp != null) attrs.push(`data-elapsed-seconds-to="${escapeHtml(String(endTimestamp))}"`);
  return attrs.join(" ");
}

function relativeMsAttrs({ startTimestamp = null, fixedMs = null } = {}) {
  if (!startTimestamp && fixedMs == null) return "";
  const attrs = [];
  if (startTimestamp) attrs.push(`data-elapsed-ms-from="${escapeHtml(String(startTimestamp))}"`);
  if (fixedMs != null) attrs.push(`data-elapsed-ms-fixed="${escapeHtml(String(fixedMs))}"`);
  return attrs.join(" ");
}

function stageLabel(stageId) {
  if (!stageId) return "Waiting";
  if (stageId === "queued") return "Queued";
  return STAGE_META[stageId]?.title || String(stageId).replaceAll("_", " ");
}

function llmPhaseLabel(phase) {
  return String(phase || "idle").replaceAll("_", " ");
}

function queueItemStuck(item) {
  if (!item || item.status !== "processing") return false;
  const stageStarted = item.stage_started_ts || item.started_ts;
  if (!stageStarted) return false;
  const seconds = (Date.now() / 1000) - Number(stageStarted);
  return seconds > (item.current_stage === "llm_reasoner" ? 8 : 5);
}

function fmtMs(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(1)} ms`;
}

function fmtCount(value) {
  if (value == null || Number.isNaN(value)) return "0";
  return String(value);
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function fmtIso(value) {
  if (!value) return "unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function truncate(value, max = 88) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function eventTitle(event) {
  if (!event) return "No event";
  const data = event.data || {};
  if (event.type === "speech_text") return data.text ? truncate(data.text, 62) : "Speech event";
  if (event.type === "action_detected") return `${data.action || "Unknown"} detected`;
  if (event.type === "action_changed") return `${data.to || data.action || "Unknown"} committed`;
  if (event.type === "speaker_active") return `${data.who || "Unknown"} active`;
  if (event.type === "speaker_silent") return `${data.who || "Unknown"} silent`;
  if (event.type === "person_left") return `${data.who || "Unknown"} left`;
  if (event.type === "person_entered") return `${data.who || "Unknown"} entered`;
  return event.type.replaceAll("_", " ");
}

function previewData(data) {
  if (!data || !Object.keys(data).length) return "No payload";
  return Object.entries(data)
    .slice(0, 4)
    .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`)
    .join(" • ");
}

function layersByName(trace) {
  const grouped = new Map();
  for (const layer of trace?.layers || []) {
    if (!grouped.has(layer.layer)) grouped.set(layer.layer, []);
    grouped.get(layer.layer).push(layer);
  }
  return grouped;
}

function lastLayer(trace, name) {
  const grouped = layersByName(trace);
  const list = grouped.get(name) || [];
  return list[list.length - 1] || null;
}

function allLayers(trace, name) {
  return layersByName(trace).get(name) || [];
}

function rulePermissionRank(permission) {
  return { auto: 0, notify: 1, ask: 2, suggest: 3 }[permission] ?? 9;
}

function ruleTriggerRank(triggerType) {
  return {
    action_changed: 0,
    action_detected: 1,
    person_left: 2,
    person_entered: 3,
    speaker_active: 4,
    speaker_silent: 5,
    speech_text: 6,
    system_status: 7,
  }[triggerType] ?? 9;
}

function smartArea(rule) {
  const params = rule.action?.params || {};
  if (params.area) return String(params.area);
  if (params.scene) return String(params.scene);
  const command = String(rule.action?.command || "");
  if (command.includes("light")) return "lighting";
  if (command.includes("scene")) return "scenes";
  if (command.includes("vent")) return "ventilation";
  return "general";
}

function sortRulesForSmartHome(rules) {
  return [...rules].sort((a, b) => {
    if (!!a.active !== !!b.active) return a.active ? -1 : 1;
    const area = smartArea(a).localeCompare(smartArea(b));
    if (area) return area;
    const trigger = ruleTriggerRank(a.trigger?.type) - ruleTriggerRank(b.trigger?.type);
    if (trigger) return trigger;
    const permission = rulePermissionRank(a.permission) - rulePermissionRank(b.permission);
    if (permission) return permission;
    return String(a.description || a.id).localeCompare(String(b.description || b.id));
  });
}

function groupRules(rules) {
  const grouped = new Map();
  for (const rule of sortRulesForSmartHome(rules)) {
    const key = smartArea(rule);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(rule);
  }
  return grouped;
}

function primaryTrace() {
  if (!state.selectedTraceId) return null;
  return state.traceMap.get(state.selectedTraceId) || null;
}

function setSelectedTrace(traceId, stageId = state.selectedStageId, optionId = null, openDetail = false) {
  if (traceId) state.selectedTraceId = traceId;
  state.selectedStageId = stageId;
  state.selectedOptionId = optionId;
  if (openDetail) state.detailOpen = true;
  rebuildStageModels();
  renderAll();
  if (primaryTrace()) animateTrace(false);
}

function statusTone(active, warning = false) {
  if (warning) return "warn";
  return active ? "live" : "idle";
}

function applySnapshot(payload) {
  state.status = payload.status || state.status;
  state.world = payload.world || state.world;
  state.rules = payload.rules || state.rules;
  state.queue = payload.queue || state.queue;
  state.actions = payload.actions || state.actions;
  state.pending = payload.pending || state.pending;
  state.fireHistory = payload.fire_history || state.fireHistory;
  state.traceMap.clear();
  state.eventMap.clear();
  for (const trace of payload.traces || []) upsertTrace(trace);
  for (const event of payload.events || []) upsertEvent(event);
  if (state.selectedTraceId && !state.traceMap.has(state.selectedTraceId)) {
    state.selectedTraceId = null;
  }
  if (!state.selectedTraceId) {
    state.selectedTraceId = null;
    state.selectedStageId = "start";
    state.selectedOptionId = null;
    state.detailOpen = false;
  }
  rebuildStageModels();
}

function upsertTrace(trace) {
  const id = trace?.event?.event_id;
  if (!id) return;
  state.traceMap.set(id, trace);
  state.traces = Array.from(state.traceMap.values())
    .sort((a, b) => (b.start_time || 0) - (a.start_time || 0))
    .slice(0, 80);
}

function upsertEvent(event) {
  const id = event?.event_id;
  if (!id) return;
  state.eventMap.set(id, event);
  state.events = Array.from(state.eventMap.values())
    .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
    .slice(0, 120);
}

function makePendingTrace(event) {
  return {
    event,
    start_time: event?.timestamp || Date.now() / 1000,
    total_ms: 0,
    layers: [
      {
        layer: "event_bus",
        status: "received",
        details: `Event: ${event?.type || "unknown"}`,
        data: event || {},
        timestamp: event?.timestamp || Date.now() / 1000,
        elapsed_ms: 0,
      },
    ],
    final_decision: "pending",
    provisional: true,
  };
}

function metricChip(label, value) {
  return `<div class="metric-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function makeOption(id, title, body, extra = {}) {
  return {
    id,
    title,
    body,
    meta: extra.meta || [],
    chips: extra.chips || [],
    details: extra.details || [],
    selected: !!extra.selected,
    secondary: !!extra.secondary,
    raw: extra.raw || {},
  };
}

function eventLabel(type) {
  return EVENT_OPTIONS.find((item) => item.id === type)?.label || String(type || "unknown").replaceAll("_", " ");
}

function emptyStage(stageId, message) {
  const summary = message || `Waiting for an event to reach ${STAGE_META[stageId].title.toLowerCase()}.`;
  return {
    id: stageId,
    tone: STAGE_META[stageId].tone,
    status: "waiting",
    summary,
    primaryOptionId: null,
    options: [],
    emptyMessage: summary,
  };
}

function describeCondition(condition) {
  if (Array.isArray(condition?.time_of_day)) return `time is ${condition.time_of_day.join(" or ")}`;
  if (condition?.people_present_any === true) return "someone is home";
  if (condition?.people_present_count != null) return `people present = ${condition.people_present_count}`;
  return Object.entries(condition || {})
    .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join(", ") : value}`)
    .join(", ");
}

function summarizeConditions(conditions) {
  if (!conditions?.length) return "No extra conditions.";
  return conditions.map((item) => describeCondition(item)).join(" • ");
}

function ruleResultsForTrace(trace) {
  const results = [];
  for (const layer of allLayers(trace, "rule_engine")) {
    for (const result of layer?.data?.results || []) results.push(result);
  }
  return results;
}

function ruleMutationLayers(trace) {
  return allLayers(trace, "rule_engine").filter((layer) => !Array.isArray(layer?.data?.results) && String(layer?.status || "").startsWith("rule_"));
}

function decisionLabel(decision) {
  const value = String(decision || "");
  if (value === "conditions_not_met") return "Conditions Failed";
  if (value === "cooldown") return "Blocked By Cooldown";
  if (value.startsWith("fire_")) return `Fired · ${value.slice(5)}`;
  return statusLabel(value);
}

function decisionTone(decision) {
  const value = String(decision || "");
  if (value.startsWith("fire_")) return "gold";
  if (value === "conditions_not_met" || value === "cooldown") return "red";
  return "blue";
}

function emphasizePrimary(options, preferredId = null) {
  if (!options.length) return null;
  const primary = options.find((item) => item.id === preferredId)
    || options.find((item) => item.selected)
    || options[0];
  for (const item of options) {
    item.selected = item.id === primary.id;
    item.secondary = item.id !== primary.id && !String(item.id).startsWith("empty:");
  }
  return primary;
}

function explainRuleResult(result, rule) {
  const trigger = ruleSummary(rule);
  const conditions = summarizeConditions(rule?.trigger?.conditions || []);
  if (result?.decision?.startsWith("fire_")) {
    return `Matched ${trigger}. ${conditions} Permission route: ${result.permission || rule?.permission || "ask"}.`;
  }
  if (result?.decision === "cooldown") {
    return `Matched ${trigger}, but execution stopped because the rule is cooling down. ${result.details || ""}`.trim();
  }
  if (result?.decision === "conditions_not_met") {
    return `Matched ${trigger}, but the required conditions were not satisfied. ${conditions}`;
  }
  return result?.details || trigger;
}

function buildStartStage(trace) {
  const event = trace?.event || null;
  if (!event) return emptyStage("start", "No event in flight yet.");
  const option = makeOption(
    `event:${event.event_id}`,
    eventLabel(event.type),
    previewData(event.data),
    {
      selected: true,
      meta: [fmtIso(event.iso_time), event.event_id],
      chips: [{ label: event.type, tone: "gold" }],
      details: [
        { label: "Event Type", value: eventLabel(event.type) },
        { label: "When", value: fmtIso(event.iso_time) },
        { label: "Trace ID", value: event.event_id },
        { label: "Payload", value: previewData(event.data) },
      ],
      raw: event,
    },
  );
  return {
    id: "start",
    tone: STAGE_META.start.tone,
    status: event.type,
    summary: eventTitle(event),
    primaryOptionId: option.id,
    options: [option],
  };
}

function buildEventBusStage(trace) {
  if (!trace?.event) return emptyStage("event_bus");
  const event = trace.event;
  const busLayer = lastLayer(trace, "event_bus");
  const worldLayer = lastLayer(trace, "world_state");
  const derivedEvents = worldLayer?.data?.derived_events || [];
  const options = [
    makeOption(
      "accepted",
      "Accepted Into Brain",
      busLayer?.details || "The event entered the bus and was handed to the decision loop.",
      {
        selected: true,
        meta: [eventLabel(event.type), fmtIso(event.iso_time)],
        chips: [{ label: "primary route", tone: "gold" }],
        details: [
          { label: "Incoming Event", value: eventLabel(event.type) },
          { label: "Accepted At", value: fmtIso(event.iso_time) },
          { label: "Derived Count", value: String(derivedEvents.length) },
        ],
        raw: busLayer || event,
      },
    ),
  ];
  for (const derived of derivedEvents) {
    options.push(makeOption(
      `derived:${derived}`,
      `Derived Route · ${eventLabel(derived)}`,
      `World-state emitted ${eventLabel(derived)} as a follow-up route from this same event.`,
      {
        secondary: true,
        meta: ["follow-up event"],
        chips: [{ label: derived, tone: "blue" }],
        details: [
          { label: "Source Event", value: eventLabel(event.type) },
          { label: "Derived Event", value: eventLabel(derived) },
          { label: "Why", value: "World-state logic turned the original event into an additional bus event." },
        ],
        raw: { source_event: event, derived_event: derived },
      },
    ));
  }
  const primary = emphasizePrimary(options, "accepted");
  return {
    id: "event_bus",
    tone: STAGE_META.event_bus.tone,
    status: busLayer?.status || "received",
    summary: derivedEvents.length
      ? `${busLayer?.details || "Event accepted."} ${derivedEvents.length} follow-up route${derivedEvents.length === 1 ? "" : "s"} emitted.`
      : (busLayer?.details || STAGE_META.event_bus.summary),
    primaryOptionId: primary?.id || null,
    options,
  };
}

function buildWorldStage(trace) {
  if (!trace?.event) return emptyStage("world_state");
  const event = trace.event;
  const worldLayer = lastLayer(trace, "world_state");
  const derivedEvents = worldLayer?.data?.derived_events || [];
  const world = state.world || {};
  const presenceActive = ["speaker_active", "speaker_silent", "person_left", "person_entered"].includes(event.type);
  const speechActive = event.type === "speech_text";
  const actionActive = event.type === "action_detected" || event.type === "action_changed";
  const systemActive = event.type === "system_status";
  const options = [];
  if (presenceActive) {
    const people = Object.keys(world.people_present || {});
    options.push(makeOption(
      "presence",
      "Presence Tracker",
      `${event.data?.who || "A person"} changed presence state. People home now: ${people.join(", ") || "nobody"}.`,
      {
        meta: [`count ${fmtCount(world.people_count || 0)}`],
        chips: [{ label: event.type, tone: "gold" }],
        details: [
          { label: "Event", value: eventLabel(event.type) },
          { label: "Person", value: event.data?.who || "unknown" },
          { label: "People Home", value: people.join(", ") || "nobody" },
          { label: "Count", value: fmtCount(world.people_count || 0) },
        ],
        raw: { event, people_present: world.people_present, people_count: world.people_count },
      },
    ));
  }
  if (speechActive) {
    options.push(makeOption(
      "speech_memory",
      "Speech Memory Updated",
      "This transcript was stored as the latest utterance and added to recent speech context.",
      {
        meta: [truncate(event.data?.text || "", 52)],
        chips: [{ label: "speech context", tone: "blue" }],
        details: [
          { label: "Speaker", value: event.data?.who || "unknown" },
          { label: "Transcript", value: event.data?.text || "(empty)" },
          { label: "Last Speech", value: truncate(world.last_speech || "", 120) || "none" },
        ],
        raw: { event, last_speech: world.last_speech, recent_speech: world.recent_speech },
      },
    ));
  }
  if (actionActive) {
    options.push(makeOption(
      "action_tracker",
      "Action Tracker Updated",
      `Action state moved through debounce and now reads ${world.current_action || "none"}.`,
      {
        meta: [world.current_action || "none", fmtPct(world.action_confidence || 0)],
        chips: [{ label: "action state", tone: "blue" }],
        details: [
          { label: "Incoming Action", value: event.data?.to || event.data?.action || "unknown" },
          { label: "Stable Action", value: world.current_action || "none" },
          { label: "Confidence", value: fmtPct(world.action_confidence || 0) },
        ],
        raw: {
          event,
          current_action: world.current_action,
          action_confidence: world.action_confidence,
          recent_actions: world.recent_actions,
        },
      },
    ));
  }
  if (systemActive) {
    options.push(makeOption(
      "runtime_flags",
      "Runtime Flags Updated",
      "The event changed runtime state used by later decisions.",
      {
        meta: [`voice ${world.voice_running ? "on" : "off"}`, `vision ${world.vision_running ? "on" : "off"}`],
        chips: [{ label: "runtime", tone: "blue" }],
        details: [
          { label: "Voice Runtime", value: world.voice_running ? "on" : "off" },
          { label: "Vision Runtime", value: world.vision_running ? "on" : "off" },
        ],
        raw: { event, voice_running: world.voice_running, vision_running: world.vision_running },
      },
    ));
  }
  if (derivedEvents.length) {
    options.push(makeOption(
      "derived",
      "Derived Events Created",
      `World-state produced ${derivedEvents.map((item) => eventLabel(item)).join(", ")} from the original event.`,
      {
        meta: [`${derivedEvents.length} derived`],
        chips: [{ label: "follow-up", tone: "gold" }],
        details: [
          { label: "Source Event", value: eventLabel(event.type) },
          { label: "Derived Events", value: derivedEvents.map((item) => eventLabel(item)).join(", ") },
        ],
        raw: { event, derived_events: derivedEvents },
      },
    ));
  }
  if (!options.length) {
    const option = makeOption(
      "state_check",
      "State Checked",
      worldLayer?.details || "World-state inspected the event, but no visible tracker changed.",
      {
        selected: true,
        details: [
          { label: "Event", value: eventLabel(event.type) },
          { label: "Outcome", value: "No visible world-state lane changed for this event." },
        ],
        raw: { event, world_layer: worldLayer },
      },
    );
    return {
      id: "world_state",
      tone: STAGE_META.world_state.tone,
      status: worldLayer?.status || "updated",
      summary: worldLayer?.details || STAGE_META.world_state.summary,
      primaryOptionId: option.id,
      options: [option],
    };
  }
  let primaryOptionId = options[0].id;
  if (speechActive) primaryOptionId = "speech_memory";
  if (actionActive) primaryOptionId = "action_tracker";
  if (systemActive) primaryOptionId = "runtime_flags";
  const primary = emphasizePrimary(options, primaryOptionId);

  return {
    id: "world_state",
    tone: STAGE_META.world_state.tone,
    status: worldLayer?.status || "updated",
    summary: worldLayer?.details || STAGE_META.world_state.summary,
    primaryOptionId: primary?.id || null,
    options,
  };
}

function relevantTriggerTypes(trace) {
  const types = new Set();
  if (trace?.event?.type) types.add(trace.event.type);
  const derivedEvents = lastLayer(trace, "world_state")?.data?.derived_events || [];
  for (const item of derivedEvents) types.add(item);
  return types;
}

function ruleSummary(rule) {
  const trigger = rule.trigger || {};
  if (trigger.action) return `${trigger.type} → ${trigger.action}`;
  if (Array.isArray(trigger.action_in) && trigger.action_in.length) return `${trigger.type} → ${trigger.action_in.join(", ")}`;
  if (trigger.who) return `${trigger.type} → ${trigger.who}`;
  return trigger.type || "unknown";
}

function ruleActionSummary(rule) {
  const action = rule.action || {};
  const params = action.params || {};
  const paramSummary = Object.entries(params)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
  return `${action.command || "noop"}${paramSummary ? ` (${paramSummary})` : ""}`;
}

function buildRuleStage(trace) {
  if (!trace?.event) return emptyStage("rule_engine");
  const ruleLayers = allLayers(trace, "rule_engine");
  const results = ruleResultsForTrace(trace);
  const mutations = ruleMutationLayers(trace);
  const triggerTypes = relevantTriggerTypes(trace);
  const options = [];
  let preferredId = null;

  for (const [index, result] of results.entries()) {
    const rule = state.rules.find((item) => item.id === result.rule_id) || {
      id: result.rule_id,
      description: result.rule_description || result.rule_id,
      trigger: { type: result.event?.type },
      action: result.action || {},
      permission: result.permission,
    };
    const optionId = `rule:${result.rule_id}:${index}`;
    if (!preferredId && String(result.decision || "").startsWith("fire_")) preferredId = optionId;
    options.push(makeOption(
      optionId,
      rule.description || rule.id,
      explainRuleResult(result, rule),
      {
        meta: [
          decisionLabel(result.decision),
          smartArea(rule),
          ruleActionSummary(rule),
        ],
        chips: [
          { label: decisionLabel(result.decision), tone: decisionTone(result.decision) },
          { label: rule.trigger?.type || result.event?.type || "trigger", tone: "blue" },
          ...(result.permission ? [{ label: result.permission, tone: "gold" }] : []),
        ],
        details: [
          { label: "Decision", value: decisionLabel(result.decision) },
          { label: "Triggered By", value: eventLabel(result.event?.type) },
          { label: "Trigger Match", value: ruleSummary(rule) },
          { label: "Conditions", value: summarizeConditions(rule.trigger?.conditions || []) },
          { label: "Action", value: ruleActionSummary(rule) },
          { label: "Reason", value: result.details || explainRuleResult(result, rule) },
        ],
        raw: { rule, result },
      },
    ));
  }

  for (const [index, layer] of mutations.entries()) {
    const rule = layer.data || {};
    const optionId = `mutation:${layer.status}:${index}`;
    if (!preferredId) preferredId = optionId;
    options.push(makeOption(
      optionId,
      rule.description || statusLabel(layer.status),
      layer.details || "Rule library changed during this trace.",
      {
        meta: [statusLabel(layer.status), rule.id || "speech driven"],
        chips: [{ label: statusLabel(layer.status), tone: "blue" }],
        details: [
          { label: "Change", value: statusLabel(layer.status) },
          { label: "Rule", value: rule.description || rule.id || "rule mutation" },
          { label: "Trigger", value: ruleSummary(rule) },
          { label: "Action", value: ruleActionSummary(rule) },
        ],
        raw: { layer, rule },
      },
    ));
  }

  if (!options.length) {
    const option = makeOption(
      "rule:none",
      "No Rule Used",
      "No active rule matched this event or any follow-up event created from it.",
      {
        selected: true,
        details: [
          { label: "Checked Trigger Types", value: Array.from(triggerTypes).map((item) => eventLabel(item)).join(", ") || "none" },
          { label: "Outcome", value: "Nothing moved from the rule engine into the executor." },
        ],
        raw: { trigger_types: Array.from(triggerTypes) },
      },
    );
    return {
      id: "rule_engine",
      tone: STAGE_META.rule_engine.tone,
      status: "no_match",
      summary: "No rule used on this event.",
      primaryOptionId: option.id,
      options: [option],
    };
  }

  const primary = emphasizePrimary(options, preferredId);
  const lastRuleLayer = ruleLayers[ruleLayers.length - 1] || null;
  return {
    id: "rule_engine",
    tone: STAGE_META.rule_engine.tone,
    status: lastRuleLayer?.status || "evaluated",
    summary: results.length
      ? `${results.length} rule path${results.length === 1 ? "" : "s"} were touched on this event.`
      : (lastRuleLayer?.details || STAGE_META.rule_engine.summary),
    primaryOptionId: primary?.id || null,
    options,
  };
}

function buildReasonerStage(trace) {
  if (!trace?.event) return emptyStage("llm_reasoner");
  const llmLayers = allLayers(trace, "llm_reasoner");
  const llmLast = llmLayers[llmLayers.length - 1] || null;
  const llmStatus = state.status?.llm || {};
  const liveQueue = queueEntry(trace.event.event_id);
  const liveLlm = liveQueue?.llm || {};
  const event = trace.event;
  if (event.type !== "speech_text") {
    const option = makeOption(
      "llm:skipped",
      "Skipped",
      "This event never needed the LLM. The route stayed inside deterministic systems.",
      {
        selected: true,
        details: [
          { label: "Event", value: eventLabel(event.type) },
          { label: "Reason", value: "Only speech traces enter the reasoner right now." },
        ],
        raw: { event, llm_status: llmStatus },
      },
    );
    return {
      id: "llm_reasoner",
      tone: STAGE_META.llm_reasoner.tone,
      status: "bypassed",
      summary: "This trace skipped the LLM.",
      primaryOptionId: option.id,
      options: [option],
    };
  }
  if (liveLlm.active || liveLlm.stream_text || liveLlm.latest_output) {
    const liveText = liveLlm.stream_text || liveLlm.latest_output || "";
    const latency = liveLlm.active ? ((Date.now() / 1000) - Number(liveLlm.phase_started_ts || Date.now() / 1000)) * 1000 : liveLlm.latency_ms;
    const option = makeOption(
      `llm-live:${liveLlm.phase || "stream"}`,
      liveLlm.active ? `Streaming · ${llmPhaseLabel(liveLlm.phase)}` : llmPhaseLabel(liveLlm.phase || "complete"),
      liveText ? truncate(liveText, 220) : "Waiting for the LLM to emit output.",
      {
        selected: true,
        meta: [llmStatus.model || "model unknown", fmtMs(latency)],
        chips: [
          { label: liveLlm.active ? "streaming" : (liveLlm.status || "complete"), tone: liveLlm.active ? "blue" : "gold" },
          ...(liveLlm.phase ? [{ label: llmPhaseLabel(liveLlm.phase), tone: "gold" }] : []),
        ],
        details: [
          { label: "Transcript", value: trace.event.data?.text || "(empty)" },
          { label: "Phase", value: llmPhaseLabel(liveLlm.phase) },
          { label: "Elapsed", value: fmtMs(latency) },
          { label: "Chunks", value: fmtCount(liveLlm.chunks || 0) },
          { label: "Live Output", value: liveText || "(no tokens yet)" },
        ],
        raw: { event, llm_live: liveLlm, llm_layer: llmLast, llm_status: llmStatus },
      },
    );
    return {
      id: "llm_reasoner",
      tone: STAGE_META.llm_reasoner.tone,
      status: liveLlm.active ? "streaming" : (liveLlm.status || llmLast?.status || "complete"),
      summary: liveLlm.active ? `Streaming ${llmPhaseLabel(liveLlm.phase)}.` : "Latest LLM output is available.",
      primaryOptionId: option.id,
      options: [option],
    };
  }
  const selectedIntent = llmLast?.data?.intent || (llmStatus.available ? "waiting" : "offline");
  const intentMeta = INTENT_OPTIONS.find((item) => item.id === llmLast?.data?.intent);
  const title = llmLast
    ? (intentMeta?.label || eventLabel(llmLast?.data?.intent))
    : (llmStatus.available ? "Pending Classification" : "LLM Offline");
  const body = llmLast?.details || (llmStatus.available
    ? "Speech arrived, but the intent classification result has not landed yet."
    : "Speech can reach the reasoner, but the LLM is offline.");
  const option = makeOption(
    `llm:${selectedIntent}`,
    title,
    body,
    {
      selected: true,
      meta: [llmStatus.model || "model unknown", fmtMs(llmStatus.last_latency_ms)],
      chips: llmLast?.data?.intent ? [{ label: llmLast.data.intent, tone: "blue" }] : [],
      details: [
        { label: "Transcript", value: trace.event.data?.text || "(empty)" },
        { label: "Intent", value: llmLast?.data?.intent || "not classified yet" },
        { label: "Confidence", value: llmLast?.data?.confidence != null ? fmtPct(llmLast.data.confidence) : "-" },
        { label: "Model", value: llmStatus.model || "unknown" },
        { label: "Latency", value: fmtMs(llmStatus.last_latency_ms) },
      ],
      raw: { event, llm_layer: llmLast, llm_status: llmStatus },
    },
  );
  return {
    id: "llm_reasoner",
    tone: STAGE_META.llm_reasoner.tone,
    status: llmLast?.status || (llmStatus.available ? "waiting" : "offline"),
    summary: body,
    primaryOptionId: option.id,
    options: [option],
  };
}

function buildExecutorStage(trace) {
  if (!trace?.event) return emptyStage("executor");
  const executorLayers = allLayers(trace, "executor");
  const ruleResults = ruleResultsForTrace(trace);
  const fired = ruleResults.find((item) => String(item.decision || "").startsWith("fire_"));
  const execLast = executorLayers[executorLayers.length - 1] || null;
  const execData = execLast?.data || {};
  const options = [];

  if (execLast) {
    let routeTitle = "Executor Route";
    if (["executed", "executed_notified", "direct_executed", "confirmed"].includes(execLast.status)) routeTitle = "Executed";
    if (execLast.status === "asked") routeTitle = "Waiting For Confirmation";
    if (execLast.status === "suggested") routeTitle = "Suggestion Sent";
    if (execLast.status === "rejected") routeTitle = "Rejected";
    options.push(makeOption(
      `exec:${execLast.status}`,
      routeTitle,
      execLast.details || "The executor handled the final action route for this trace.",
      {
        selected: true,
        meta: [statusLabel(execLast.status)],
        chips: fired?.permission ? [{ label: fired.permission, tone: "gold" }] : [],
        details: [
          { label: "Route", value: statusLabel(execLast.status) },
          { label: "Rule", value: fired?.rule_description || execData.rule_description || execData.rule_id || "-" },
          { label: "Reason", value: execLast.details || "Executor handled this route." },
        ],
        raw: { layer: execLast, fired },
      },
    ));

    if (execData.action?.command || execData.action_type) {
      options.push(makeOption(
        `output:${execData.action?.command || execData.action_type}`,
        execData.action?.command || execData.action_type || "Output Command",
        execData.details || execData.rule_description || "Final executor payload.",
        {
          meta: [execData.status || "-", execData.executed_at || "-"],
          chips: execData.rule_id ? [{ label: execData.rule_id, tone: "blue" }] : [],
          details: [
            { label: "Command", value: execData.action?.command || execData.action_type || "-" },
            { label: "Status", value: execData.status || "-" },
            { label: "Executed At", value: execData.executed_at || "-" },
            { label: "Payload", value: JSON.stringify(execData.action || {}) },
          ],
          raw: execData,
        },
      ));
    }
  } else if (fired) {
    options.push(makeOption(
      `permission:${fired.permission || "ask"}`,
      `Permission Route · ${fired.permission || "ask"}`,
      `The rule fired with ${fired.permission || "ask"} permission, but no executor layer was recorded yet.`,
      {
        selected: true,
        details: [
          { label: "Rule", value: fired.rule_description || fired.rule_id || "-" },
          { label: "Permission", value: fired.permission || "ask" },
          { label: "Action", value: ruleActionSummary({ action: fired.action || {} }) },
        ],
        raw: fired,
      },
    ));
  } else {
    options.push(makeOption(
      "executor:none",
      "No Action Sent",
      "This trace stopped before the executor needed to do anything.",
      {
        selected: true,
        details: [
          { label: "Outcome", value: "No action left the brain for this event." },
        ],
        raw: { trace_id: trace.event.event_id },
      },
    ));
  }

  const primary = emphasizePrimary(options, options[0]?.id || null);
  return {
    id: "executor",
    tone: STAGE_META.executor.tone,
    status: execLast?.status || (fired?.permission ? `fire_${fired.permission}` : "no_output"),
    summary: execLast?.details || options[0]?.body || STAGE_META.executor.summary,
    primaryOptionId: primary?.id || null,
    options,
  };
}

function rebuildStageModels() {
  const trace = primaryTrace();
  state.stageModels = {
    start: buildStartStage(trace),
    event_bus: buildEventBusStage(trace),
    world_state: buildWorldStage(trace),
    rule_engine: buildRuleStage(trace),
    llm_reasoner: buildReasonerStage(trace),
    executor: buildExecutorStage(trace),
  };

  if (!state.selectedOptionId) {
    state.selectedOptionId = state.stageModels[state.selectedStageId]?.primaryOptionId || null;
  } else {
    const exists = state.stageModels[state.selectedStageId]?.options?.some((item) => item.id === state.selectedOptionId);
    if (!exists) {
      state.selectedOptionId = state.stageModels[state.selectedStageId]?.primaryOptionId || null;
    }
  }
}

function statusLabel(status) {
  return String(status || "idle").replaceAll("_", " ");
}

function stageStatusTone(stageId, stageModel) {
  const status = String(stageModel.status || "idle");
  if (status === "offline" || status === "asked") return "warn";
  if (["waiting", "bypassed", "no_output", "no_match"].includes(status)) return "idle";
  return stageModel.options.length ? "live" : "idle";
}

function buildStageCards() {
  if (state.stageOrderRendered) return;
  const workspace = byId("workspace");
  for (const stageId of STAGE_ORDER) {
    const stage = document.createElement("article");
    stage.className = "stage-card";
    stage.dataset.stage = stageId;
    stage.dataset.tone = STAGE_META[stageId].tone;
    stage.innerHTML = `
      <div class="stage-head">
        <div>
          <p class="stage-kicker">${escapeHtml(STAGE_META[stageId].kicker)}</p>
          <h3 class="stage-title">${escapeHtml(STAGE_META[stageId].title)}</h3>
          <p class="stage-copy"></p>
        </div>
        <span class="stage-status" data-state="idle">idle</span>
      </div>
      <div class="stage-options"></div>
    `;
    workspace.appendChild(stage);
  }
  state.stageOrderRendered = true;
}

function renderStageCards() {
  for (const stageId of STAGE_ORDER) {
    const el = document.querySelector(`.stage-card[data-stage="${stageId}"]`);
    const model = state.stageModels[stageId];
    if (!el || !model) continue;
    el.dataset.lit = "false";
    el.classList.toggle("is-empty", !model.options.length);
    el.querySelector(".stage-copy").textContent = model.summary;
    const statusEl = el.querySelector(".stage-status");
    statusEl.textContent = statusLabel(model.status);
    statusEl.dataset.state = stageStatusTone(stageId, model);
    const optionsRoot = el.querySelector(".stage-options");
    optionsRoot.innerHTML = "";
    if (!model.options.length) {
      optionsRoot.innerHTML = `<div class="stage-empty">${escapeHtml(model.emptyMessage || "Waiting for a live event.")}</div>`;
      continue;
    }
    for (const option of model.options) {
      const optionEl = document.createElement("article");
      const isDrawerSelected = state.selectedStageId === stageId && state.selectedOptionId === option.id;
      optionEl.className = `option-card${option.selected ? " is-selected" : ""}${option.secondary ? " is-secondary" : ""}${isDrawerSelected ? " is-drawer-selected" : ""}`;
      optionEl.dataset.option = option.id;
      optionEl.innerHTML = `
        <h4>${escapeHtml(option.title)}</h4>
        <p>${escapeHtml(option.body)}</p>
        ${option.meta.length ? `<div class="option-meta">${escapeHtml(option.meta.join(" • "))}</div>` : ""}
        ${option.chips.length ? `
          <div class="option-flag-row">
            ${option.chips.map((chip) => `<span class="chip" data-tone="${escapeHtml(chip.tone || "gold")}">${escapeHtml(chip.label)}</span>`).join("")}
          </div>
        ` : ""}
      `;
      optionEl.addEventListener("click", (event) => {
        event.stopPropagation();
        state.selectedStageId = stageId;
        state.selectedOptionId = option.id;
        state.detailOpen = true;
        renderAll();
      });
      optionsRoot.appendChild(optionEl);
    }
  }
}

function selectedStageOption(stageId) {
  const model = state.stageModels[stageId];
  if (!model) return null;
  const chosenId = state.selectedStageId === stageId ? state.selectedOptionId : model.primaryOptionId;
  return model.options.find((item) => item.id === chosenId)
    || model.options.find((item) => item.selected)
    || model.options.find((item) => item.secondary)
    || model.options[0]
    || null;
}

function renderStatusStrip() {
  const status = state.status || {};
  const world = status.world_state || state.world || {};
  const llm = status.llm || {};
  const rules = status.rules || {};
  const queue = state.queue || {};
  byId("status-strip").innerHTML = `
    <span class="status-chip" data-state="${statusTone(status.running)}">Brain <strong>${status.running ? "Live" : "Idle"}</strong></span>
    <span class="status-chip" data-state="${statusTone(world.voice_running)}">Voice <strong>${world.voice_running ? "On" : "Off"}</strong></span>
    <span class="status-chip" data-state="${statusTone(world.vision_running)}">Vision <strong>${world.vision_running ? "On" : "Off"}</strong></span>
    <span class="status-chip" data-state="${statusTone(!!llm.available, !llm.available)}">LLM <strong>${llm.available ? "Ready" : "Offline"}</strong></span>
    <span class="status-chip" data-state="${queue.pending_count ? "warn" : "idle"}">Queue <strong>${fmtCount(queue.pending_count || 0)} waiting</strong></span>
    <span class="status-chip" data-state="${statusTone(state.streamConnected, !state.streamConnected)}">Stream <strong>${state.streamConnected ? "Linked" : "Retrying"}</strong></span>
    <span class="status-chip" data-state="${(rules.pending_confirmations || 0) ? "warn" : "idle"}">Pending <strong>${fmtCount(rules.pending_confirmations || 0)}</strong></span>
  `;
}

function renderHeroMetrics() {
  const status = state.status || {};
  const llm = status.llm || {};
  const exec = status.executor || {};
  byId("hero-metrics").innerHTML = [
    metricChip("Events", fmtCount(status.events_processed || 0)),
    metricChip("Rules Fired", fmtCount(status.rules_fired || 0)),
    metricChip("LLM Calls", fmtCount(status.llm_calls || llm.call_count || 0)),
    metricChip("Actions", fmtCount(exec.total_actions || 0)),
  ].join("");
}

function renderProgress() {
  const trace = primaryTrace();
  const title = byId("progress-title");
  const steps = byId("progress-steps");
  if (!trace) {
    title.textContent = "Waiting for a live event";
    steps.innerHTML = "";
    return;
  }
  title.textContent = eventTitle(trace.event);
  steps.innerHTML = "";
  for (const stageId of SYSTEM_ORDER) {
    const model = state.stageModels[stageId];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `progress-step${state.selectedStageId === stageId ? " is-active" : ""}`;
    const chosen = selectedStageOption(stageId);
    button.textContent = `${STAGE_META[stageId].title} · ${chosen?.title || statusLabel(model.status)}`;
    button.addEventListener("click", () => {
      state.selectedStageId = stageId;
      state.selectedOptionId = chosen?.id || model.primaryOptionId;
      state.detailOpen = true;
      renderAll();
    });
    steps.appendChild(button);
  }
}

function traceStageSummary(trace, stageId) {
  const currentModels = {
    start: buildStartStage(trace),
    event_bus: buildEventBusStage(trace),
    world_state: buildWorldStage(trace),
    rule_engine: buildRuleStage(trace),
    llm_reasoner: buildReasonerStage(trace),
    executor: buildExecutorStage(trace),
  };
  const stage = currentModels[stageId];
  const chosen = stage.options.find((item) => item.selected) || stage.options.find((item) => item.secondary) || stage.options[0];
  return chosen ? chosen.title : statusLabel(stage.status);
}

function renderTraceHistory() {
  const root = byId("trace-history");
  if (!state.traces.length) {
    root.innerHTML = '<div class="empty-state">No traces yet. Inject one from the quick actions or wait for the live runtime.</div>';
    return;
  }
  root.innerHTML = "";
  for (const trace of state.traces) {
    const selected = trace.event.event_id === state.selectedTraceId;
    const ruleResult = ruleResultsForTrace(trace);
    const card = document.createElement("article");
    card.className = `trace-card${selected ? " is-selected" : ""}`;
    card.innerHTML = `
      <div class="trace-head">
        <div>
          <p class="trace-kicker">${escapeHtml(trace.event.type)}</p>
          <h3 class="trace-title">${escapeHtml(eventTitle(trace.event))}</h3>
        </div>
        <span class="trace-chip">${escapeHtml(trace.final_decision || "pending")}</span>
      </div>
      <p>${escapeHtml(previewData(trace.event.data))}</p>
      <div class="trace-meta">
        <span>${escapeHtml(fmtIso(trace.event.iso_time))}</span>
        <span>${escapeHtml(fmtMs(trace.total_ms))}</span>
        <span>${escapeHtml(trace.event.event_id)}</span>
      </div>
      <div class="trace-chip-row">
        ${SYSTEM_ORDER.map((stageId) => `<span class="trace-chip">${escapeHtml(traceStageSummary(trace, stageId))}</span>`).join("")}
        ${ruleResult.length ? `<span class="trace-chip">${escapeHtml(`${ruleResult.length} rule result${ruleResult.length === 1 ? "" : "s"}`)}</span>` : ""}
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedTraceId = trace.event.event_id;
      state.selectedStageId = "start";
      state.selectedOptionId = null;
      state.detailOpen = false;
      rebuildStageModels();
      renderAll();
      animateTrace(false);
    });
    root.appendChild(card);
  }
}

function renderQuickActions() {
  const root = byId("quick-actions");
  root.innerHTML = "";
  for (const item of QUICK_ACTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-ghost btn-small";
    button.textContent = item.label;
    button.addEventListener("click", async () => {
      try {
        await item.handler();
      } catch (error) {
        setFormStatus(error.message || String(error), true);
      }
    });
    root.appendChild(button);
  }
}

function renderQueuePanel() {
  const root = byId("queue-list");
  const meta = byId("queue-meta");
  const queue = state.queue || {};
  const items = queue.items || [];
  meta.textContent = `${fmtCount(queue.active_count || 0)} active • ${fmtCount(queue.pending_count || 0)} queued • depth ${fmtCount(queue.queue_depth || 0)}`;
  if (!items.length) {
    root.innerHTML = '<div class="empty-state">No queued or running events right now.</div>';
    return;
  }
  root.innerHTML = "";
  for (const item of items.slice(0, 14)) {
    const selected = item.event_id === state.selectedTraceId;
    const stuck = queueItemStuck(item);
    const card = document.createElement("article");
    card.className = `queue-item${selected ? " is-selected" : ""}${item.status === "processing" ? " is-active" : ""}${stuck ? " is-stuck" : ""}`;
    card.dataset.queueStatus = item.status || "";
    card.dataset.currentStage = item.current_stage || "";
    if (item.stage_started_ts || item.started_ts) card.dataset.stageStartedTs = String(item.stage_started_ts || item.started_ts);
    const llm = item.llm || {};
    const endedTs = item.status === "processing" || item.status === "queued" ? null : (item.completed_ts || item.updated_ts || null);
    const stageTime = item.status === "queued"
      ? elapsedSeconds(item.queued_ts, endedTs)
      : elapsedSeconds(item.stage_started_ts || item.started_ts, endedTs);
    const statusTone = stuck ? "red" : (item.status === "processing" ? "blue" : item.status === "queued" ? "gold" : "gold");
    card.innerHTML = `
      <div class="queue-head">
        <div class="queue-copy">
          <strong>${escapeHtml(eventTitle(item.event))}</strong>
          <p>${escapeHtml(item.details || previewData(item.event?.data))}</p>
        </div>
        <span class="queue-pill" data-tone="${escapeHtml(statusTone)}">${escapeHtml(statusLabel(item.status))}</span>
      </div>
      <div class="queue-row">
        <span>Stage <strong>${escapeHtml(stageLabel(item.current_stage))}</strong></span>
        <span>Stage Time <strong ${item.status === "queued" ? relativeSecondsAttrs(item.queued_ts, endedTs) : relativeSecondsAttrs(item.stage_started_ts || item.started_ts, endedTs)}>${escapeHtml(stageTime)}</strong></span>
        ${item.started_ts ? `<span>Total <strong ${relativeSecondsAttrs(item.started_ts, endedTs)}>${escapeHtml(elapsedSeconds(item.started_ts, endedTs))}</strong></span>` : ""}
      </div>
      <div class="queue-pill-row">
        <span class="queue-pill" data-tone="gold">${escapeHtml(statusLabel(item.stage_status || item.status))}</span>
        ${llm.phase ? `<span class="queue-pill" data-tone="blue">${escapeHtml(llmPhaseLabel(llm.phase))}</span>` : ""}
        ${llm.phase_started_ts ? `<span class="queue-pill" data-tone="blue"><span ${relativeMsAttrs({ startTimestamp: llm.active ? llm.phase_started_ts : null, fixedMs: llm.active ? null : llm.latency_ms })}>${escapeHtml(fmtMs(llm.active ? ((Date.now() / 1000) - Number(llm.phase_started_ts)) * 1000 : llm.latency_ms))}</span></span>` : ""}
        <span class="queue-pill" data-tone="red" data-stuck-pill ${stuck ? "" : "hidden"}>watching for stall</span>
      </div>
    `;
    card.addEventListener("click", () => {
      if (!state.traceMap.has(item.event_id)) upsertTrace(makePendingTrace(item.event));
      state.selectedTraceId = item.event_id;
      state.selectedStageId = item.current_stage && STAGE_ORDER.includes(item.current_stage) ? item.current_stage : "start";
      state.selectedOptionId = null;
      state.detailOpen = true;
      rebuildStageModels();
      renderAll();
      if (item.current_stage && STAGE_ORDER.includes(item.current_stage)) showProgressThroughStage(item.current_stage);
    });
    root.appendChild(card);
  }
}

function renderLlmLivePanel() {
  const root = byId("llm-live");
  const meta = byId("llm-live-meta");
  const queueItem = selectedQueueEntry();
  const llm = queueItem?.llm || null;
  const trace = primaryTrace();
  const llmLayers = allLayers(trace, "llm_reasoner");
  const lastLlmLayer = llmLayers[llmLayers.length - 1] || null;

  if (!llm && !lastLlmLayer) {
    meta.textContent = "No active LLM work";
    root.innerHTML = '<div class="empty-state">Speech traces will stream the LLM output here while they run.</div>';
    return;
  }

  const liveText = llm?.stream_text || llm?.latest_output || lastLlmLayer?.data?._raw_response || JSON.stringify(lastLlmLayer?.data || {}, null, 2);
  const liveLatency = llm?.phase_started_ts
    ? (llm.active ? ((Date.now() / 1000) - Number(llm.phase_started_ts)) * 1000 : llm.latency_ms)
    : (lastLlmLayer?.data?._latency_ms || state.status?.llm?.last_latency_ms);
  meta.innerHTML = `${escapeHtml(llmPhaseLabel(llm?.phase || lastLlmLayer?.status || "complete"))} • <span ${relativeMsAttrs({ startTimestamp: llm?.active ? llm?.phase_started_ts : null, fixedMs: llm?.active ? null : liveLatency })}>${escapeHtml(fmtMs(liveLatency))}</span>`;

  const phases = llm?.history?.length
    ? llm.history
    : llmLayers.map((layer) => ({
        phase: layer.status,
        status: layer.status,
        text: layer.data?._raw_response || JSON.stringify(layer.data || {}, null, 2),
        latency_ms: layer.data?._latency_ms || null,
        updated_ts: layer.timestamp || null,
      }));

  root.innerHTML = `
    <article class="llm-live-card">
      <div class="detail-card-head">
        <strong>${escapeHtml(trace?.event ? eventTitle(trace.event) : eventTitle(queueItem?.event))}</strong>
        <button type="button" class="btn btn-ghost btn-small" data-copy-llm>Copy</button>
      </div>
      <p>${escapeHtml(queueItem?.status === "processing" ? `Currently in ${stageLabel(queueItem?.current_stage)}.` : "Latest LLM output for the selected trace.")}</p>
      <pre>${escapeHtml(liveText || "(no LLM tokens yet)")}</pre>
      <div class="llm-phase-list">
        ${phases.map((phase) => `
          <div class="llm-phase">
            <div class="llm-phase-head">
              <strong>${escapeHtml(llmPhaseLabel(phase.phase || phase.status))}</strong>
              <span ${relativeMsAttrs({ startTimestamp: phase.latency_ms ? null : phase.started_ts, fixedMs: phase.latency_ms || null })}>${escapeHtml(fmtMs(phase.latency_ms || (phase.started_ts ? ((Date.now() / 1000) - Number(phase.started_ts)) * 1000 : null)))}</span>
            </div>
            <p>${escapeHtml(truncate(phase.text || "(no output)", 220))}</p>
          </div>
        `).join("")}
      </div>
    </article>
  `;
  root.querySelector("[data-copy-llm]")?.addEventListener("click", async () => {
    try {
      await copyText(liveText || "");
    } catch (error) {
      console.error(error);
    }
  });
}

function renderRulesPanel() {
  const root = byId("rule-groups");
  const trace = primaryTrace();
  const stage = state.stageModels.rule_engine;
  const usedOptions = stage?.options?.filter((option) => option.id !== "rule:none") || [];
  if (!trace) {
    byId("rules-meta").textContent = "Waiting for an event";
    root.innerHTML = '<div class="empty-state">No event selected yet. When a trace arrives, only the rules used on that event will appear here.</div>';
    return;
  }
  if (!usedOptions.length) {
    byId("rules-meta").textContent = "0 rules used";
    root.innerHTML = '<div class="empty-state">This event did not use any rules.</div>';
    return;
  }
  byId("rules-meta").textContent = `${usedOptions.length} rule path${usedOptions.length === 1 ? "" : "s"} on this event`;
  root.innerHTML = "";
  for (const option of usedOptions) {
    const card = document.createElement("article");
    const selected = state.selectedStageId === "rule_engine" && state.selectedOptionId === option.id;
    card.className = `rule-card is-hit${selected ? " is-selected" : ""}`;
    card.innerHTML = `
      <div class="rule-head">
        <div>
          <p class="trace-kicker">${escapeHtml(option.raw?.rule?.id || option.raw?.result?.rule_id || option.id)}</p>
          <h3>${escapeHtml(option.title)}</h3>
        </div>
        <span class="trace-chip">${escapeHtml(option.meta[0] || "used")}</span>
      </div>
      <p>${escapeHtml(option.body)}</p>
      ${option.chips.length ? `
        <div class="rule-chip-row">
          ${option.chips.map((chip) => `<span class="chip" data-tone="${escapeHtml(chip.tone || "gold")}">${escapeHtml(chip.label)}</span>`).join("")}
        </div>
      ` : ""}
      <div class="rule-stats">
        ${option.details.slice(0, 4).map((item) => `
          <div class="rule-stat">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </div>
        `).join("")}
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedStageId = "rule_engine";
      state.selectedOptionId = option.id;
      state.detailOpen = true;
      renderAll();
    });
    root.appendChild(card);
  }
}

function renderRuleLibraryManager() {
  const root = byId("rule-library-groups");
  const meta = byId("rule-library-meta");
  if (!root || !meta) return;
  meta.textContent = `${state.rules.filter((item) => item.active).length} active of ${state.rules.length} total`;
  if (!state.rules.length) {
    root.innerHTML = '<div class="empty-state">No rules defined yet.</div>';
    return;
  }
  const groups = groupRules(state.rules);
  root.innerHTML = "";
  for (const [groupName, rules] of groups.entries()) {
    const section = document.createElement("section");
    section.className = "rule-group";
    section.innerHTML = `
      <div class="rule-group-head">
        <h3>${escapeHtml(groupName)}</h3>
        <span class="rules-meta">${escapeHtml(`${rules.length} rule${rules.length === 1 ? "" : "s"}`)}</span>
      </div>
      <div class="rule-group-list"></div>
    `;
    const list = section.querySelector(".rule-group-list");
    for (const rule of rules) {
      const conditions = rule.trigger?.conditions || [];
      const timeCond = conditions.find((item) => item.time_of_day)?.time_of_day || [];
      const card = document.createElement("article");
      card.className = "rule-card";
      card.innerHTML = `
        <div class="rule-head">
          <div>
            <p class="trace-kicker">${escapeHtml(rule.id)}</p>
            <h3>${escapeHtml(rule.description || rule.id)}</h3>
          </div>
          <span class="trace-chip">${escapeHtml(rule.active ? "active" : "paused")}</span>
        </div>
        <p>${escapeHtml(ruleSummary(rule))}</p>
        <div class="rule-stats">
          <div class="rule-stat"><span>Command</span><strong>${escapeHtml(ruleActionSummary(rule))}</strong></div>
          <div class="rule-stat"><span>Permission</span><strong>${escapeHtml(rule.permission || "ask")}</strong></div>
          <div class="rule-stat"><span>Time</span><strong>${escapeHtml(timeCond.length ? timeCond.join(", ") : "any")}</strong></div>
          <div class="rule-stat"><span>Cooldown</span><strong>${escapeHtml(`${rule.cooldown_sec || 0}s`)}</strong></div>
        </div>
        <div class="rule-actions">
          <button class="btn btn-ghost btn-small" data-action="toggle">${rule.active ? "Pause" : "Resume"}</button>
          <button class="btn btn-danger btn-small" data-action="delete">Delete</button>
        </div>
      `;
      card.querySelector('[data-action="toggle"]').addEventListener("click", async (event) => {
        event.stopPropagation();
        await toggleRule(rule.id);
      });
      card.querySelector('[data-action="delete"]').addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!window.confirm(`Delete rule "${rule.description || rule.id}"?`)) return;
        await deleteRule(rule.id);
      });
      list.appendChild(card);
    }
    root.appendChild(section);
  }
}

function summaryCard(label, value) {
  return `<div class="summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderWorldState() {
  const world = state.world || {};
  byId("state-meta").textContent = `${fmtCount(world.people_count || 0)} people • ${world.current_action || "no stable action"} • ${world.time_of_day || "time unknown"}`;
  byId("world-summary").innerHTML = `
    <div class="summary-grid">
      ${summaryCard("People", fmtCount(world.people_count || 0))}
      ${summaryCard("Current Action", world.current_action || "None")}
      ${summaryCard("Action Stable", world.action_stable ? "yes" : "no")}
      ${summaryCard("Voice Runtime", world.voice_running ? "running" : "stopped")}
      ${summaryCard("Vision Runtime", world.vision_running ? "running" : "stopped")}
      ${summaryCard("Time", world.current_time || "-")}
    </div>
  `;

  renderFeed(byId("event-feed"), state.events.slice(0, 12), (event) => ({
    title: event.type,
    body: previewData(event.data),
    meta: fmtIso(event.iso_time),
    passive: false,
    onClick: () => {
      if (state.traceMap.has(event.event_id)) {
        state.selectedTraceId = event.event_id;
        state.selectedStageId = "start";
        state.selectedOptionId = null;
        state.detailOpen = true;
        rebuildStageModels();
        renderAll();
        animateTrace(false);
      }
    },
  }), "No events yet.");

  renderFeed(byId("action-feed"), [...state.actions].slice(-12).reverse(), (action) => ({
    title: action.action?.command || action.action_type || "action",
    body: action.details || action.rule_description || "No detail",
    meta: `${action.executed_at || "-"} • ${action.status || "pending"}`,
    passive: true,
  }), "No action output yet.");

  renderFeed(byId("pending-feed"), state.pending, (pending) => ({
    title: pending.rule_id,
    body: pending.description || "Awaiting confirmation",
    meta: pending.asked_at || "-",
    passive: true,
  }), "Nothing is waiting for confirmation.");
}

function renderFeed(root, items, mapper, emptyText) {
  if (!items.length) {
    root.innerHTML = `<div class="empty-state">${escapeHtml(emptyText)}</div>`;
    return;
  }
  root.innerHTML = "";
  for (const item of items) {
    const view = mapper(item);
    const article = document.createElement("article");
    article.className = `feed-item${view.passive ? " is-passive" : ""}`;
    article.innerHTML = `
      <div class="feed-head">
        <strong>${escapeHtml(view.title)}</strong>
      </div>
      <p>${escapeHtml(view.body)}</p>
      <div class="feed-meta">${escapeHtml(view.meta)}</div>
    `;
    if (!view.passive && view.onClick) article.addEventListener("click", view.onClick);
    root.appendChild(article);
  }
}

function renderDetailDrawer() {
  const drawer = byId("detail-drawer");
  const trace = primaryTrace();
  const stage = state.stageModels[state.selectedStageId];
  const option = selectedStageOption(state.selectedStageId);

  if (!trace || !stage || !option || !state.detailOpen) {
    drawer.classList.remove("is-open");
    byId("detail-title").textContent = "Nothing selected";
    byId("detail-body").innerHTML = '<div class="empty-state">Click a decision option in the workspace to inspect it here.</div>';
    return;
  }

  const detailRows = option.details.length
    ? option.details.map((item) => `<div class="key-value"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.value)}</span></div>`).join("")
    : '<div class="empty-state">No extra structured detail on this item.</div>';

  drawer.classList.add("is-open");
  byId("detail-title").textContent = `${STAGE_META[state.selectedStageId].title} · ${option.title}`;
  const rawText = JSON.stringify(option.raw || {}, null, 2);
  byId("detail-body").innerHTML = `
    <section class="detail-card">
      <h3>Selection</h3>
      <div class="key-value"><strong>Event</strong><span>${escapeHtml(eventTitle(trace.event))}</span></div>
      <div class="key-value"><strong>Trace ID</strong><span>${escapeHtml(trace.event.event_id)}</span></div>
      <div class="key-value"><strong>Status</strong><span>${escapeHtml(statusLabel(stage.status))}</span></div>
      <div class="key-value"><strong>Chosen Row</strong><span>${escapeHtml(option.title)}</span></div>
      <div class="key-value"><strong>Total Trace Time</strong><span>${escapeHtml(fmtMs(trace.total_ms))}</span></div>
    </section>
    <section class="detail-card">
      <h3>Why This Path</h3>
      <p>${escapeHtml(option.body)}</p>
      ${option.meta.length ? `
        <div class="key-value"><strong>Meta</strong><span>${escapeHtml(option.meta.join(" • "))}</span></div>
      ` : ""}
      ${option.chips.length ? `
        <div class="rule-chip-row">
          ${option.chips.map((chip) => `<span class="chip" data-tone="${escapeHtml(chip.tone || "gold")}">${escapeHtml(chip.label)}</span>`).join("")}
        </div>
      ` : ""}
    </section>
    <section class="detail-card">
      <h3>Explained</h3>
      ${detailRows}
    </section>
    <section class="detail-card">
      <div class="detail-card-head">
        <h3>Raw Data</h3>
        <button type="button" class="btn btn-ghost btn-small" data-copy-detail>Copy</button>
      </div>
      <pre class="json-block">${escapeHtml(rawText)}</pre>
    </section>
  `;
  drawer.querySelector("[data-copy-detail]")?.addEventListener("click", async () => {
    try {
      await copyText(rawText);
    } catch (error) {
      console.error(error);
    }
  });
}

function renderLiveShell() {
  renderStatusStrip();
  renderHeroMetrics();
  renderStageCards();
  renderProgress();
  renderQueuePanel();
  renderLlmLivePanel();
  renderRulesPanel();
  renderDetailDrawer();
  updateCanvasSize();
  drawConnections();
  updateProgressVisuals();
  refreshRelativeTimers();
}

function scheduleRender(mode = "full", { animate = false } = {}) {
  if (mode === "full") renderQueueState.full = true;
  else renderQueueState.live = true;
  renderQueueState.animate = renderQueueState.animate || animate;
  if (renderQueueState.frame) return;
  renderQueueState.frame = window.requestAnimationFrame(() => {
    renderQueueState.frame = 0;
    const doFull = renderQueueState.full;
    const doLive = renderQueueState.live;
    const shouldAnimate = renderQueueState.animate;
    renderQueueState.full = false;
    renderQueueState.live = false;
    renderQueueState.animate = false;
    if (doFull) renderAll();
    else if (doLive) renderLiveShell();
    if (shouldAnimate && primaryTrace()) animateTrace(false);
  });
}

function refreshRelativeTimers() {
  for (const el of document.querySelectorAll("[data-elapsed-seconds-from]")) {
    el.textContent = elapsedSeconds(el.dataset.elapsedSecondsFrom, el.dataset.elapsedSecondsTo || null);
  }
  for (const el of document.querySelectorAll("[data-elapsed-ms-from], [data-elapsed-ms-fixed]")) {
    const fixed = el.dataset.elapsedMsFixed;
    if (fixed != null && fixed !== "") {
      el.textContent = fmtMs(Number(fixed));
      continue;
    }
    el.textContent = fmtMs(((Date.now() / 1000) - Number(el.dataset.elapsedMsFrom || 0)) * 1000);
  }
  for (const item of document.querySelectorAll(".queue-item[data-queue-status='processing'][data-stage-started-ts]")) {
    const stageStarted = Number(item.dataset.stageStartedTs || 0);
    const currentStage = item.dataset.currentStage || "";
    const stuck = stageStarted && ((Date.now() / 1000) - stageStarted) > (currentStage === "llm_reasoner" ? 8 : 5);
    item.classList.toggle("is-stuck", !!stuck);
    item.querySelector("[data-stuck-pill]")?.toggleAttribute("hidden", !stuck);
  }
}

function renderAll() {
  renderStatusStrip();
  renderHeroMetrics();
  renderStageCards();
  renderProgress();
  renderQueuePanel();
  renderLlmLivePanel();
  renderTraceHistory();
  renderRulesPanel();
  renderRuleLibraryManager();
  renderWorldState();
  renderDetailDrawer();
  updateCanvasSize();
  drawConnections();
  updateProgressVisuals();
  refreshRelativeTimers();
}

function defaultLayout(width, height) {
  if (width < 920) {
    return {
      start: { x: 24, y: 24 },
      event_bus: { x: width * 0.48 - 150, y: 120 },
      world_state: { x: 24, y: 360 },
      rule_engine: { x: width * 0.48 - 150, y: 560 },
      llm_reasoner: { x: 24, y: 860 },
      executor: { x: width * 0.48 - 150, y: 1060 },
    };
  }
  return {
    start: { x: width * 0.03, y: height * 0.34 },
    event_bus: { x: width * 0.16, y: height * 0.08 },
    world_state: { x: width * 0.33, y: height * 0.56 },
    rule_engine: { x: width * 0.5, y: height * 0.12 },
    llm_reasoner: { x: width * 0.66, y: height * 0.54 },
    executor: { x: width * 0.82, y: height * 0.24 },
  };
}

function loadLayout() {
  try {
    return JSON.parse(window.localStorage.getItem(LAYOUT_KEY) || "{}");
  } catch (error) {
    return {};
  }
}

function saveLayout() {
  window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(state.stagePositions));
}

function initializeLayout(force = false) {
  const workspace = byId("workspace");
  const defaults = defaultLayout(workspace.clientWidth, workspace.clientHeight);
  const saved = loadLayout();
  for (const stageId of STAGE_ORDER) {
    if (force || !state.stagePositions[stageId]) {
      state.stagePositions[stageId] = deepClone(saved[stageId] || defaults[stageId]);
    }
    applyStageTransform(stageId);
  }
  clampStagePositions();
}

function applyStageTransform(stageId) {
  const el = document.querySelector(`.stage-card[data-stage="${stageId}"]`);
  const pos = state.stagePositions[stageId];
  if (!el || !pos) return;
  el.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
}

function clampStagePositions() {
  const workspace = byId("workspace");
  for (const stageId of STAGE_ORDER) {
    const el = document.querySelector(`.stage-card[data-stage="${stageId}"]`);
    const pos = state.stagePositions[stageId];
    if (!el || !pos) continue;
    const maxX = Math.max(0, workspace.clientWidth - el.offsetWidth - 16);
    const maxY = Math.max(0, workspace.clientHeight - el.offsetHeight - 16);
    pos.x = clamp(pos.x, 8, maxX);
    pos.y = clamp(pos.y, 8, maxY);
    applyStageTransform(stageId);
  }
  drawConnections();
}

function stageAnchor(stageId) {
  const workspaceRect = byId("workspace").getBoundingClientRect();
  const card = document.querySelector(`.stage-card[data-stage="${stageId}"]`);
  const stageModel = state.stageModels[stageId];
  if (!card || !stageModel) return { x: 0, y: 0 };
  const chosen = selectedStageOption(stageId);
  const optionEl = chosen ? card.querySelector(`.option-card[data-option="${CSS.escape(chosen.id)}"]`) : null;
  const rect = (optionEl || card).getBoundingClientRect();
  return {
    x: rect.left - workspaceRect.left + rect.width / 2,
    y: rect.top - workspaceRect.top + rect.height / 2,
  };
}

function pathForPoints(points) {
  if (!points.length) return "";
  const parts = [`M ${points[0].x} ${points[0].y}`];
  for (let index = 1; index < points.length; index += 1) {
    const prev = points[index - 1];
    const next = points[index];
    const curve = clamp(Math.abs(next.x - prev.x) * 0.35, 50, 160);
    parts.push(`C ${prev.x + curve} ${prev.y}, ${next.x - curve} ${next.y}, ${next.x} ${next.y}`);
  }
  return parts.join(" ");
}

function updateCanvasSize() {
  const canvas = byId("flow-canvas");
  const workspace = byId("workspace");
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(workspace.clientWidth * ratio));
  canvas.height = Math.max(1, Math.round(workspace.clientHeight * ratio));
  canvas.style.width = `${workspace.clientWidth}px`;
  canvas.style.height = `${workspace.clientHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawConnections() {
  const svg = byId("flow-lines");
  const workspace = byId("workspace");
  svg.setAttribute("viewBox", `0 0 ${workspace.clientWidth} ${workspace.clientHeight}`);
  const points = STAGE_ORDER.map((stageId) => stageAnchor(stageId));
  const d = pathForPoints(points);
  svg.innerHTML = `
    <defs>
      <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#d4a449"></stop>
        <stop offset="40%" stop-color="#ff6c5d"></stop>
        <stop offset="75%" stop-color="#4db8ff"></stop>
        <stop offset="100%" stop-color="#d4a449"></stop>
      </linearGradient>
      <linearGradient id="lineBrightGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#f0d389"></stop>
        <stop offset="40%" stop-color="#ffb1a3"></stop>
        <stop offset="75%" stop-color="#9cdfff"></stop>
        <stop offset="100%" stop-color="#f0d389"></stop>
      </linearGradient>
    </defs>
    <path class="flow-base" d="${d}"></path>
    <path class="flow-travel" d="${d}"></path>
    <path id="master-path" d="${d}" fill="none" stroke="transparent" stroke-width="1"></path>
    <path class="flow-progress" d="${d}"></path>
  `;
  state.masterPathEl = byId("master-path");
  state.progressPathEl = svg.querySelector(".flow-progress");
}

function pathLength() {
  return state.masterPathEl ? state.masterPathEl.getTotalLength() : 0;
}

function stageProgressPoints() {
  const points = [0];
  if (!state.masterPathEl) return points;
  for (let index = 1; index < STAGE_ORDER.length; index += 1) {
    const point = stageAnchor(STAGE_ORDER[index]);
    let best = 0;
    let bestDistance = Infinity;
    const total = pathLength();
    const step = Math.max(8, total / 180);
    for (let distance = 0; distance <= total; distance += step) {
      const probe = state.masterPathEl.getPointAtLength(distance);
      const delta = Math.hypot(probe.x - point.x, probe.y - point.y);
      if (delta < bestDistance) {
        bestDistance = delta;
        best = distance;
      }
    }
    points.push(best);
  }
  return points;
}

function showPendingTracePreview() {
  const total = pathLength();
  if (!total) {
    state.replay.progress = 0;
    updateProgressVisuals();
    return;
  }
  const breakpoints = stageProgressPoints();
  const previewDistance = breakpoints[1] ?? total * 0.18;
  state.replay.lastBurstIndex = -1;
  state.replay.progress = clamp(previewDistance / total, 0, 1);
  updateProgressVisuals();
}

function showProgressThroughStage(stageId) {
  const total = pathLength();
  if (!total) {
    state.replay.progress = 0;
    updateProgressVisuals();
    return;
  }
  const index = Math.max(0, STAGE_ORDER.indexOf(stageId));
  const breakpoints = stageProgressPoints();
  const previewDistance = breakpoints[index] ?? total;
  state.replay.lastBurstIndex = -1;
  state.replay.progress = clamp(previewDistance / total, 0, 1);
  updateProgressVisuals();
}

function burstAt(stageId) {
  const point = stageAnchor(stageId);
  for (let i = 0; i < 18; i += 1) {
    const angle = (Math.PI * 2 * i) / 18 + Math.random() * 0.35;
    const speed = 28 + Math.random() * 62;
    state.particles.push({
      mode: "burst",
      x: point.x,
      y: point.y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size: 1.6 + Math.random() * 2.2,
      life: 0.6 + Math.random() * 0.45,
      age: 0,
      color: PARTICLE_COLORS[(i + Math.floor(Math.random() * 3)) % PARTICLE_COLORS.length],
    });
  }
}

function updateProgressVisuals() {
  const pct = clamp(state.replay.progress, 0, 1);
  byId("progress-fill").style.width = `${pct * 100}%`;
  const barWidth = document.querySelector(".progress-bar")?.clientWidth || 0;
  byId("progress-spark").style.transform = `translateX(${pct * barWidth}px)`;

  if (state.progressPathEl && state.masterPathEl) {
    const total = pathLength();
    const drawn = total * pct;
    state.progressPathEl.style.strokeDasharray = `${drawn} ${Math.max(total, 1)}`;
    state.progressPathEl.style.strokeDashoffset = "0";
  }

  const breakpoints = stageProgressPoints();
  let litIndex = 0;
  const total = pathLength();
  const currentDistance = total * pct;
  for (let i = 0; i < breakpoints.length; i += 1) {
    if (currentDistance >= breakpoints[i]) litIndex = i;
  }
  if (litIndex > state.replay.lastBurstIndex) {
    for (let index = state.replay.lastBurstIndex + 1; index <= litIndex; index += 1) {
      burstAt(STAGE_ORDER[index]);
    }
    state.replay.lastBurstIndex = litIndex;
  }

  for (let i = 0; i < STAGE_ORDER.length; i += 1) {
    const card = document.querySelector(`.stage-card[data-stage="${STAGE_ORDER[i]}"]`);
    if (card) card.dataset.lit = String(i <= litIndex);
  }
}

function animateTrace(immediate) {
  state.replay.lastBurstIndex = -1;
  if (immediate || !window.gsap || !primaryTrace()) {
    state.replay.progress = primaryTrace() ? 1 : 0;
    updateProgressVisuals();
    return;
  }
  gsap.killTweensOf(state.replay);
  state.replay.progress = 0;
  updateProgressVisuals();
  gsap.to(state.replay, {
    progress: 1,
    duration: 2.4,
    ease: "power3.inOut",
    onUpdate: updateProgressVisuals,
  });
}

function alphaHex(alpha) {
  return Math.round(clamp(alpha, 0, 1) * 255).toString(16).padStart(2, "0");
}

function drawGlow(ctx, x, y, size, color, alpha) {
  const radius = size * 3.8;
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, `${color}${alphaHex(alpha)}`);
  gradient.addColorStop(0.35, `${color}${alphaHex(alpha * 0.5)}`);
  gradient.addColorStop(1, `${color}00`);
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function animateParticles(now) {
  const canvas = byId("flow-canvas");
  const ctx = canvas.getContext("2d");
  const dt = state.lastFrameAt ? (now - state.lastFrameAt) / 1000 : 0.016;
  state.lastFrameAt = now;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  if (state.masterPathEl && primaryTrace() && now - state.lastParticleAt > 80) {
    state.lastParticleAt = now;
    const total = pathLength();
    const activeDistance = total * clamp(state.replay.progress, 0, 1);
    if (activeDistance > 24) {
      for (let i = 0; i < 2; i += 1) {
        state.particles.push({
          mode: "path",
          distance: Math.max(0, activeDistance - (8 + Math.random() * 48)),
          speed: 40 + Math.random() * 72,
          size: 1.4 + Math.random() * 2.1,
          life: 0.65 + Math.random() * 0.4,
          age: 0,
          color: PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
        });
      }
    }
  }

  const total = pathLength();
  const activeDistance = total * clamp(state.replay.progress, 0, 1);
  const next = [];
  for (const particle of state.particles) {
    particle.age += dt;
    if (particle.age >= particle.life) continue;
    const alpha = 1 - particle.age / particle.life;
    if (particle.mode === "path" && state.masterPathEl) {
      particle.distance += particle.speed * dt;
      if (particle.distance > activeDistance + 24) continue;
      const point = state.masterPathEl.getPointAtLength(clamp(particle.distance, 0, Math.max(activeDistance, 1)));
      drawGlow(ctx, point.x, point.y, particle.size, particle.color, alpha);
      next.push(particle);
      continue;
    }
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    particle.vx *= 0.985;
    particle.vy *= 0.985;
    drawGlow(ctx, particle.x, particle.y, particle.size, particle.color, alpha);
    next.push(particle);
  }
  state.particles = next;
  window.requestAnimationFrame(animateParticles);
}

function initializeInteractions() {
  interact(".stage-card").draggable({
    allowFrom: ".stage-head",
    modifiers: [
      interact.modifiers.restrictRect({ restriction: "parent", endOnly: true }),
    ],
    listeners: {
      start(event) {
        event.target.classList.add("is-dragging");
      },
      move(event) {
        const stageId = event.target.dataset.stage;
        state.stagePositions[stageId].x += event.dx;
        state.stagePositions[stageId].y += event.dy;
        applyStageTransform(stageId);
        drawConnections();
        updateProgressVisuals();
      },
      end(event) {
        event.target.classList.remove("is-dragging");
        clampStagePositions();
        saveLayout();
      },
    },
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

async function refreshDashboard(animate = false) {
  const payload = await fetchJson("/api/brain/bootstrap");
  applySnapshot(payload);
  renderAll();
  if (animate && primaryTrace()) animateTrace(false);
}

async function injectEvent(type, data) {
  await fetchJson("/api/brain/inject", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, data }),
  });
}

async function toggleRule(ruleId) {
  await fetchJson(`/api/brain/rules/${ruleId}/toggle`, { method: "POST" });
  state.rules = await fetchJson("/api/brain/rules");
  rebuildStageModels();
  renderAll();
}

async function deleteRule(ruleId) {
  await fetchJson(`/api/brain/rules/${ruleId}`, { method: "DELETE" });
  state.rules = await fetchJson("/api/brain/rules");
  rebuildStageModels();
  renderAll();
}

function setFormStatus(message, isError = false) {
  const el = byId("rule-form-status");
  el.textContent = message;
  el.style.color = isError ? "var(--red-soft)" : "var(--muted)";
}

function buildRulePayloadFromForm() {
  const description = byId("rule-description").value.trim();
  const triggerType = byId("rule-trigger-type").value;
  const actionMatch = byId("rule-action-match").value.trim();
  const whoMatch = byId("rule-who-match").value.trim();
  const command = byId("rule-command").value.trim();
  const area = byId("rule-area").value.trim();
  const scene = byId("rule-scene").value.trim();
  const brightnessRaw = byId("rule-brightness").value.trim();
  const cooldown = Number(byId("rule-cooldown").value || 0);
  const permission = byId("rule-permission").value;
  const requirePresence = byId("rule-presence").checked;
  const times = Array.from(document.querySelectorAll(".rule-time:checked")).map((el) => el.value);

  if (!description) throw new Error("Description is required.");
  if (!command) throw new Error("Command is required.");

  const trigger = { type: triggerType, conditions: [] };
  if (actionMatch) {
    const actions = actionMatch.split(",").map((item) => item.trim()).filter(Boolean);
    if (actions.length === 1) trigger.action = actions[0];
    if (actions.length > 1) trigger.action_in = actions;
  }
  if (whoMatch) trigger.who = whoMatch;
  if (times.length) trigger.conditions.push({ time_of_day: times });
  if (requirePresence) trigger.conditions.push({ people_present_any: true });

  const params = {};
  if (area) params.area = area;
  if (scene) params.scene = scene;
  if (brightnessRaw) {
    const brightness = Number(brightnessRaw);
    if (!Number.isFinite(brightness)) throw new Error("Brightness must be numeric.");
    params.brightness = brightness;
  }

  return {
    description,
    trigger,
    action: {
      type: "smart_home",
      command,
      params,
    },
    permission,
    active: true,
    cooldown_sec: Math.max(0, cooldown),
    expires: null,
    created_by: "dashboard",
  };
}

async function createRuleFromForm(event) {
  event.preventDefault();
  try {
    const rule = buildRulePayloadFromForm();
    await fetchJson("/api/brain/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rule }),
    });
    byId("rule-form").reset();
    byId("rule-presence").checked = true;
    state.rules = await fetchJson("/api/brain/rules");
    rebuildStageModels();
    renderAll();
    setFormStatus("Rule created.");
  } catch (error) {
    setFormStatus(error.message || String(error), true);
  }
}

function resetRuleForm() {
  byId("rule-form").reset();
  byId("rule-presence").checked = true;
  byId("rule-trigger-type").value = "action_changed";
  byId("rule-permission").value = "auto";
  byId("rule-cooldown").value = "60";
  setFormStatus("Build rules around actions, presence, time of day, and command output.");
}

function applyTraceProgressUpdate(payload) {
  if (!payload?.trace) return;
  upsertTrace(payload.trace);
  state.queue = payload.queue || state.queue;
  if (state.autoFollow || !state.selectedTraceId || state.selectedTraceId === payload.trace.event.event_id) {
    state.selectedTraceId = payload.trace.event.event_id;
    if (payload.progress?.current_stage && STAGE_ORDER.includes(payload.progress.current_stage)) {
      state.selectedStageId = payload.progress.current_stage;
    }
    if (!state.detailOpen) state.selectedOptionId = null;
  }
  rebuildStageModels();
  scheduleRender("live");
  if (payload.progress?.current_stage && STAGE_ORDER.includes(payload.progress.current_stage)) {
    showProgressThroughStage(payload.progress.current_stage);
  }
}

function queueTraceProgress(payload) {
  streamState.pendingProgress = payload;
  if (streamState.progressTimer) return;
  streamState.progressTimer = window.setTimeout(() => {
    streamState.progressTimer = 0;
    const next = streamState.pendingProgress;
    streamState.pendingProgress = null;
    applyTraceProgressUpdate(next);
  }, STREAM_PROGRESS_RENDER_MS);
}

function connectStream() {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource("/api/brain/stream");
  state.eventSource = source;

  source.onopen = () => {
    state.streamConnected = true;
    renderStatusStrip();
  };

  source.onerror = () => {
    state.streamConnected = false;
    renderStatusStrip();
  };

  source.onmessage = (message) => {
    const payload = JSON.parse(message.data);
    if (payload.type === "bootstrap") {
      applySnapshot(payload);
      scheduleRender("full");
      if (primaryTrace()) animateTrace(true);
      return;
    }
    if (payload.type === "bus_event" && payload.event) {
      upsertEvent(payload.event);
      if ((state.autoFollow || !state.selectedTraceId) && !state.traceMap.has(payload.event.event_id)) {
        upsertTrace(makePendingTrace(payload.event));
        state.selectedTraceId = payload.event.event_id;
        state.selectedStageId = "start";
        state.selectedOptionId = null;
        rebuildStageModels();
        scheduleRender("full");
        showPendingTracePreview();
      } else {
        scheduleRender("full");
      }
      return;
    }
    if (payload.type === "queue") {
      state.queue = payload.queue || state.queue;
      rebuildStageModels();
      scheduleRender("live");
      return;
    }
    if (payload.type === "trace_progress" && payload.trace) {
      queueTraceProgress(payload);
      return;
    }
    if (payload.type === "trace" && payload.trace) {
      if (streamState.progressTimer) {
        window.clearTimeout(streamState.progressTimer);
        streamState.progressTimer = 0;
      }
      streamState.pendingProgress = null;
      upsertTrace(payload.trace);
      state.status = payload.status || state.status;
      state.world = payload.world || state.world;
      state.actions = payload.actions || state.actions;
      state.pending = payload.pending || state.pending;
      state.fireHistory = payload.fire_history || state.fireHistory;
      state.queue = payload.queue || state.queue;
      if (state.autoFollow || !state.selectedTraceId) {
        state.selectedTraceId = payload.trace.event.event_id;
        state.selectedStageId = "start";
        state.selectedOptionId = null;
      }
      rebuildStageModels();
      scheduleRender("full", { animate: true });
      return;
    }
    if (payload.type === "rules") {
      state.rules = payload.rules || state.rules;
      state.pending = payload.pending || state.pending;
      state.status = payload.status || state.status;
      rebuildStageModels();
      scheduleRender("full");
      return;
    }
    if (payload.type === "pending") {
      state.pending = payload.pending || state.pending;
      state.actions = payload.actions || state.actions;
      state.status = payload.status || state.status;
      rebuildStageModels();
      scheduleRender("full");
    }
  };
}

function wireControls() {
  byId("btn-refresh").addEventListener("click", async () => {
    try {
      await refreshDashboard(false);
    } catch (error) {
      setFormStatus(error.message || String(error), true);
    }
  });

  byId("btn-clear").addEventListener("click", async () => {
    try {
      await fetchJson("/api/brain/clear", { method: "POST" });
    } catch (error) {
      setFormStatus(error.message || String(error), true);
    }
  });

  byId("btn-autofollow").addEventListener("click", () => {
    state.autoFollow = !state.autoFollow;
    byId("btn-autofollow").textContent = state.autoFollow ? "Auto Follow On" : "Auto Follow Off";
  });

  byId("btn-detail-close").addEventListener("click", () => {
    state.detailOpen = false;
    renderDetailDrawer();
  });

  byId("rule-form").addEventListener("submit", createRuleFromForm);
  byId("btn-rule-reset").addEventListener("click", resetRuleForm);
}

function bootAnimation() {
  if (!window.gsap) return;
  gsap.from(".panel", {
    y: 18,
    opacity: 0,
    duration: 0.75,
    stagger: 0.05,
    ease: "power3.out",
  });
}

function handleResize() {
  updateCanvasSize();
  initializeLayout(false);
  renderAll();
}

async function boot() {
  buildStageCards();
  initializeLayout(true);
  renderQuickActions();
  wireControls();
  initializeInteractions();
  connectStream();
  await refreshDashboard(false);
  animateTrace(true);
  bootAnimation();
  window.addEventListener("resize", handleResize);
  window.setInterval(() => {
    if (hasActiveTextSelection()) return;
    refreshRelativeTimers();
  }, 500);
  window.requestAnimationFrame(animateParticles);
}

window.addEventListener("DOMContentLoaded", () => {
  boot().catch((error) => {
    console.error(error);
    byId("detail-drawer").classList.add("is-open");
    byId("detail-title").textContent = "Failed to load";
    byId("detail-body").innerHTML = `<div class="empty-state">${escapeHtml(error.message || String(error))}</div>`;
  });
});
