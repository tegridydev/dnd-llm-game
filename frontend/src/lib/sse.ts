import type {
  ChoiceUpdate,
  PendingRoll,
  RollResult,
  StreamDone,
  StreamEnvelope,
  StreamError,
  StreamEventData,
  StreamEventName,
  StreamPhase,
  TypedStreamEvent
} from "../types";

export type RawSseEvent = {
  event: string;
  data: string;
  id?: string;
};

export class SseParser {
  private buffer = "";

  push(chunk: string, final = false): RawSseEvent[] {
    this.buffer += chunk;
    const normalised = this.buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const frames = normalised.split("\n\n");
    if (final) {
      this.buffer = "";
    } else {
      this.buffer = frames.pop() ?? "";
    }
    return frames.map(parseFrame).filter((event): event is RawSseEvent => event !== null);
  }
}

function parseFrame(frame: string): RawSseEvent | null {
  if (!frame.trim()) return null;
  let event = "message";
  let id: string | undefined;
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    if (field === "data") data.push(value);
    if (field === "id") id = value;
  }
  if (!data.length) return null;
  return { event, data: data.join("\n"), id };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isString);
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(isNumber);
}

function isPendingRoll(value: unknown): value is PendingRoll {
  return (
    isRecord(value) &&
    isNumber(value.id) &&
    isNumber(value.campaign_id) &&
    isString(value.action_text) &&
    isString(value.formula) &&
    isString(value.ability) &&
    (value.skill === null || isString(value.skill)) &&
    isNumber(value.dc) &&
    isString(value.reason) &&
    isString(value.narration) &&
    isString(value.status)
  );
}

function isRollResult(value: unknown): value is RollResult {
  return (
    isRecord(value) &&
    isNumber(value.id) &&
    (value.pending_roll_id === null || isNumber(value.pending_roll_id)) &&
    isString(value.formula) &&
    isNumberArray(value.rolls) &&
    isNumber(value.modifier) &&
    isNumber(value.total) &&
    (value.dc === null || isNumber(value.dc)) &&
    isString(value.outcome) &&
    isString(value.reason)
  );
}

function isChoiceUpdate(value: unknown): value is ChoiceUpdate {
  return (
    isRecord(value) &&
    isStringArray(value.choices) &&
    isString(value.location) &&
    isString(value.objective) &&
    isString(value.summary)
  );
}

function isEncounter(value: unknown): boolean {
  return isRecord(value) && isNumber(value.id) && isString(value.status) &&
    Array.isArray(value.combatants) && Array.isArray(value.legal_actions) &&
    isStringArray(value.recent_events);
}

function isStreamError(value: unknown): value is StreamError {
  return (
    isRecord(value) &&
    isString(value.code) &&
    isString(value.message) &&
    isBoolean(value.retryable) &&
    (value.detail === undefined || value.detail === null || isString(value.detail)) &&
    (value.dice_result_saved === undefined || isBoolean(value.dice_result_saved))
  );
}

function isStreamDone(value: unknown): value is StreamDone {
  return (
    isRecord(value) &&
    ["complete", "roll_required", "failed"].includes(String(value.status)) &&
    isBoolean(value.authoritative_refresh)
  );
}

function isStreamPhase(value: unknown): value is { status: StreamPhase } {
  return (
    isRecord(value) &&
    ["checking_action", "dm_streaming", "utility_analyzing"].includes(String(value.status))
  );
}

function validData<K extends StreamEventName>(name: K, value: unknown): value is StreamEventData[K] {
  switch (name) {
    case "stream_started":
      return (
        isRecord(value) &&
        ["action", "roll_resolution", "combat_action", "campaign_opening"].includes(String(value.operation)) &&
        isNumber(value.operation_id) &&
        isString(value.idempotency_key) &&
        isBoolean(value.replayed)
      );
    case "phase":
      return isStreamPhase(value);
    case "narration":
    case "narration_delta":
    case "narration_replace":
      return isRecord(value) && isString(value.content);
    case "roll_required":
      return isPendingRoll(value);
    case "roll_result":
      return isRollResult(value);
    case "choices_updated":
      return isChoiceUpdate(value);
    case "encounter_started":
    case "encounter_updated":
      return isEncounter(value);
    case "combat_log":
      return isRecord(value) && isStringArray(value.events);
    case "party_recovered":
      return isRecord(value) && isString(value.message);
    case "replay":
      return isRecord(value) && value.status === "complete";
    case "error":
      return isStreamError(value);
    case "done":
      return isStreamDone(value);
  }
}

const EVENT_NAMES: StreamEventName[] = [
  "stream_started",
  "phase",
  "narration",
  "narration_delta",
  "narration_replace",
  "roll_required",
  "roll_result",
  "choices_updated",
  "encounter_started",
  "encounter_updated",
  "combat_log",
  "party_recovered",
  "replay",
  "error",
  "done"
];

export class StreamProtocolError extends Error {
  constructor(
    message: string,
    public readonly code: string
  ) {
    super(message);
    this.name = "StreamProtocolError";
  }
}

export function decodeTypedEvent(raw: RawSseEvent): TypedStreamEvent {
  if (!EVENT_NAMES.includes(raw.event as StreamEventName)) {
    throw new StreamProtocolError(`Unknown stream event: ${raw.event}`, "unknown_event");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw.data);
  } catch {
    throw new StreamProtocolError("Stream event was not valid JSON.", "invalid_json");
  }
  if (
    !isRecord(parsed) ||
    parsed.protocol_version !== 1 ||
    !isString(parsed.request_id) ||
    !isNumber(parsed.campaign_id) ||
    !isNumber(parsed.sequence) ||
    !isString(parsed.timestamp) ||
    !("data" in parsed)
  ) {
    throw new StreamProtocolError("Stream envelope was invalid.", "invalid_envelope");
  }
  const name = raw.event as StreamEventName;
  if (!validData(name, parsed.data)) {
    throw new StreamProtocolError(`Invalid payload for ${name}.`, "invalid_event_payload");
  }
  return {
    name,
    envelope: parsed as StreamEnvelope<never>
  } as TypedStreamEvent;
}
