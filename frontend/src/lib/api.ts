import type { StreamEventName, TypedStreamEvent } from "../types";
import { decodeTypedEvent, SseParser, StreamProtocolError } from "./sse";

export const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8765/api").replace(
  /\/$/,
  ""
);

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code = "http_error",
    public readonly retryable = false,
    public readonly requestId?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError && error.requestId) return `${error.message} Request ${error.requestId}.`;
  return error instanceof Error ? error.message : String(error);
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  const requestId = response.headers.get("X-Request-ID") ?? undefined;
  const text = await response.text();
  try {
    const value = JSON.parse(text) as Record<string, unknown>;
    const detail = typeof value.detail === "string" ? value.detail : undefined;
    const message =
      typeof value.message === "string"
        ? value.message
        : detail ?? `Request failed with status ${response.status}.`;
    return new ApiError(
      message,
      response.status,
      typeof value.code === "string" ? value.code : "http_error",
      value.retryable === true,
      typeof value.request_id === "string" ? value.request_id : requestId
    );
  } catch {
    return new ApiError(
      text.trim() || `Request failed with status ${response.status}.`,
      response.status,
      "http_error",
      response.status >= 500,
      requestId
    );
  }
}

export async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    signal,
    headers: {
      Accept: "application/json",
      ...(init.body instanceof FormData ? {} : init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers
    }
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  return requestJson<T>(path, {}, signal);
}

export async function getStatusJson<T>(
  path: string,
  acceptedStatuses: readonly number[],
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    signal,
    headers: { Accept: "application/json" }
  });
  if (!acceptedStatuses.includes(response.status)) throw await errorFromResponse(response);
  return (await response.json()) as T;
}

export function postJson<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  return requestJson<T>(
    path,
    { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) },
    signal
  );
}

export function patchJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return requestJson<T>(path, { method: "PATCH", body: JSON.stringify(body) }, signal);
}

export function deleteJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  return requestJson<T>(path, { method: "DELETE" }, signal);
}

export async function downloadJson(path: string, fallbackName: string): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) throw await errorFromResponse(response);
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="([^"]+)"/i);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = match?.[1] ?? fallbackName;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

export async function uploadFile<T>(path: string, file: File, signal?: AbortSignal): Promise<T> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<T>(path, { method: "POST", body }, signal);
}

type StreamOptions = {
  path: string;
  body?: unknown;
  idempotencyKey: string;
  signal?: AbortSignal;
  expectedCampaignId: number;
  onEvent: (event: TypedStreamEvent) => void;
};

export async function streamRequest(options: StreamOptions): Promise<void> {
  const response = await fetch(`${API_BASE}${options.path}`, {
    method: "POST",
    signal: options.signal,
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      "Idempotency-Key": options.idempotencyKey
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body)
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body) {
    throw new StreamProtocolError("Streaming response had no readable body.", "missing_body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  let terminal = false;
  let expectedSequence = 1;
  let streamRequestId: string | null = null;

  const consume = (rawEvents: ReturnType<SseParser["push"]>) => {
    for (const raw of rawEvents) {
      const event = decodeTypedEvent(raw);
      if (event.envelope.campaign_id !== options.expectedCampaignId) {
        throw new StreamProtocolError("Stream event targeted the wrong campaign.", "campaign_mismatch");
      }
      if (streamRequestId === null) streamRequestId = event.envelope.request_id;
      if (event.envelope.request_id !== streamRequestId) {
        throw new StreamProtocolError("Stream request identity changed mid-response.", "request_mismatch");
      }
      if (event.envelope.sequence !== expectedSequence) {
        throw new StreamProtocolError("Stream events arrived out of sequence.", "sequence_mismatch");
      }
      expectedSequence += 1;
      options.onEvent(event);
      if (event.name === "done") terminal = true;
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      consume(parser.push(decoder.decode(value, { stream: true })));
    }
    consume(parser.push(decoder.decode(), true));
  } finally {
    reader.releaseLock();
  }
  if (!terminal) {
    throw new StreamProtocolError(
      "The stream ended before a terminal event was received.",
      "missing_terminal_event"
    );
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
