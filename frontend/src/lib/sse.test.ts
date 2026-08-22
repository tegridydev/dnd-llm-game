import { expect, test } from "vitest";

import { decodeTypedEvent, SseParser, StreamProtocolError } from "./sse";

function envelope(data: unknown, sequence = 1): string {
  return JSON.stringify({
    protocol_version: 1,
    request_id: "request-123",
    campaign_id: 7,
    sequence,
    timestamp: "2026-08-15T00:00:00Z",
    data
  });
}

test("SSE parser handles fragmented and multiline events", () => {
  const parser = new SseParser();
  expect(parser.push("event: narration_delta\ndata: ")).toEqual([]);
  const events = parser.push(`${envelope({ content: "Hello" })}\n\n`);
  expect(events).toHaveLength(1);
  expect(events[0]?.event).toBe("narration_delta");
  expect(decodeTypedEvent(events[0]!).envelope.data).toEqual({ content: "Hello" });
});

test("typed decoder accepts authoritative narration replacement", () => {
  const decoded = decodeTypedEvent({
    event: "narration_replace",
    data: envelope({ content: "You raise your blade." })
  });
  expect(decoded.name).toBe("narration_replace");
});

test("SSE parser flushes a terminal frame without trailing blank line", () => {
  const parser = new SseParser();
  const events = parser.push(
    `event: done\ndata: ${envelope({ status: "complete", authoritative_refresh: true })}`,
    true
  );
  expect(events).toHaveLength(1);
  expect(decodeTypedEvent(events[0]!).name).toBe("done");
});

test("typed decoder rejects invalid payloads and protocol versions", () => {
  expect(() => decodeTypedEvent({
    event: "error",
    data: envelope({ code: "x", message: "bad", retryable: "false" })
  })).toThrow(StreamProtocolError);
  const missing = JSON.parse(envelope({ content: "Hello" })) as Record<string, unknown>;
  delete missing.protocol_version;
  expect(() => decodeTypedEvent({ event: "narration", data: JSON.stringify(missing) }))
    .toThrow(StreamProtocolError);
  const unknown = JSON.parse(envelope({ content: "Hello" })) as Record<string, unknown>;
  unknown.protocol_version = 2;
  expect(() => decodeTypedEvent({ event: "narration", data: JSON.stringify(unknown) }))
    .toThrow(StreamProtocolError);
});
