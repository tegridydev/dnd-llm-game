import { expect, test } from "vitest";

import type { PendingRoll } from "../../types";
import { initialPlayState, playReducer } from "./state";

const pending: PendingRoll = {
  id: 4,
  campaign_id: 1,
  action_text: "Climb the wall",
  formula: "1d20+2",
  ability: "Strength",
  skill: "Athletics",
  dc: 13,
  reason: "Athletics: climb the wall",
  narration: "",
  status: "pending"
};

test("campaign reset cannot retain a previous pending roll", () => {
  const withRoll = playReducer(initialPlayState, { type: "roll_required", pendingRoll: pending });
  const reset = playReducer(withRoll, { type: "reset", hasCampaign: true });
  expect(reset.pendingRoll).toBeNull();
  expect(reset.phase).toBe("loading");
});

test("retryable operation errors preserve the idempotent operation", () => {
  const operation = { kind: "action" as const, campaignId: 1, content: "Open the gate", idempotencyKey: "request-123" };
  const started = playReducer(initialPlayState, { type: "operation_started", operation });
  const failed = playReducer(started, { type: "operation_error", message: "Ollama unavailable", retryable: true });
  expect(failed.retry).toEqual(operation);
  expect(failed.phase).toBe("error");
});

test("completion clears retry and streamed narration can be replaced", () => {
  const operation = { kind: "action" as const, campaignId: 1, content: "Wait", idempotencyKey: "request-456" };
  const started = playReducer(initialPlayState, { type: "operation_started", operation });
  const done = playReducer(started, { type: "operation_done", status: "complete" });
  expect(done).toMatchObject({ streaming: false, retry: null, phase: "ready" });
  const streamed = playReducer(initialPlayState, { type: "append_draft", content: "Unsafe text" });
  const replaced = playReducer(streamed, { type: "replace_draft", content: "Corrected second-person narration" });
  expect(replaced.draft).toBe("Corrected second-person narration");
});

test("authoritative load restores a durable interrupted operation", () => {
  const retry = { kind: "roll" as const, campaignId: 1, pendingRollId: pending.id, idempotencyKey: "roll-recovery-1" };
  const loaded = playReducer(initialPlayState, {
    type: "loaded",
    choices: [],
    pendingRoll: { ...pending, status: "resolved" },
    retry,
    error: "Narration was interrupted",
    rollResult: "12 + 2 = 14 vs DC 13: success"
  });
  expect(loaded.retry).toEqual(retry);
  expect(loaded).toMatchObject({ phase: "error", streaming: false });
});
