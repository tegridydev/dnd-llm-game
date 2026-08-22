import { expect, test } from "vitest";

import { createIdempotencyKey } from "./idempotency";

test("idempotency keys are valid and distinct", () => {
  const first = createIdempotencyKey();
  const second = createIdempotencyKey();
  expect(first).toMatch(/^[A-Za-z0-9._:-]{8,128}$/);
  expect(first).not.toBe(second);
});
