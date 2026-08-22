import { describe, expect, test } from "bun:test";
import { httpUrl, urlHost } from "./network";

describe("launcher URL hosts", () => {
  test.each([
    ["127.0.0.1", "127.0.0.1"],
    ["localhost", "localhost"],
    ["::1", "[::1]"],
    ["[::1]", "[::1]"]
  ])("formats %s", (input, expected) => expect(urlHost(input)).toBe(expected));

  test("builds a valid IPv6 URL", () => {
    expect(httpUrl("::1", 8765)).toBe("http://[::1]:8765");
  });
});
