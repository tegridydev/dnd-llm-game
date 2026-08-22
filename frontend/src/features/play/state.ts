import type { PendingRoll } from "../../types";

export type PlayPhase =
  | "idle"
  | "loading"
  | "ready"
  | "checking"
  | "generating"
  | "roll_required"
  | "rolling"
  | "error";

export type RetryOperation =
  | { kind: "opening"; campaignId: number; idempotencyKey: string }
  | { kind: "action"; campaignId: number; content: string; idempotencyKey: string }
  | {
      kind: "roll";
      campaignId: number;
      pendingRollId: number;
      idempotencyKey: string;
    }
  | {
      kind: "combat";
      campaignId: number;
      encounterId: number;
      actionId: string;
      targetId: number | null;
      destinationLane: "front" | "back" | null;
      idempotencyKey: string;
    };

export type PlayState = {
  action: string;
  choices: string[];
  draft: string;
  error: string | null;
  pendingRoll: PendingRoll | null;
  phase: PlayPhase;
  rollResult: string;
  statusMessage: string;
  streaming: boolean;
  retry: RetryOperation | null;
};

export const initialPlayState: PlayState = {
  action: "",
  choices: [],
  draft: "",
  error: null,
  pendingRoll: null,
  phase: "idle",
  rollResult: "No rolls yet",
  statusMessage: "Select or create a campaign.",
  streaming: false,
  retry: null
};

export type PlayAction =
  | { type: "reset"; hasCampaign: boolean; draft?: string }
  | { type: "set_action"; value: string }
  | {
      type: "loaded";
      choices: string[];
      pendingRoll: PendingRoll | null;
      retry: RetryOperation | null;
      error: string | null;
      rollResult: string;
    }
  | { type: "load_failed"; message: string }
  | { type: "operation_started"; operation: RetryOperation }
  | { type: "phase"; phase: PlayPhase; message: string }
  | { type: "append_draft"; content: string }
  | { type: "replace_draft"; content: string }
  | { type: "roll_required"; pendingRoll: PendingRoll }
  | { type: "roll_result"; text: string }
  | { type: "choices"; choices: string[] }
  | { type: "operation_error"; message: string; retryable: boolean }
  | { type: "operation_done"; status: "complete" | "roll_required" | "failed" }
  | { type: "cancelled" }
  | { type: "clear_error" };

export function playReducer(state: PlayState, action: PlayAction): PlayState {
  switch (action.type) {
    case "reset":
      return {
        ...initialPlayState,
        action: action.draft ?? "",
        phase: action.hasCampaign ? "loading" : "idle",
        statusMessage: action.hasCampaign ? "Loading campaign..." : initialPlayState.statusMessage
      };
    case "set_action":
      return { ...state, action: action.value };
    case "replace_draft":
      return { ...state, draft: action.content };
    case "loaded":
      const openingPending = action.retry?.kind === "opening";
      return {
        ...state,
        choices: action.choices,
        pendingRoll: action.pendingRoll,
        retry: action.retry,
        error: action.error,
        rollResult: action.rollResult,
        phase: action.error ? "error" : action.pendingRoll ? "roll_required" : "ready",
        statusMessage: openingPending
          ? "Generate the opening scene to begin."
          : action.retry
          ? "An interrupted operation must be resumed before continuing."
          : action.pendingRoll
            ? "Dice check required. Roll to continue."
            : "Ready for your next move.",
        streaming: false
      };
    case "load_failed":
      return {
        ...state,
        error: action.message,
        phase: "error",
        statusMessage: "Campaign could not be loaded."
      };
    case "operation_started":
      return {
        ...state,
        action: action.operation.kind === "action" ? "" : state.action,
        draft: "",
        error: null,
        phase: action.operation.kind === "roll" ? "rolling" : "checking",
        rollResult: action.operation.kind === "roll" ? "Rolling..." : state.rollResult,
        statusMessage:
          action.operation.kind === "roll"
            ? "Rolling and resolving the check..."
            : action.operation.kind === "opening"
              ? "The Dungeon Master is preparing the opening scene..."
              : "The rules referee is checking your action...",
        streaming: true,
        retry: action.operation
      };
    case "phase":
      return { ...state, phase: action.phase, statusMessage: action.message };
    case "append_draft":
      return { ...state, draft: state.draft + action.content };
    case "roll_required":
      return {
        ...state,
        pendingRoll: action.pendingRoll,
        phase: "roll_required",
        statusMessage: "Dice check required. Roll to continue."
      };
    case "roll_result":
      return { ...state, rollResult: action.text, pendingRoll: null };
    case "choices":
      return { ...state, choices: action.choices };
    case "operation_error":
      return {
        ...state,
        error: action.message,
        phase: "error",
        statusMessage: action.retryable
          ? "The operation stopped safely. Retry is available."
          : "The operation could not be completed.",
        retry: action.retryable ? state.retry : null
      };
    case "operation_done":
      if (action.status === "failed") {
        return { ...state, streaming: false, draft: "" };
      }
      if (action.status === "roll_required") {
        return {
          ...state,
          streaming: false,
          draft: "",
          phase: "roll_required",
          retry: null
        };
      }
      return {
        ...state,
        streaming: false,
        draft: "",
        phase: "ready",
        statusMessage: "Ready for your next move.",
        retry: null
      };
    case "cancelled":
      return {
        ...state,
        streaming: false,
        draft: "",
        phase: "error",
        error: "Generation was cancelled before completion.",
        statusMessage: "The operation was cancelled safely. Retry is available."
      };
    case "clear_error":
      return { ...state, error: null, phase: state.pendingRoll ? "roll_required" : "ready" };
  }
}
