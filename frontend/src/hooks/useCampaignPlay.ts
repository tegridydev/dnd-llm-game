import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type {
  CampaignDetail,
  ChoiceUpdate,
  PendingRoll,
  RollResult,
  StreamDone,
  TypedStreamEvent
} from "../types";
import { ApiError, errorMessage, getJson, isAbortError, postJson, streamRequest } from "../lib/api";
import { createIdempotencyKey } from "../lib/idempotency";
import {
  initialPlayState,
  playReducer,
  type RetryOperation
} from "../features/play/state";

function formatRollResult(result: RollResult): string {
  const modifier =
    result.modifier > 0
      ? ` + ${result.modifier}`
      : result.modifier < 0
        ? ` - ${Math.abs(result.modifier)}`
        : "";
  const dc = result.dc === null ? "" : ` vs DC ${result.dc}`;
  return `${result.rolls.join(" + ")}${modifier} = ${result.total}${dc}: ${result.outcome}`;
}

function temporaryPlayerTurn(campaignId: number, content: string) {
  return {
    id: -Date.now(),
    campaign_id: campaignId,
    speaker: "Player" as const,
    content,
    created_at: new Date().toISOString()
  };
}

export function useCampaignPlay(
  campaignId: number | null,
  onOperationSettled?: () => Promise<void> | void
) {
  const [detail, setDetail] = useState<CampaignDetail | null>(null);
  const [state, dispatch] = useReducer(playReducer, initialPlayState);
  const [mutationPending, setMutationPending] = useState<string | null>(null);
  const streamController = useRef<AbortController | null>(null);
  const operationVersion = useRef(0);
  const activeCampaign = useRef<number | null>(campaignId);

  const setAction = useCallback((value: string) => {
    dispatch({ type: "set_action", value });
    if (campaignId) {
      const key = `dndllm26.actionDraft.${campaignId}`;
      if (value) window.localStorage.setItem(key, value);
      else window.localStorage.removeItem(key);
    }
  }, [campaignId]);

  useEffect(() => {
    activeCampaign.current = campaignId;
  }, [campaignId]);

  const refreshDetail = useCallback(
    async (targetId = campaignId, signal?: AbortSignal) => {
      if (!targetId) {
        setDetail(null);
        return null;
      }
      const next = await getJson<CampaignDetail>(`/campaigns/${targetId}`, signal);
      if (activeCampaign.current !== targetId) return null;
      setDetail(next);
      const recovery = next.recoverable_operation;
      const retry: RetryOperation | null = recovery
        ? recovery.kind === "action"
          ? {
              kind: "action",
              campaignId: targetId,
              content: recovery.action_text,
              idempotencyKey: recovery.idempotency_key
            }
          : recovery.pending_roll
            ? {
                kind: "roll",
                campaignId: targetId,
                pendingRollId: recovery.pending_roll.id,
                idempotencyKey: recovery.idempotency_key
              }
            : null
        : next.opening.retryable
          ? {
              kind: "opening",
              campaignId: targetId,
              idempotencyKey: next.opening.idempotency_key ?? createIdempotencyKey()
            }
          : null;
      dispatch({
        type: "loaded",
        choices: next.choices,
        pendingRoll: next.pending_roll ?? recovery?.pending_roll ?? null,
        retry,
        error: recovery?.error ?? null,
        rollResult: recovery?.dice_roll
          ? formatRollResult(recovery.dice_roll)
          : next.last_roll
            ? formatRollResult(next.last_roll)
            : "No rolls yet"
      });
      return next;
    },
    [campaignId]
  );

  useEffect(() => {
    streamController.current?.abort();
    operationVersion.current += 1;
    const draft = campaignId ? window.localStorage.getItem(`dndllm26.actionDraft.${campaignId}`) ?? "" : "";
    dispatch({ type: "reset", hasCampaign: campaignId !== null, draft });
    setDetail(null);
    if (!campaignId) return;
    const controller = new AbortController();
    refreshDetail(campaignId, controller.signal).catch((error: unknown) => {
      if (!isAbortError(error) && activeCampaign.current === campaignId) {
        dispatch({ type: "load_failed", message: errorMessage(error) });
      }
    });
    return () => controller.abort();
  }, [campaignId, refreshDetail]);

  const applyChoiceUpdate = useCallback((update: ChoiceUpdate) => {
    dispatch({ type: "choices", choices: update.choices });
    setDetail((current) =>
      current
        ? {
            ...current,
            choices: update.choices,
            world_state: {
              ...current.world_state,
              current_location: update.location,
              active_objective: update.objective,
              scene_summary: update.summary,
              choices_json: JSON.stringify(update.choices)
            }
          }
        : current
    );
  }, []);

  const handleEvent = useCallback(
    (event: TypedStreamEvent, targetCampaignId: number) => {
      if (activeCampaign.current !== targetCampaignId) return;
      switch (event.name) {
        case "phase": {
          const phase = event.envelope.data.status;
          if (phase === "checking_action") {
            dispatch({ type: "phase", phase: "checking", message: "Checking the action..." });
          } else if (phase === "dm_streaming") {
            dispatch({ type: "phase", phase: "generating", message: "The DM is shaping the scene..." });
          } else {
            dispatch({
              type: "phase",
              phase: "checking",
              message: "Preparing the next scene state and choices..."
            });
          }
          break;
        }
        case "narration":
        case "narration_delta":
          dispatch({ type: "append_draft", content: event.envelope.data.content });
          break;
        case "narration_replace":
          dispatch({ type: "replace_draft", content: event.envelope.data.content });
          break;
        case "roll_required":
          dispatch({ type: "roll_required", pendingRoll: event.envelope.data });
          break;
        case "roll_result":
          dispatch({ type: "roll_result", text: formatRollResult(event.envelope.data) });
          break;
        case "choices_updated":
          applyChoiceUpdate(event.envelope.data);
          break;
        case "encounter_started":
        case "encounter_updated":
          setDetail((current) => current ? { ...current, encounter: event.envelope.data } : current);
          break;
        case "combat_log":
          dispatch({ type: "phase", phase: "generating", message: event.envelope.data.events.join(" ") });
          break;
        case "party_recovered":
          dispatch({ type: "phase", phase: "ready", message: event.envelope.data.message });
          break;
        case "error":
          const error = event.envelope.data;
          dispatch({
            type: "operation_error",
            message: error.detail ? `${error.message} ${error.detail}` : error.message,
            retryable: error.retryable
          });
          break;
        case "stream_started":
        case "replay":
        case "done":
          break;
      }
    },
    [applyChoiceUpdate]
  );

  const runOperation = useCallback(
    async (operation: RetryOperation, optimistic: boolean) => {
      const targetId = operation.campaignId;
      if (activeCampaign.current !== targetId) return;
      streamController.current?.abort();
      const controller = new AbortController();
      streamController.current = controller;
      const version = ++operationVersion.current;
      dispatch({ type: "operation_started", operation });
      if (optimistic && operation.kind === "action") {
        window.localStorage.removeItem(`dndllm26.actionDraft.${targetId}`);
        setDetail((current) =>
          current
            ? {
                ...current,
                turns: [...current.turns, temporaryPlayerTurn(targetId, operation.content)]
              }
            : current
        );
      }

      const completion: { status?: StreamDone["status"] } = {};
      try {
        const path = operation.kind === "action"
          ? `/campaigns/${targetId}/actions/stream`
          : operation.kind === "opening"
            ? `/campaigns/${targetId}/opening/stream`
          : operation.kind === "roll"
            ? `/campaigns/${targetId}/rolls/${operation.pendingRollId}/resolve/stream`
            : `/campaigns/${targetId}/encounters/${operation.encounterId}/actions/stream`;
        const body = operation.kind === "action"
          ? { content: operation.content }
          : operation.kind === "combat"
            ? { action_id: operation.actionId, target_id: operation.targetId, destination_lane: operation.destinationLane }
            : undefined;
        await streamRequest({
          path,
          body,
          idempotencyKey: operation.idempotencyKey,
          signal: controller.signal,
          expectedCampaignId: targetId,
          onEvent: (event) => {
            if (version !== operationVersion.current || activeCampaign.current !== targetId) return;
            handleEvent(event, targetId);
            if (event.name === "done") completion.status = event.envelope.data.status;
          }
        });
        if (version !== operationVersion.current || activeCampaign.current !== targetId) return;
        if (!completion.status) {
          throw new Error("The server stream ended without a completion state.");
        }
        await refreshDetail(targetId);
        dispatch({ type: "operation_done", status: completion.status });
        await onOperationSettled?.();
      } catch (error: unknown) {
        if (version !== operationVersion.current || activeCampaign.current !== targetId) return;
        if (isAbortError(error)) {
          dispatch({ type: "cancelled" });
          return;
        }
        dispatch({
          type: "operation_error",
          message: errorMessage(error),
          retryable: error instanceof ApiError ? error.retryable : true
        });
        try {
          await refreshDetail(targetId);
        } catch {
          // Preserve the primary stream failure; the explicit retry remains available.
        }
      } finally {
        if (streamController.current === controller) streamController.current = null;
      }
    },
    [handleEvent, onOperationSettled, refreshDetail]
  );

  const submitAction = useCallback(
    async (content: string) => {
      if (!campaignId || !content.trim() || state.streaming || state.pendingRoll || state.retry) return;
      await runOperation(
        {
          kind: "action",
          campaignId,
          content: content.trim(),
          idempotencyKey: createIdempotencyKey()
        },
        true
      );
    },
    [campaignId, runOperation, state.pendingRoll, state.retry, state.streaming]
  );

  const generateOpening = useCallback(async (targetId = campaignId) => {
    if (!targetId || state.streaming) return;
    const existing = detail?.campaign.id === targetId ? detail.opening.idempotency_key : null;
    await runOperation(
      {
        kind: "opening",
        campaignId: targetId,
        idempotencyKey: existing ?? createIdempotencyKey()
      },
      false
    );
  }, [campaignId, detail, runOperation, state.streaming]);

  const resolvePendingRoll = useCallback(async () => {
    if (!campaignId || !state.pendingRoll || state.streaming) return;
    if (state.retry?.kind === "roll" && state.retry.pendingRollId === state.pendingRoll.id) {
      await runOperation(state.retry, false);
      return;
    }
    await runOperation(
      {
        kind: "roll",
        campaignId,
        pendingRollId: state.pendingRoll.id,
        idempotencyKey: createIdempotencyKey()
      },
      false
    );
  }, [campaignId, runOperation, state.pendingRoll, state.retry, state.streaming]);

  const submitCombatAction = useCallback(async (
    encounterId: number,
    actionId: string,
    targetId: number | null,
    destinationLane: "front" | "back" | null = null
  ) => {
    if (!campaignId || state.streaming || state.pendingRoll || state.retry) return;
    await runOperation({ kind: "combat", campaignId, encounterId, actionId, targetId,
      destinationLane, idempotencyKey: createIdempotencyKey() }, false);
  }, [campaignId, runOperation, state.pendingRoll, state.retry, state.streaming]);

  const retry = useCallback(async () => {
    if (!state.retry || state.streaming || activeCampaign.current !== state.retry.campaignId) return;
    await runOperation(state.retry, false);
  }, [runOperation, state.retry, state.streaming]);

  const cancel = useCallback(() => {
    streamController.current?.abort();
  }, []);

  const loadOlderTurns = useCallback(async () => {
    if (!campaignId || mutationPending || !detail?.turn_page.has_more || !detail.turn_page.next_cursor) return;
    setMutationPending("history");
    try {
      const older = await getJson<CampaignDetail>(
        `/campaigns/${campaignId}?cursor=${encodeURIComponent(detail.turn_page.next_cursor)}&limit=${detail.turn_page.limit}`
      );
      if (activeCampaign.current !== campaignId) return;
      setDetail((current) => current ? {
        ...current,
        turns: [...older.turns, ...current.turns],
        turn_page: older.turn_page
      } : current);
    } catch (error: unknown) {
      dispatch({ type: "operation_error", message: errorMessage(error), retryable: false });
    } finally {
      setMutationPending(null);
    }
  }, [campaignId, detail, mutationPending]);

  const rest = useCallback(async (kind: "short" | "long", hitDice = 1) => {
    if (!campaignId || mutationPending || state.streaming || state.pendingRoll || state.retry || detail?.encounter?.status === "active") return;
    setMutationPending(`rest-${kind}`);
    try {
      await postJson(`/campaigns/${campaignId}/rest`, { kind, hit_dice: kind === "short" ? hitDice : 0 });
      await refreshDetail(campaignId);
      await onOperationSettled?.();
    } catch (error: unknown) {
      try { await refreshDetail(campaignId); } catch { /* Preserve the mutation error. */ }
      dispatch({ type: "operation_error", message: errorMessage(error), retryable: false });
    } finally {
      setMutationPending(null);
    }
  }, [campaignId, detail?.encounter?.status, mutationPending, onOperationSettled, refreshDetail, state.pendingRoll, state.retry, state.streaming]);

  const completeQuest = useCallback(async (questId: number) => {
    if (!campaignId || mutationPending || state.streaming || state.pendingRoll || state.retry) return;
    setMutationPending(`quest-${questId}`);
    try {
      await postJson(`/campaigns/${campaignId}/quests/${questId}/complete`);
      await refreshDetail(campaignId);
      await onOperationSettled?.();
    } catch (error: unknown) {
      try { await refreshDetail(campaignId); } catch { /* Preserve the mutation error. */ }
      dispatch({ type: "operation_error", message: errorMessage(error), retryable: false });
    } finally {
      setMutationPending(null);
    }
  }, [campaignId, mutationPending, onOperationSettled, refreshDetail, state.pendingRoll, state.retry, state.streaming]);

  return {
    detail,
    state,
    mutationPending,
    dispatch,
    setAction,
    refreshDetail,
    submitAction,
    submitCombatAction,
    resolvePendingRoll,
    retry,
    cancel,
    loadOlderTurns,
    rest,
    completeQuest,
    generateOpening
  };
}
