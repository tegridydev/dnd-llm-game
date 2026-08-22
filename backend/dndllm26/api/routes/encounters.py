from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from dndllm26.api.deps import get_resources
from dndllm26.api.schemas import CombatAction, validate_idempotency_key
from dndllm26.api.sse import SseEmitter, stream_headers
from dndllm26.core.resources import AppResources
from dndllm26.game.combat import prepare_combat_action
from dndllm26.game.campaigns import pending_roll_snapshot


router = APIRouter(prefix="/campaigns/{campaign_id}/encounters", tags=["encounters"])


def _key(value: str | None) -> str:
    try:
        return validate_idempotency_key(value) if value else uuid4().hex
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{encounter_id}/actions/stream")
async def combat_action_stream(
    campaign_id: int,
    encounter_id: int,
    payload: CombatAction,
    request: Request,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    key = _key(idempotency_header)
    with Session(resources.engine) as session:
        pending, replayed = prepare_combat_action(
            session,
            campaign_id=campaign_id,
            encounter_id=encounter_id,
            idempotency_key=key,
            action_id=payload.action_id,
            target_id=payload.target_id,
            destination_lane=payload.destination_lane,
        )
    emitter = SseEmitter(request.state.request_id, campaign_id)

    async def events():
        yield emitter.emit(
            "stream_started",
            {
                "operation": "combat_action",
                "operation_id": pending.action_request_id,
                "idempotency_key": key,
                "replayed": replayed,
            },
        )
        yield emitter.emit("roll_required", pending_roll_snapshot(pending).__dict__)
        yield emitter.emit("done", {"status": "roll_required", "authoritative_refresh": True})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers=stream_headers(request.state.request_id)
    )
