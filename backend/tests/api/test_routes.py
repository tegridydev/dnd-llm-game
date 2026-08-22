from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session

from dndllm26.db.models import LoreDocument
from dndllm26.game.schemas import WorldUpdateOutput
from dndllm26.main import create_app


def _create_campaign(client: TestClient) -> dict[str, object]:
    hero = client.get("/api/heroes").json()[0]
    response = client.post(
        "/api/campaigns",
        json={
            "title": "Archive Test",
            "setting": "A quiet testing ground.",
            "tone": "hopeful",
            "protagonist_id": hero["id"],
            "companion_ids": [],
            "lore_document_ids": [],
        },
    )
    assert response.status_code == 201
    return response.json()


def test_route_contracts_security_headers_and_legacy_removal(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        ready = client.get("/api/health/ready")
        assert ready.status_code == 200
        assert ready.headers["x-content-type-options"] == "nosniff"
        assert ready.headers["x-frame-options"] == "DENY"

        invalid = client.post(
            "/api/heroes",
            json={"name": "Nox", "ancestry": "Human", "character_class": "Necromancer"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "request_validation_error"

        rejected_origin = client.post(
            "/api/heroes",
            headers={"Origin": "https://example.invalid"},
            json={"name": "Nox"},
        )
        assert rejected_origin.status_code == 403
        assert rejected_origin.json()["code"] == "origin_rejected"

        assert client.get("/api/health").status_code == 404
        assert client.get("/api/health/live").status_code == 404
        assert client.get("/api/models").status_code == 404
        assert client.get("/api/rules/options").status_code == 404
        assert client.get("/api/lore/worker").status_code in {404, 405}
        assert client.post("/api/characters", json={}).status_code == 404
        assert (
            client.post("/api/campaigns/1/turns/stream", json={"content": "Wait"}).status_code
            == 404
        )


def test_declared_oversized_upload_is_rejected_before_multipart(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/lore/upload",
            content=b"small",
            headers={
                "Content-Type": "multipart/form-data; boundary=x",
                "Content-Length": str(settings.max_upload_request_bytes + 1),
            },
        )
        assert response.status_code == 413
        assert response.json()["code"] == "upload_too_large"


def test_chunked_oversized_upload_and_core_readiness_failure(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:

        def chunks():
            for _ in range(settings.max_upload_request_bytes // 65_536 + 2):
                yield b"x" * 65_536

        oversized = client.post(
            "/api/lore/upload",
            content=chunks(),
            headers={"Content-Type": "multipart/form-data; boundary=x"},
        )
        assert oversized.status_code == 413

        resources = app.state.resources
        resources.lore_worker._running = False
        readiness = client.get("/api/health/ready")
        assert readiness.status_code == 503
        assert readiness.json()["status"] == "error"


def test_campaign_rename_archive_filter_and_export(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        campaign = _create_campaign(client)
        campaign_id = campaign["id"]
        renamed = client.patch(f"/api/campaigns/{campaign_id}", json={"title": "A Better Name"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "A Better Name"

        archived = client.patch(f"/api/campaigns/{campaign_id}", json={"archived": True})
        assert archived.status_code == 200
        assert archived.json()["archived_at"] is not None
        assert client.get("/api/campaigns").json() == []
        assert len(client.get("/api/campaigns?include_archived=true").json()) == 1

        exported = client.get(f"/api/campaigns/{campaign_id}/export")
        assert exported.status_code == 200
        assert exported.headers["content-disposition"].endswith('filename="a-better-name.json"')
        payload = exported.json()
        assert payload["schema_version"] == 1
        assert payload["campaign"]["id"] == campaign_id
        assert payload["opening"]["status"] == "needed"
        assert payload["characters"][0]["role"] == "protagonist"
        assert payload["turns"]
        assert "action_requests" not in payload

        restored = client.patch(f"/api/campaigns/{campaign_id}", json={"archived": False})
        assert restored.json()["archived_at"] is None


def test_model_settings_validate_roles_and_confirm_lore_reindex(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        resources = app.state.resources

        async def catalog() -> list[dict[str, object]]:
            return [
                {"name": "story:latest", "capabilities": ["completion"]},
                {"name": "rules:latest", "capabilities": ["completion"]},
                {"name": "vectors:latest", "capabilities": ["embedding"]},
                {"name": "vectors:new", "capabilities": ["embedding"]},
            ]

        resources.ollama.model_catalog = catalog
        current = client.get("/api/settings/models")
        assert current.status_code == 200
        assert current.json()["narration_style"] == "balanced"

        invalid = client.patch(
            "/api/settings/models",
            json={
                "chat_model": "vectors:latest",
                "utility_model": "rules:latest",
                "embed_model": "vectors:latest",
                "narration_style": "focused",
            },
        )
        assert invalid.status_code == 422
        assert "does not support completion" in invalid.json()["message"]

        with Session(resources.engine) as session:
            session.add(
                LoreDocument(
                    filename="setting.pdf",
                    storage_name="setting.pdf",
                    content_sha256="a" * 64,
                    status="ready",
                )
            )
            session.commit()

        change = {
            "chat_model": "story:latest",
            "utility_model": "rules:latest",
            "embed_model": "vectors:new",
            "narration_style": "focused",
        }
        confirmation = client.patch("/api/settings/models", json=change)
        assert confirmation.status_code == 409
        assert "requires reindexing 1 lore documents" in confirmation.json()["message"]

        saved = client.patch(
            "/api/settings/models",
            json={**change, "confirm_lore_reindex": True},
        )
        assert saved.status_code == 200
        assert saved.json()["embed_model"] == "vectors:new"
        assert saved.json()["narration_style"] == "focused"
        assert saved.json()["reindex_queued"] == 1
        assert resources.ollama.embed_model == "vectors:new"


def test_campaign_opening_failure_is_retryable_without_persisting_fallback(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        campaign = _create_campaign(client)
        campaign_id = campaign["id"]
        resources = app.state.resources

        async def empty_stream(*_args, **_kwargs):
            if False:
                yield ""

        resources.ollama.stream_dm = empty_stream
        failed = client.post(
            f"/api/campaigns/{campaign_id}/opening/stream",
            headers={"Idempotency-Key": "opening-retry-1"},
        )
        assert failed.status_code == 200
        assert '"status":"failed"' in failed.text
        detail = client.get(f"/api/campaigns/{campaign_id}").json()
        assert detail["opening"]["status"] == "failed"
        assert not any(turn["speaker"] == "DM" for turn in detail["turns"])
        assert "situation shifts" not in failed.text.casefold()

        async def good_stream(*_args, **_kwargs):
            yield "Rain lashes the Old Gate as Captain Rusk bars the road. "
            yield "A frightened courier points toward a broken watchtower."

        async def structured(*_args, **_kwargs) -> WorldUpdateOutput:
            return WorldUpdateOutput(
                location="Old Gate",
                objective="Reach the broken watchtower",
                summary="Captain Rusk blocks the rain-soaked gate.",
                choices=["Question Captain Rusk", "Help the courier"],
                facts=["A courier needs help."],
                npcs=["Captain Rusk, gate commander"],
                location_changed=True,
                objective_changed=True,
            )

        resources.ollama.stream_dm = good_stream
        resources.ollama.chat_structured = structured
        retried = client.post(
            f"/api/campaigns/{campaign_id}/opening/stream",
            headers={"Idempotency-Key": "opening-retry-1"},
        )
        assert retried.status_code == 200
        assert '"status":"complete"' in retried.text
        detail = client.get(f"/api/campaigns/{campaign_id}").json()
        assert detail["opening"]["status"] == "complete"
        assert detail["world_state"]["current_location"] == "Old Gate"
        assert sum(turn["speaker"] == "DM" for turn in detail["turns"]) == 1
