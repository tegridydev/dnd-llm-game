from __future__ import annotations

from fastapi import APIRouter

from dndllm26.api.routes import (
    campaigns,
    encounters,
    health,
    heroes,
    lore,
    openings,
    play,
    settings,
)

router = APIRouter()
router.include_router(health.router)
router.include_router(campaigns.router)
router.include_router(openings.router)
router.include_router(heroes.router)
router.include_router(lore.router)
router.include_router(play.router)
router.include_router(encounters.router)
router.include_router(settings.router)
