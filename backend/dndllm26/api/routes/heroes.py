from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from dndllm26.api.deps import get_session
from dndllm26.api.schemas import HeroCreate, HeroOut
from dndllm26.core.errors import NotFoundError
from dndllm26.db.models import Hero, now_utc
from dndllm26.game.campaigns import seed_default_heroes
from dndllm26.game.catalog import ABILITIES, build_hero_sheet

router = APIRouter(prefix="/heroes", tags=["heroes"])


@router.get("", response_model=list[HeroOut])
def list_heroes(session: Session = Depends(get_session)) -> list[Hero]:
    seed_default_heroes(session)
    return list(session.exec(select(Hero).order_by(Hero.updated_at.desc())).all())


@router.post("", response_model=HeroOut, status_code=201)
def create_hero(payload: HeroCreate, session: Session = Depends(get_session)) -> Hero:
    supplied = {ability: getattr(payload, ability) for ability in ABILITIES}
    custom = None if any(value is None for value in supplied.values()) else supplied
    sheet = build_hero_sheet(payload.ancestry, payload.character_class, custom)
    hero = Hero(
        name=payload.name,
        ancestry=payload.ancestry,
        character_class=payload.character_class,
        backstory=payload.backstory,
        inventory_json=json.dumps(payload.inventory or sheet["inventory"], ensure_ascii=False),
        **{ability: sheet[ability] for ability in ABILITIES},
        level=sheet["level"],
        max_hp=sheet["max_hp"],
        armor_class=sheet["armor_class"],
        speed=sheet["speed"],
        skills_json=json.dumps(sheet["skills"]),
        saves_json=json.dumps(sheet["saves"]),
        spells_json=json.dumps(sheet["spells"]),
        resources_json=json.dumps(sheet["resources"]),
    )
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero


@router.patch("/{hero_id}", response_model=HeroOut)
def update_hero(
    hero_id: int,
    payload: HeroCreate,
    session: Session = Depends(get_session),
) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise NotFoundError("Hero not found.")
    updates = payload.model_dump()
    rebuild = (
        "ancestry" in updates
        or "character_class" in updates
        or any(ability in updates for ability in ABILITIES)
    )
    for key, value in updates.items():
        if key == "inventory":
            hero.inventory_json = json.dumps(value, ensure_ascii=False)
        else:
            setattr(hero, key, value)
    if rebuild:
        rules_changed = "ancestry" in updates or "character_class" in updates
        ability_changed = any(ability in updates for ability in ABILITIES)
        scores = (
            {ability: int(getattr(hero, ability)) for ability in ABILITIES}
            if ability_changed or not rules_changed
            else None
        )
        sheet = build_hero_sheet(
            hero.ancestry,
            hero.character_class,
            scores,
            apply_ancestry_bonuses=scores is None,
        )
        if rules_changed and not ability_changed:
            for ability in ABILITIES:
                setattr(hero, ability, sheet[ability])
        hero.max_hp = sheet["max_hp"]
        hero.armor_class = sheet["armor_class"]
        hero.speed = sheet["speed"]
        hero.skills_json = json.dumps(sheet["skills"])
        hero.saves_json = json.dumps(sheet["saves"])
        hero.spells_json = json.dumps(sheet["spells"])
        hero.resources_json = json.dumps(sheet["resources"])
    hero.updated_at = now_utc()
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero


@router.delete("/{hero_id}")
def delete_hero(hero_id: int, session: Session = Depends(get_session)) -> dict[str, object]:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise NotFoundError("Hero not found.")
    session.delete(hero)
    session.commit()
    return {"status": "deleted", "id": hero_id}
