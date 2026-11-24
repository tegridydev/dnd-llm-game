from __future__ import annotations

import json

from sqlmodel import Session, select

from dndllm26.game.lifecycle import playable_campaign
from dndllm26.core.errors import ConflictError, NotFoundError
from dndllm26.db.models import (
    ActionRequest,
    CampaignOpening,
    Character,
    Encounter,
    Quest,
    Turn,
    now_utc,
)
from dndllm26.game.catalog import CLASSES, ability_modifier, normalise_resources
from dndllm26.game.dice import roll_formula

LEVEL_THRESHOLDS = {1: 0, 2: 1, 3: 2, 4: 4, 5: 6}


def _require_idle(session: Session, campaign_id: int) -> None:
    if (
        session.exec(
            select(ActionRequest.id)
            .where(ActionRequest.campaign_id == campaign_id)
            .where(ActionRequest.status.in_(["processing", "roll_required"]))
        ).first()
        is not None
        or session.exec(
            select(CampaignOpening.id)
            .where(CampaignOpening.campaign_id == campaign_id)
            .where(CampaignOpening.status == "processing")
        ).first()
        is not None
    ):
        raise ConflictError("Finish the current turn before changing party progression.")


def rest_party(session: Session, campaign_id: int, kind: str, hit_dice: int) -> list[str]:
    campaign = playable_campaign(session, campaign_id)
    _require_idle(session, campaign_id)
    active = session.exec(
        select(Encounter)
        .where(Encounter.campaign_id == campaign_id)
        .where(Encounter.status == "active")
    ).first()
    if active:
        raise ConflictError("The party cannot rest during an active encounter.")
    characters = session.exec(select(Character).where(Character.campaign_id == campaign_id)).all()
    events: list[str] = []
    for character in characters:
        resources = normalise_resources(
            character.character_class, character.level, json.loads(character.resources_json)
        )
        if kind == "long":
            character.current_hp = character.max_hp
            refreshed = normalise_resources(character.character_class, character.level, {})
            refreshed.update(
                {
                    "primary_ability": resources.get("primary_ability"),
                    "class_actions": resources.get(
                        "class_actions", CLASSES[character.character_class]["actions"]
                    ),
                    "hit_dice": character.level,
                }
            )
            resources = refreshed
            character.conditions_json = "[]"
            events.append(f"{character.name} completes a long rest.")
        else:
            available = int(resources.get("hit_dice", character.level))
            spent = min(hit_dice, available) if 0 < character.current_hp < character.max_hp else 0
            if spent:
                hit_die = int(CLASSES[character.character_class]["hit_die"])
                healing = roll_formula(
                    f"{spent}d{hit_die}{ability_modifier(character.constitution) * spent:+d}"
                ).total
                recovered = min(character.max_hp - character.current_hp, max(0, healing))
                character.current_hp += recovered
                resources["hit_dice"] = available - spent
                events.append(
                    f"{character.name} spends {spent} hit dice and recovers {recovered} HP."
                )
        character.resources_json = json.dumps(resources)
        session.add(character)
    if not events:
        return ["No injured party member could spend a hit die."]
    session.add(Turn(campaign_id=campaign_id, speaker="System", content=" ".join(events)))
    campaign.updated_at = now_utc()
    session.add(campaign)
    session.commit()
    return events


def complete_quest(session: Session, campaign_id: int, quest_id: int) -> dict[str, object]:
    campaign = playable_campaign(session, campaign_id)
    _require_idle(session, campaign_id)
    quest = session.get(Quest, quest_id)
    if not quest or quest.campaign_id != campaign_id:
        raise NotFoundError("Quest not found.")
    if quest.status != "active":
        raise ConflictError("Only an active quest can be completed.")
    quest.status = "complete"
    quest.updated_at = now_utc()
    if not quest.reward_claimed:
        campaign.milestone_points += quest.milestone_reward
        quest.reward_claimed = True
    target_level = max(
        level
        for level, threshold in LEVEL_THRESHOLDS.items()
        if campaign.milestone_points >= threshold
    )
    levelled: list[str] = []
    for character in session.exec(
        select(Character).where(Character.campaign_id == campaign_id)
    ).all():
        resources = normalise_resources(
            character.character_class, character.level, json.loads(character.resources_json)
        )
        previous_level = character.level
        while character.level < target_level:
            character.level += 1
            gain = max(
                1,
                (int(CLASSES[character.character_class]["hit_die"]) // 2 + 1)
                + ability_modifier(character.constitution),
            )
            character.max_hp += gain
            character.current_hp += gain
            if character.level == 4:
                primary = str(resources.get("primary_ability", "strength"))
                setattr(character, primary, min(20, int(getattr(character, primary)) + 2))
        resources = normalise_resources(
            character.character_class, character.level, json.loads(character.resources_json)
        )
        resources["class_actions"] = CLASSES[character.character_class]["actions"]
        character.resources_json = json.dumps(resources)
        session.add(character)
        if character.level > previous_level:
            levelled.append(character.name)
    campaign.updated_at = now_utc()
    session.add(quest)
    session.add(campaign)
    session.add(
        Turn(
            campaign_id=campaign_id,
            speaker="System",
            content=f"Quest completed: {quest.title}. Milestone total: {campaign.milestone_points}.",
        )
    )
    session.commit()
    return {
        "quest_id": quest.id,
        "milestone_points": campaign.milestone_points,
        "level": target_level,
        "levelled_characters": levelled,
    }
