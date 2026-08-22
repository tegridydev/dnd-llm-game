from __future__ import annotations

import json
import re
from dataclasses import dataclass
from random import SystemRandom
from typing import Sequence

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from dndllm26.core.errors import ConflictError, NotFoundError, ValidationError
from dndllm26.db.models import (
    ActionRequest,
    Campaign,
    Character,
    Combatant,
    DiceRoll,
    Encounter,
    GameEvent,
    PendingRoll,
    Turn,
    WorldState,
    now_utc,
)
from dndllm26.game.catalog import (
    CLASSES,
    COMBAT_ACTIONS,
    ENEMIES,
    ability_modifier,
    normalise_resources,
    proficiency_bonus,
)
from dndllm26.game.dice import roll_formula

_ENCOUNTER_CONDITIONS = {
    "blessed",
    "dodging",
    "exposed",
    "hindered",
    "inspired",
    "mage_armor",
    "marked",
    "restrained",
    "shielded",
    "sleeping",
    "surging",
}


@dataclass(frozen=True)
class CombatAssessment:
    starts: bool
    enemy_keys: tuple[str, ...] = ()
    display_names: tuple[str, ...] = ()
    trigger: str = "none"
    evidence: str = ""
    duel: bool = False


_COMMITTED_ACTION_RE = re.compile(
    r"\b(?:attack|engage|strike|stab|slash|shoot|fire\s+(?:at|on)|kill|ambush|"
    r"grapple|tackle|punch|kick|charge|lunge|accept(?:ing|ed)?\b[^.]{0,40}\b"
    r"(?:duel|challenge|fight)|cast\s+(?:fire|bolt|missile|blast|ray))\b",
    re.IGNORECASE,
)
_PREPARATORY_ACTION_RE = re.compile(
    r"\b(?:look|wait|watch|search)\s+for\s+(?:an?\s+)?opening\b|"
    r"\b(?:prepare|ready|plan|consider|threaten|warn|challenge)\b|"
    r"\b(?:draw|unsheathe|raise|aim)\b(?![^.]{0,45}\b(?:attack|shoot|fire|strike)\b)",
    re.IGNORECASE,
)
_DUEL_RE = re.compile(
    r"\b(?:duel|one[ -]on[ -]one|1\s*on\s*1|single combat|"
    r"accept(?:ing|ed)?\s+(?:the\s+)?(?:challenge|fight))\b",
    re.IGNORECASE,
)
_NPC_ATTACK_RE = re.compile(
    r"\b(?:attacks?|strikes?|stabs?|slashes?|shoots?|grapples?|tackles?|"
    r"(?:charges?|lunges?)\s+(?:at|toward|forward|in|with)\b|"
    r"swings?\s+(?:at|a\s+weapon|his\s+weapon|her\s+weapon|their\s+weapon)\b|"
    r"fires?\s+(?:at|upon)\b|casts?\s+(?:fire|bolt|missile|blast|ray))\b",
    re.IGNORECASE,
)
_TARGET_ROLE_RE = re.compile(
    r"\b(?:(?:closest|nearest|arena|city|armed|hostile|enemy)\s+)?"
    r"(guard|soldier|gladiator|knight|warrior|archer|mage|thug|raider|bandit|"
    r"goblin|wolf|skeleton|cultist|ogre)(s)?\b",
    re.IGNORECASE,
)


def normalise_npc_identity(value: str) -> str:
    """Return a stable NPC identity while discarding changing descriptors."""
    identity = re.split(r"[,:(]", value, maxsplit=1)[0].strip()
    words = identity.split()
    if words and words[-1].casefold() in {
        "soldiers",
        "guards",
        "bandits",
        "goblins",
        "wolves",
        "skeletons",
        "cultists",
        "ogres",
    }:
        singular = {
            "soldiers": "Soldier",
            "guards": "Guard",
            "bandits": "Bandit",
            "goblins": "Goblin",
            "wolves": "Wolf",
            "skeletons": "Skeleton",
            "cultists": "Cultist",
            "ogres": "Ogre",
        }[words[-1].casefold()]
        words[-1] = singular if words[-1][:1].isupper() else singular.casefold()
    return " ".join(words)[:120]


def _enemy_template(value: str) -> str:
    lower = value.casefold()
    for key in ("goblin", "wolf", "skeleton", "cultist", "ogre", "bandit"):
        if re.search(rf"\b{key}(?:s)?\b", lower):
            return key
    return "bandit"


def _mentioned_npc(text: str, known_npcs: Sequence[str]) -> str:
    lower = text.casefold()
    matches: list[str] = []
    for raw in known_npcs:
        identity = normalise_npc_identity(raw)
        if not identity:
            continue
        terms = {identity.casefold(), identity.casefold().removesuffix("s")}
        last = identity.split()[-1].casefold().removesuffix("s")
        terms.add(last)
        if any(re.search(rf"\b{re.escape(term)}s?\b", lower) for term in terms if term):
            if identity.casefold() not in {item.casefold() for item in matches}:
                matches.append(identity)
    return matches[0] if len(matches) == 1 else ""


def _explicit_target(text: str) -> tuple[str, bool]:
    match = _TARGET_ROLE_RE.search(text)
    if not match:
        return "", False
    return match.group(1).capitalize(), bool(match.group(2))


def assess_combat(
    action: str,
    narration: str = "",
    *,
    model_enemy_keys: Sequence[str] = (),
    model_names: Sequence[str] = (),
    known_npcs: Sequence[str] = (),
    context_text: str = "",
) -> CombatAssessment:
    """Combine deterministic evidence with model hints; model output is never the sole gate."""
    committed = bool(_COMMITTED_ACTION_RE.search(action)) and not bool(
        _PREPARATORY_ACTION_RE.search(action)
    )
    npc_attack = bool(narration and _NPC_ATTACK_RE.search(narration))
    if not committed and not npc_attack:
        return CombatAssessment(starts=False)

    duel = bool(_DUEL_RE.search(action))
    combined_text = f"{action}\n{narration}\n{context_text}"
    target = _mentioned_npc(combined_text, known_npcs)
    role_target, plural_target = _explicit_target(action or narration)
    target = target or role_target
    cleaned_model_names = tuple(
        name for raw in model_names if (name := normalise_npc_identity(raw))
    )
    valid_model_keys = tuple(key for key in model_enemy_keys if key in ENEMIES)

    if duel:
        name = target or (cleaned_model_names[0] if cleaned_model_names else "Opponent")
        key = _enemy_template(name or combined_text)
        return CombatAssessment(
            starts=True,
            enemy_keys=(key,),
            display_names=(name,),
            trigger="player_commitment" if committed else "npc_attack",
            evidence="duel",
            duel=True,
        )

    if valid_model_keys:
        names = cleaned_model_names[: len(valid_model_keys)]
        if target and len(valid_model_keys) == 1:
            names = (target,)
        elif target and plural_target and not names:
            names = tuple(f"{target} {index}" for index in range(1, len(valid_model_keys) + 1))
        return CombatAssessment(
            starts=True,
            enemy_keys=valid_model_keys,
            display_names=names,
            trigger="player_commitment" if committed else "npc_attack",
            evidence="action" if committed else "narration",
        )

    name = target or "Opponent"
    key = _enemy_template(name or combined_text)
    count = 2 if plural_target else 1
    return CombatAssessment(
        starts=True,
        enemy_keys=(key,) * count,
        display_names=tuple(
            f"{name} {index}" if count > 1 else name for index in range(1, count + 1)
        ),
        trigger="player_commitment" if committed else "npc_attack",
        evidence="action" if committed else "narration",
    )


def encounter_detail(session: Session, campaign_id: int) -> dict[str, object] | None:
    encounter = session.exec(
        select(Encounter)
        .where(Encounter.campaign_id == campaign_id)
        .order_by(Encounter.created_at.desc(), Encounter.id.desc())
    ).first()
    if not encounter or encounter.id is None:
        return None
    recent_events: list[str] = []
    event_rows = session.exec(
        select(GameEvent)
        .where(GameEvent.campaign_id == campaign_id)
        .where(GameEvent.event_type == "combat_resolved")
        .where(GameEvent.created_at >= encounter.created_at)
        .order_by(GameEvent.created_at.desc(), GameEvent.id.desc())
        .limit(4)
    ).all()
    for row in reversed(event_rows):
        try:
            values = json.loads(row.payload_json).get("events", [])
        except (json.JSONDecodeError, AttributeError):
            values = []
        recent_events.extend(str(value)[:220] for value in values if str(value).strip())
    recent_events = recent_events[-6:]
    combatants = session.exec(
        select(Combatant)
        .where(Combatant.encounter_id == encounter.id)
        .order_by(Combatant.initiative.desc(), Combatant.id)
    ).all()
    current = combatants[encounter.turn_index % len(combatants)] if combatants else None
    protagonist = next(
        (
            item
            for item in combatants
            if item.character_id
            and (character := session.get(Character, item.character_id))
            and character.role == "protagonist"
        ),
        None,
    )
    legal_actions: list[dict[str, object]] = []
    if encounter.status == "active" and protagonist and current and protagonist.id == current.id:
        character = (
            session.get(Character, protagonist.character_id) if protagonist.character_id else None
        )
        if character:
            resources = normalise_resources(
                character.character_class, character.level, json.loads(character.resources_json)
            )
            action_ids = (
                ["death_save"]
                if protagonist.current_hp == 0
                else [
                    "weapon_attack",
                    "move_lane",
                    "dodge",
                    "flee",
                    *CLASSES[character.character_class]["actions"],
                ]
            )
            for action_id in dict.fromkeys(action_ids):
                if action_id == "move_lane" and "restrained" in _conditions(protagonist):
                    continue
                definition = COMBAT_ACTIONS[action_id]
                if int(definition.get("min_level", 1)) > character.level:
                    continue
                spell_level = str(definition.get("spell_level", ""))
                if spell_level and resources.get("spell_slots", {}).get(spell_level, 0) <= 0:
                    continue
                if (
                    definition.get("uses")
                    and resources.get("class_uses", {}).get(action_id, 0) <= 0
                ):
                    continue
                target_type = str(definition.get("target", "none"))
                resource_label = None
                if spell_level:
                    resource_label = (
                        f"Level {spell_level} slot · "
                        f"{resources.get('spell_slots', {}).get(spell_level, 0)} remaining"
                    )
                elif definition.get("uses"):
                    resource_label = (
                        f"{resources.get('class_uses', {}).get(action_id, 0)} uses remaining"
                    )
                legal_actions.append(
                    {
                        "id": action_id,
                        "label": definition["label"],
                        "target_type": target_type,
                        "requires_target": target_type in {"enemy", "ally"},
                        "description": _action_description(definition),
                        "resource": resource_label,
                    }
                )
    return {
        "id": encounter.id,
        "name": encounter.name,
        "status": encounter.status,
        "round_number": encounter.round_number,
        "current_combatant_id": current.id if current else None,
        "protagonist_combatant_id": protagonist.id if protagonist else None,
        "legal_actions": legal_actions,
        "recent_events": recent_events,
        "combatants": [
            {
                "id": item.id,
                "character_id": item.character_id,
                "name": item.name,
                "side": item.side,
                "lane": item.lane,
                "initiative": item.initiative,
                "max_hp": item.max_hp,
                "current_hp": item.current_hp,
                "armor_class": _effective_defense(item),
                "conditions": json.loads(item.conditions_json),
                "death_successes": item.death_successes,
                "death_failures": item.death_failures,
                "defeated": item.defeated,
            }
            for item in combatants
        ],
    }


def active_encounter(session: Session, campaign_id: int) -> Encounter | None:
    return session.exec(
        select(Encounter)
        .where(Encounter.campaign_id == campaign_id)
        .where(Encounter.status == "active")
    ).first()


def start_encounter(
    session: Session,
    campaign_id: int,
    enemy_keys: Sequence[str],
    display_names: Sequence[str] = (),
    opening_events: list[str] | None = None,
) -> Encounter:
    if not session.get(Campaign, campaign_id):
        raise NotFoundError("Campaign not found.")
    active = active_encounter(session, campaign_id)
    if active:
        raise ConflictError("An encounter is already active.")
    keys = balanced_enemy_keys(session, campaign_id, enemy_keys)
    selected_names: list[str] = []
    unused = list(enumerate(enemy_keys))
    for key in keys:
        match = next(
            ((position, index) for position, (index, value) in enumerate(unused) if value == key),
            None,
        )
        if match is None:
            selected_names.append("")
            continue
        position, original_index = match
        unused.pop(position)
        selected_names.append(
            display_names[original_index].strip() if original_index < len(display_names) else ""
        )
    named = [name for name in selected_names if name]
    encounter_name = (
        f"Conflict with {named[0]}"
        if len(keys) == 1 and named
        else f"Battle with {ENEMIES[keys[0]]['name']}s"
    )
    encounter = Encounter(campaign_id=campaign_id, name=encounter_name)
    session.add(encounter)
    session.flush()
    if encounter.id is None:
        raise RuntimeError("Encounter id was not generated")
    rng = SystemRandom()
    characters = session.exec(
        select(Character)
        .where(Character.campaign_id == campaign_id)
        .order_by(Character.role.desc())
    ).all()
    if not any(character.role == "protagonist" for character in characters):
        raise ValidationError("A protagonist is required before starting an encounter.")
    for character in characters:
        primary = (
            "dexterity" if character.character_class in {"Rogue", "Ranger", "Bard"} else "strength"
        )
        session.add(
            Combatant(
                encounter_id=encounter.id,
                character_id=character.id,
                name=character.name,
                side="party",
                lane="front"
                if character.character_class in {"Fighter", "Rogue", "Ranger"}
                else "back",
                initiative=rng.randint(1, 20) + ability_modifier(character.dexterity),
                max_hp=character.max_hp,
                current_hp=character.current_hp,
                armor_class=character.armor_class,
                dexterity_modifier=ability_modifier(character.dexterity),
                attack_bonus=ability_modifier(getattr(character, primary))
                + proficiency_bonus(character.level),
                damage_formula="1d8+2",
                save_dc=8
                + proficiency_bonus(character.level)
                + ability_modifier(getattr(character, primary)),
                conditions_json=character.conditions_json,
                resources_json=character.resources_json,
            )
        )
    for index, key in enumerate(keys, start=1):
        enemy = ENEMIES[key]
        session.add(
            Combatant(
                encounter_id=encounter.id,
                name=selected_names[index - 1]
                or (f"{enemy['name']} {index}" if len(keys) > 1 else enemy["name"]),
                side="enemy",
                lane=enemy["lane"],
                initiative=rng.randint(1, 20) + 2,
                max_hp=enemy["hp"],
                current_hp=enemy["hp"],
                armor_class=enemy["ac"],
                dexterity_modifier=2,
                attack_bonus=enemy["attack"],
                damage_formula=enemy["damage"],
            )
        )
    session.flush()
    encounter.turn_index = 0
    session.add(encounter)
    resolved_opening = run_opening_turns(session, encounter)
    if opening_events is not None:
        opening_events.extend(resolved_opening)
    return encounter


def run_opening_turns(session: Session, encounter: Encounter) -> list[str]:
    """Resolve automated combatants that beat the protagonist's initiative."""
    combatants = list(
        session.exec(
            select(Combatant)
            .where(Combatant.encounter_id == encounter.id)
            .order_by(Combatant.initiative.desc(), Combatant.id)
        ).all()
    )
    protagonist = next(
        (
            item
            for item in combatants
            if item.character_id
            and (character := session.get(Character, item.character_id))
            and character.role == "protagonist"
        ),
        None,
    )
    if not protagonist:
        return []
    protagonist_index = combatants.index(protagonist)
    events: list[str] = []
    for actor in combatants[:protagonist_index]:
        if actor.defeated or actor.current_hp <= 0:
            continue
        enemies = [item for item in combatants if item.side == "enemy" and not item.defeated]
        party = [item for item in combatants if item.side == "party" and not item.defeated]
        target = (
            min(enemies, key=lambda item: item.current_hp, default=None)
            if actor.side == "party"
            else next(
                (item for item in party if item.current_hp > 0 and item.lane == "front"),
                next((item for item in party if item.current_hp > 0), None),
            )
        )
        if not target:
            continue
        attack = roll_formula(f"1d20{actor.attack_bonus:+d}")
        if attack.total >= _effective_defense(target):
            damage = roll_formula(actor.damage_formula).total
            target.current_hp = max(0, target.current_hp - damage)
            events.append(f"{actor.name} hits {target.name} for {damage} damage.")
            session.add(target)
            if target.character_id:
                character = session.get(Character, target.character_id)
                if character:
                    character.current_hp = target.current_hp
                    session.add(character)
        else:
            events.append(f"{actor.name} misses {target.name}.")
        session.add(actor)
    encounter.turn_index = protagonist_index
    encounter.updated_at = now_utc()
    session.add(encounter)
    if events:
        session.add(
            Turn(campaign_id=encounter.campaign_id, speaker="Roll", content=" ".join(events))
        )
        _record_combat_events(session, encounter.campaign_id, encounter.id, events)
    return events


def _record_combat_events(
    session: Session, campaign_id: int, encounter_id: int | None, events: Sequence[str]
) -> None:
    if not events:
        return
    session.add(
        GameEvent(
            campaign_id=campaign_id,
            event_type="combat_resolved",
            payload_json=json.dumps(
                {"encounter_id": encounter_id, "events": list(events)}, ensure_ascii=False
            ),
        )
    )


def prepare_combat_action(
    session: Session,
    *,
    campaign_id: int,
    encounter_id: int,
    idempotency_key: str,
    action_id: str,
    target_id: int | None,
    destination_lane: str | None,
) -> tuple[PendingRoll, bool]:
    canonical_payload = json.dumps(
        {
            "action_id": action_id,
            "destination_lane": destination_lane,
            "encounter_id": encounter_id,
            "target_id": target_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    existing = session.exec(
        select(ActionRequest)
        .where(ActionRequest.campaign_id == campaign_id)
        .where(ActionRequest.idempotency_key == idempotency_key)
        .where(ActionRequest.operation == "combat_action")
    ).first()
    if existing:
        if existing.request_payload_json != canonical_payload:
            raise ConflictError(
                "The idempotency key was already used for a different combat action."
            )
        if existing.pending_roll_id:
            pending = session.get(PendingRoll, existing.pending_roll_id)
            if pending:
                return pending, True
    detail = encounter_detail(session, campaign_id)
    if not detail or detail["id"] != encounter_id or detail["status"] != "active":
        raise NotFoundError("Active encounter not found.")
    protagonist_id = detail["protagonist_combatant_id"]
    if protagonist_id != detail["current_combatant_id"]:
        raise ConflictError("It is not the protagonist's turn.")
    legal = {str(item["id"]) for item in detail["legal_actions"]}
    if action_id not in legal:
        raise ValidationError("That action is not currently available.")
    actor = session.get(Combatant, protagonist_id)
    if not actor or not actor.character_id:
        raise ConflictError("The protagonist combatant is unavailable.")
    character = session.get(Character, actor.character_id)
    if not character:
        raise ConflictError("The protagonist character is unavailable.")
    definition = COMBAT_ACTIONS[action_id]
    target = session.get(Combatant, target_id) if target_id else None
    if action_id == "move_lane":
        if "restrained" in _conditions(actor):
            raise ConflictError("A restrained combatant cannot change lanes.")
        if destination_lane not in {"front", "back"}:
            raise ValidationError("Choose the front or back lane.")
    target_type = definition.get("target")
    if target_type == "self":
        target = actor
    elif target_type == "party":
        target = None
    elif target_type == "ally":
        if not target or target.encounter_id != encounter_id or target.side != "party":
            target = actor
    if target_type == "enemy":
        if (
            not target
            or target.encounter_id != encounter_id
            or target.side != "enemy"
            or target.defeated
        ):
            raise ValidationError("Choose a valid enemy target.")
        if actor.lane == "front" and target.lane == "back":
            enemy_front = any(
                item["side"] == "enemy" and item["lane"] == "front" and not item["defeated"]
                for item in detail["combatants"]
            )
            if enemy_front and definition.get("range") != "any":
                raise ValidationError("A front-lane enemy blocks that melee target.")
    resources = normalise_resources(
        character.character_class, character.level, json.loads(character.resources_json)
    )
    spell_level = str(definition.get("spell_level", ""))
    if spell_level:
        if resources.get("spell_slots", {}).get(spell_level, 0) <= 0:
            raise ValidationError("No spell slot remains for that action.")
        resources["spell_slots"][spell_level] -= 1
    if definition.get("uses"):
        remaining = resources.get("class_uses", {}).get(action_id, 0)
        if remaining <= 0:
            raise ValidationError("That class feature has no uses remaining.")
        resources["class_uses"][action_id] = remaining - 1
    character.resources_json = json.dumps(resources)
    actor.resources_json = character.resources_json
    session.add(character)
    session.add(actor)
    request = ActionRequest(
        campaign_id=campaign_id,
        idempotency_key=idempotency_key,
        operation="combat_action",
        status="processing",
        action_text=action_id,
        request_payload_json=canonical_payload,
    )
    session.add(request)
    session.flush()
    context = {
        "encounter_id": encounter_id,
        "actor_id": actor.id,
        "target_id": target.id if target else None,
        "action_id": action_id,
        "destination_lane": destination_lane,
        "applied": False,
    }
    pending = PendingRoll(
        campaign_id=campaign_id,
        action_request_id=request.id,
        action_text=action_id.replace("_", " "),
        formula=_action_roll_formula(actor, definition),
        ability="Combat",
        skill=action_id.replace("_", " ").title(),
        dc=(
            _effective_defense(target)
            if target and definition.get("kind") == "attack"
            else 10
            if definition.get("kind") in {"check", "death_save"}
            else 1
        ),
        reason=f"Resolve {action_id.replace('_', ' ')}.",
        narration="The outcome is ready for your roll.",
        purpose="combat_action",
        context_json=json.dumps(context),
    )
    session.add(pending)
    session.flush()
    request.pending_roll_id = pending.id
    request.status = "roll_required"
    session.add(request)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        winner = session.exec(
            select(ActionRequest)
            .where(ActionRequest.campaign_id == campaign_id)
            .where(ActionRequest.idempotency_key == idempotency_key)
            .where(ActionRequest.operation == "combat_action")
        ).first()
        if winner and winner.request_payload_json == canonical_payload and winner.pending_roll_id:
            replay = session.get(PendingRoll, winner.pending_roll_id)
            if replay:
                return replay, True
        if winner:
            raise ConflictError(
                "The idempotency key was already used for a different combat action."
            ) from exc
        raise ConflictError("Another campaign operation is already in progress.") from exc
    session.refresh(pending)
    return pending, False


def apply_combat_roll(session: Session, pending_roll_id: int, roll: DiceRoll) -> list[str]:
    pending = session.get(PendingRoll, pending_roll_id)
    if not pending or pending.purpose != "combat_action":
        return []
    context = json.loads(pending.context_json)
    if context.get("applied"):
        return list(context.get("events", []))
    encounter = session.get(Encounter, context["encounter_id"])
    actor = session.get(Combatant, context["actor_id"])
    target = session.get(Combatant, context.get("target_id")) if context.get("target_id") else None
    if not encounter or not actor:
        raise ConflictError("The encounter state is unavailable.")
    events: list[str] = []
    action_id = context["action_id"]
    definition = COMBAT_ACTIONS[action_id]
    kind = str(definition["kind"])
    if kind == "death_save":
        natural = int(json.loads(roll.rolls_json)[0])
        if natural == 20:
            actor.current_hp = 1
            actor.death_successes = actor.death_failures = 0
            events.append(f"{actor.name} rolls a natural 20 and rises with 1 HP.")
        elif natural == 1:
            actor.death_failures = min(3, actor.death_failures + 2)
            events.append(f"{actor.name} suffers two death-save failures.")
        elif natural >= 10:
            actor.death_successes = min(3, actor.death_successes + 1)
            events.append(f"{actor.name} succeeds on a death save.")
        else:
            actor.death_failures = min(3, actor.death_failures + 1)
            events.append(f"{actor.name} fails a death save.")
        if actor.death_successes >= 3:
            events.append(f"{actor.name} stabilizes.")
            _recover_party(session, encounter, events, "The party was rescued after the battle.")
        elif actor.death_failures >= 3:
            events.append(f"{actor.name} slips toward death.")
            _recover_party(session, encounter, events, "Defeat leaves a lasting wound and setback.")
    elif action_id == "flee" and roll.outcome == "success":
        encounter.status = "fled"
        events.append(f"{actor.name} leads the party to safety.")
        _clear_encounter_conditions(session, encounter)
    elif action_id == "flee":
        events.append(f"{actor.name} cannot find a safe route out.")
    elif kind == "move":
        actor.lane = str(context.get("destination_lane") or actor.lane)
        events.append(f"{actor.name} moves to the {actor.lane} lane.")
    elif kind == "heal" and target:
        healing_total = roll.total + (
            _character_level(session, actor) if action_id == "second_wind" else 0
        )
        healed = min(target.max_hp - target.current_hp, max(1, healing_total))
        target.current_hp += healed
        target.death_successes = target.death_failures = 0
        events.append(f"{actor.name} restores {healed} HP to {target.name}.")
        session.add(target)
    elif kind == "condition":
        targets = (
            [item for item in _combatants(session, encounter.id) if item.side == "party"]
            if definition.get("target") == "party"
            else [target or actor]
        )
        for recipient in targets:
            _add_condition(recipient, str(definition["condition"]))
            session.add(recipient)
        events.append(f"{definition['label']} takes effect.")
    elif kind in {"auto_damage", "area_damage"}:
        targets = (
            [
                item
                for item in _combatants(session, encounter.id)
                if item.side == "enemy" and not item.defeated
            ]
            if kind == "area_damage"
            else [target]
            if target
            else []
        )
        for recipient in targets:
            recipient.current_hp = max(0, recipient.current_hp - max(1, roll.total))
            _remove_condition(recipient, "sleeping")
            recipient.defeated = recipient.current_hp == 0
            session.add(recipient)
        events.append(f"{actor.name} uses {definition['label']} for {max(1, roll.total)} damage.")
    elif target:
        if roll.outcome == "success":
            damage_formula = str(definition.get("damage", actor.damage_formula))
            if action_id == "sneak_attack":
                dice = 1 + (_character_level(session, actor) - 1) // 2
                damage_formula = f"{dice}d6"
            if (
                action_id == "weapon_attack"
                and _character_level(session, actor) >= 5
                and _character_class(session, actor) in {"Fighter", "Ranger"}
            ):
                first = roll_formula(damage_formula).total
                second = roll_formula(damage_formula).total
                damage_total = first + second
            else:
                damage_total = roll_formula(damage_formula).total
            natural = int(json.loads(roll.rolls_json)[0]) if roll.formula.startswith("1d20") else 0
            if natural == 20:
                damage_total += roll_formula(damage_formula).total
                events.append("Critical hit!")
            if "surging" in _conditions(actor):
                damage_total += roll_formula(actor.damage_formula).total
                _remove_condition(actor, "surging")
            if "marked" in _conditions(target):
                damage_total += roll_formula("1d6").total
            target.current_hp = max(0, target.current_hp - damage_total)
            _remove_condition(target, "sleeping")
            target.defeated = target.current_hp == 0
            events.append(f"{actor.name} hits {target.name} for {damage_total} damage.")
            if definition.get("condition") and not target.defeated:
                _add_condition(target, str(definition["condition"]))
        else:
            events.append(f"{actor.name} misses {target.name}.")
        session.add(target)
    _remove_condition(actor, "restrained")
    session.add(actor)
    if encounter.status == "active":
        _run_automatic_turns(session, encounter, events)
    context["applied"] = True
    context["events"] = events
    pending.context_json = json.dumps(context)
    session.add(pending)
    session.add(encounter)
    for combatant in _combatants(session, encounter.id):
        if combatant.character_id:
            character = session.get(Character, combatant.character_id)
            if character:
                character.current_hp = combatant.current_hp
                character.conditions_json = combatant.conditions_json
                character.resources_json = combatant.resources_json
                session.add(character)
    session.add(Turn(campaign_id=pending.campaign_id, speaker="Roll", content=" ".join(events)))
    _record_combat_events(session, pending.campaign_id, encounter.id, events)
    session.commit()
    return events


def _run_automatic_turns(session: Session, encounter: Encounter, events: list[str]) -> None:
    combatants = list(
        session.exec(
            select(Combatant)
            .where(Combatant.encounter_id == encounter.id)
            .order_by(Combatant.initiative.desc(), Combatant.id)
        ).all()
    )
    protagonist = next(
        (
            item
            for item in combatants
            if item.character_id and session.get(Character, item.character_id).role == "protagonist"
        ),
        None,
    )
    if not protagonist:
        _recover_party(session, encounter, events, "The party was overwhelmed.")
        return
    protagonist_index = combatants.index(protagonist)
    turn_order = combatants[protagonist_index + 1 :] + combatants[:protagonist_index]
    for actor in turn_order:
        if actor.defeated or actor.current_hp <= 0:
            continue
        if "sleeping" in _conditions(actor):
            _remove_condition(actor, "sleeping")
            events.append(f"{actor.name} is sleeping and loses the turn.")
            session.add(actor)
            continue
        enemies = [item for item in combatants if item.side == "enemy" and not item.defeated]
        if not enemies:
            encounter.status = "victory"
            events.append("The party wins the encounter.")
            _clear_encounter_conditions(session, encounter)
            return
        party = [item for item in combatants if item.side == "party" and not item.defeated]
        if actor.side == "party":
            target = min(enemies, key=lambda item: item.current_hp, default=None)
        else:
            standing = [item for item in party if item.current_hp > 0]
            target = next(
                (item for item in standing if item.lane == "front"),
                standing[0] if standing else None,
            )
        if not target:
            continue
        attack = roll_formula(f"1d20{actor.attack_bonus:+d}")
        defense = _effective_defense(target)
        target_conditions = _conditions(target)
        if "exposed" in target_conditions or "blessed" in _conditions(actor):
            attack = type(attack)(attack.formula, attack.rolls, attack.modifier, attack.total + 2)
        if attack.total >= defense:
            damage = roll_formula(actor.damage_formula).total
            if "hindered" in _conditions(actor):
                damage = max(0, damage - 2)
            target.current_hp = max(0, target.current_hp - damage)
            _remove_condition(target, "sleeping")
            if target.side == "enemy":
                target.defeated = target.current_hp == 0
            events.append(f"{actor.name} hits {target.name} for {damage} damage.")
            session.add(target)
            if target.character_id:
                character = session.get(Character, target.character_id)
                if character:
                    character.current_hp = target.current_hp
                    session.add(character)
        _remove_condition(actor, "restrained")
        session.add(actor)
    protagonist.conditions_json = json.dumps(
        [
            condition
            for condition in _conditions(protagonist)
            if condition not in {"dodging", "shielded", "inspired", "hindered"}
        ]
    )
    encounter.round_number += 1
    encounter.turn_index = next(
        (index for index, item in enumerate(combatants) if item.id == protagonist.id), 0
    )
    encounter.updated_at = now_utc()
    session.add(protagonist)


def _action_roll_formula(actor: Combatant, definition: dict[str, object]) -> str:
    kind = definition["kind"]
    if kind in {"heal"}:
        return str(definition["effect"])
    if kind in {"auto_damage", "area_damage"}:
        return str(definition["damage"])
    if kind == "attack":
        bonus = actor.attack_bonus
        if "blessed" in _conditions(actor) or "inspired" in _conditions(actor):
            bonus += 2
        if "restrained" in _conditions(actor):
            bonus -= 2
        return f"1d20{bonus:+d}"
    return "1d20"


def _action_description(definition: dict[str, object]) -> str:
    pieces = [str(definition["kind"]).replace("_", " ")]
    if definition.get("damage"):
        pieces.append(f"{definition['damage']} damage")
    if definition.get("effect"):
        pieces.append(f"{definition['effect']} healing")
    if definition.get("condition"):
        pieces.append(str(definition["condition"]))
    return " · ".join(pieces).capitalize()


def _combatants(session: Session, encounter_id: int | None) -> list[Combatant]:
    if encounter_id is None:
        return []
    return list(session.exec(select(Combatant).where(Combatant.encounter_id == encounter_id)).all())


def _conditions(combatant: Combatant) -> list[str]:
    try:
        return [str(value) for value in json.loads(combatant.conditions_json)]
    except (json.JSONDecodeError, TypeError):
        return []


def _add_condition(combatant: Combatant, condition: str) -> None:
    values = _conditions(combatant)
    if condition not in values:
        values.append(condition)
    combatant.conditions_json = json.dumps(values)


def _remove_condition(combatant: Combatant, condition: str) -> None:
    combatant.conditions_json = json.dumps(
        [value for value in _conditions(combatant) if value != condition]
    )


def _effective_defense(combatant: Combatant) -> int:
    defense = combatant.armor_class
    conditions = _conditions(combatant)
    if "mage_armor" in conditions:
        defense = max(defense, 13 + combatant.dexterity_modifier)
    if "dodging" in conditions or "shielded" in conditions:
        defense += 2
    if "restrained" in conditions:
        defense -= 2
    return max(1, defense)


def _clear_encounter_conditions(session: Session, encounter: Encounter) -> None:
    for combatant in _combatants(session, encounter.id):
        remaining = [
            value for value in _conditions(combatant) if value not in _ENCOUNTER_CONDITIONS
        ]
        combatant.conditions_json = json.dumps(remaining)
        session.add(combatant)
        if combatant.character_id:
            character = session.get(Character, combatant.character_id)
            if character:
                character.conditions_json = combatant.conditions_json
                session.add(character)


def _character_level(session: Session, combatant: Combatant) -> int:
    character = session.get(Character, combatant.character_id) if combatant.character_id else None
    return character.level if character else 1


def _character_class(session: Session, combatant: Combatant) -> str:
    character = session.get(Character, combatant.character_id) if combatant.character_id else None
    return character.character_class if character else ""


def _recover_party(session: Session, encounter: Encounter, events: list[str], reason: str) -> None:
    encounter.status = "defeat"
    _clear_encounter_conditions(session, encounter)
    for combatant in _combatants(session, encounter.id):
        if not combatant.character_id:
            continue
        character = session.get(Character, combatant.character_id)
        if not character:
            continue
        recovered = max(1, (character.max_hp + 3) // 4)
        character.current_hp = recovered
        conditions = set(json.loads(character.conditions_json))
        conditions.add("wounded")
        character.conditions_json = json.dumps(sorted(conditions))
        combatant.current_hp = recovered
        combatant.conditions_json = character.conditions_json
        combatant.death_successes = combatant.death_failures = 0
        session.add(character)
        session.add(combatant)
    state = session.exec(
        select(WorldState).where(WorldState.campaign_id == encounter.campaign_id)
    ).first()
    if state:
        state.current_location = "A place of recovery"
        state.active_objective = "Recover and reassess the setback."
        facts = list(json.loads(state.facts_json))
        if reason not in facts:
            facts.append(reason)
        state.facts_json = json.dumps(facts[-20:])
        session.add(state)
    events.append(reason)


def balanced_enemy_keys(session: Session, campaign_id: int, enemy_keys: Sequence[str]) -> list[str]:
    characters = session.exec(select(Character).where(Character.campaign_id == campaign_id)).all()
    budget = max(1, sum(character.level for character in characters))
    limit = 2 if len(characters) == 1 else 4
    costs = {"ogre": 6}
    chosen: list[str] = []
    spent = 0
    for key in enemy_keys:
        if key not in ENEMIES or len(chosen) >= limit:
            continue
        cost = costs.get(key, 1)
        if spent + cost <= budget:
            chosen.append(key)
            spent += cost
    return chosen or ["bandit"]
