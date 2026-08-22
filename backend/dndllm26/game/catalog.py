from __future__ import annotations

from typing import Any


ABILITIES = ("strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma")
STANDARD_ARRAY = (15, 14, 13, 12, 10, 8)

ANCESTRIES: dict[str, dict[str, Any]] = {
    "Human": {"bonuses": {ability: 1 for ability in ABILITIES}, "speed": 30},
    "Elf": {"bonuses": {"dexterity": 2, "intelligence": 1}, "speed": 30},
    "Dwarf": {"bonuses": {"constitution": 2, "wisdom": 1}, "speed": 25},
    "Halfling": {"bonuses": {"dexterity": 2, "charisma": 1}, "speed": 25},
}

CLASSES: dict[str, dict[str, Any]] = {
    "Fighter": {
        "array": (15, 14, 14, 10, 12, 8),
        "hit_die": 10,
        "ac": 16,
        "saves": ["strength", "constitution"],
        "skills": ["Athletics", "Perception"],
        "equipment": ["longsword", "shield", "chain mail", "healing potion"],
        "actions": ["weapon_attack", "second_wind", "action_surge"],
    },
    "Rogue": {
        "array": (10, 15, 12, 14, 13, 8),
        "hit_die": 8,
        "ac": 14,
        "saves": ["dexterity", "intelligence"],
        "skills": ["Stealth", "Investigation", "Perception"],
        "equipment": ["shortsword", "shortbow", "leather armor", "thieves' tools"],
        "actions": ["weapon_attack", "sneak_attack", "cunning_action"],
    },
    "Cleric": {
        "array": (13, 10, 14, 8, 15, 12),
        "hit_die": 8,
        "ac": 16,
        "saves": ["wisdom", "charisma"],
        "skills": ["Insight", "Medicine", "Religion"],
        "equipment": ["mace", "shield", "scale mail", "holy symbol"],
        "actions": [
            "weapon_attack",
            "sacred_flame",
            "cure_wounds",
            "healing_word",
            "guiding_bolt",
            "bless",
        ],
    },
    "Wizard": {
        "array": (8, 14, 13, 15, 12, 10),
        "hit_die": 6,
        "ac": 12,
        "saves": ["intelligence", "wisdom"],
        "skills": ["Arcana", "Investigation"],
        "equipment": ["quarterstaff", "spellbook", "component pouch"],
        "actions": [
            "fire_bolt",
            "magic_missile",
            "shield",
            "mage_armor",
            "burning_hands",
            "sleep",
            "scorching_ray",
            "misty_step",
            "web",
            "fireball",
        ],
    },
    "Ranger": {
        "array": (12, 15, 14, 8, 13, 10),
        "hit_die": 10,
        "ac": 14,
        "saves": ["strength", "dexterity"],
        "skills": ["Survival", "Perception", "Stealth"],
        "equipment": ["longbow", "shortsword", "leather armor", "healer's kit"],
        "actions": ["weapon_attack", "hunters_mark", "cure_wounds", "hail_of_thorns"],
    },
    "Bard": {
        "array": (8, 14, 12, 10, 13, 15),
        "hit_die": 8,
        "ac": 13,
        "saves": ["dexterity", "charisma"],
        "skills": ["Persuasion", "Performance", "Insight"],
        "equipment": ["rapier", "leather armor", "lute", "healing potion"],
        "actions": [
            "weapon_attack",
            "vicious_mockery",
            "bardic_inspiration",
            "healing_word",
            "dissonant_whispers",
            "faerie_fire",
            "shatter",
        ],
    },
}

SKILL_ABILITIES = {
    "Athletics": "strength",
    "Acrobatics": "dexterity",
    "Sleight of Hand": "dexterity",
    "Stealth": "dexterity",
    "Arcana": "intelligence",
    "History": "intelligence",
    "Investigation": "intelligence",
    "Nature": "intelligence",
    "Religion": "intelligence",
    "Animal Handling": "wisdom",
    "Insight": "wisdom",
    "Medicine": "wisdom",
    "Perception": "wisdom",
    "Survival": "wisdom",
    "Deception": "charisma",
    "Intimidation": "charisma",
    "Performance": "charisma",
    "Persuasion": "charisma",
}

ENEMIES: dict[str, dict[str, Any]] = {
    "bandit": {
        "name": "Bandit",
        "hp": 11,
        "ac": 12,
        "attack": 3,
        "damage": "1d6+1",
        "lane": "front",
    },
    "goblin": {"name": "Goblin", "hp": 7, "ac": 15, "attack": 4, "damage": "1d6+2", "lane": "back"},
    "wolf": {"name": "Wolf", "hp": 11, "ac": 13, "attack": 4, "damage": "2d4+2", "lane": "front"},
    "skeleton": {
        "name": "Skeleton",
        "hp": 13,
        "ac": 13,
        "attack": 4,
        "damage": "1d6+2",
        "lane": "front",
    },
    "cultist": {
        "name": "Cultist",
        "hp": 9,
        "ac": 12,
        "attack": 3,
        "damage": "1d6+1",
        "lane": "front",
    },
    "ogre": {"name": "Ogre", "hp": 59, "ac": 11, "attack": 6, "damage": "2d8+4", "lane": "front"},
}

# Engine-owned action definitions. The model may describe these effects but never changes them.
COMBAT_ACTIONS: dict[str, dict[str, Any]] = {
    "weapon_attack": {
        "label": "Weapon Attack",
        "kind": "attack",
        "target": "enemy",
        "damage": "1d8+2",
    },
    "move_lane": {"label": "Change Lane", "kind": "move", "target": "none"},
    "dodge": {"label": "Dodge", "kind": "condition", "target": "self", "condition": "dodging"},
    "flee": {"label": "Attempt to Flee", "kind": "check", "target": "none"},
    "second_wind": {
        "label": "Second Wind",
        "kind": "heal",
        "target": "self",
        "effect": "1d10",
        "uses": 1,
    },
    "action_surge": {
        "label": "Action Surge",
        "kind": "condition",
        "target": "self",
        "condition": "surging",
        "min_level": 2,
        "uses": 1,
    },
    "sneak_attack": {
        "label": "Sneak Attack",
        "kind": "attack",
        "target": "enemy",
        "damage": "2d6",
        "min_level": 1,
    },
    "cunning_action": {"label": "Cunning Action", "kind": "move", "target": "none", "min_level": 2},
    "sacred_flame": {
        "label": "Sacred Flame",
        "kind": "attack",
        "target": "enemy",
        "damage": "1d8",
        "range": "any",
    },
    "cure_wounds": {
        "label": "Cure Wounds",
        "kind": "heal",
        "target": "ally",
        "effect": "1d8+3",
        "spell_level": 1,
    },
    "healing_word": {
        "label": "Healing Word",
        "kind": "heal",
        "target": "ally",
        "effect": "1d4+3",
        "spell_level": 1,
        "range": "any",
    },
    "guiding_bolt": {
        "label": "Guiding Bolt",
        "kind": "attack",
        "target": "enemy",
        "damage": "4d6",
        "spell_level": 1,
        "range": "any",
    },
    "bless": {
        "label": "Bless",
        "kind": "condition",
        "target": "party",
        "condition": "blessed",
        "spell_level": 1,
    },
    "fire_bolt": {
        "label": "Fire Bolt",
        "kind": "attack",
        "target": "enemy",
        "damage": "1d10",
        "range": "any",
    },
    "magic_missile": {
        "label": "Magic Missile",
        "kind": "auto_damage",
        "target": "enemy",
        "damage": "3d4+3",
        "spell_level": 1,
        "range": "any",
    },
    "shield": {
        "label": "Shield",
        "kind": "condition",
        "target": "self",
        "condition": "shielded",
        "spell_level": 1,
    },
    "mage_armor": {
        "label": "Mage Armor",
        "kind": "condition",
        "target": "self",
        "condition": "mage_armor",
        "spell_level": 1,
    },
    "burning_hands": {
        "label": "Burning Hands",
        "kind": "area_damage",
        "target": "enemy",
        "damage": "3d6",
        "spell_level": 1,
        "min_level": 2,
    },
    "sleep": {
        "label": "Sleep",
        "kind": "condition",
        "target": "enemy",
        "condition": "sleeping",
        "spell_level": 1,
    },
    "scorching_ray": {
        "label": "Scorching Ray",
        "kind": "attack",
        "target": "enemy",
        "damage": "6d6",
        "spell_level": 2,
        "min_level": 3,
        "range": "any",
    },
    "misty_step": {
        "label": "Misty Step",
        "kind": "move",
        "target": "none",
        "spell_level": 2,
        "min_level": 3,
    },
    "web": {
        "label": "Web",
        "kind": "condition",
        "target": "enemy",
        "condition": "restrained",
        "spell_level": 2,
        "min_level": 3,
        "range": "any",
    },
    "fireball": {
        "label": "Fireball",
        "kind": "area_damage",
        "target": "enemy",
        "damage": "8d6",
        "spell_level": 3,
        "min_level": 5,
        "range": "any",
    },
    "hunters_mark": {
        "label": "Hunter's Mark",
        "kind": "condition",
        "target": "enemy",
        "condition": "marked",
        "spell_level": 1,
        "min_level": 2,
        "range": "any",
    },
    "hail_of_thorns": {
        "label": "Hail of Thorns",
        "kind": "attack",
        "target": "enemy",
        "damage": "2d8",
        "spell_level": 1,
        "min_level": 2,
        "range": "any",
    },
    "vicious_mockery": {
        "label": "Vicious Mockery",
        "kind": "attack",
        "target": "enemy",
        "damage": "1d4",
        "condition": "hindered",
        "range": "any",
    },
    "bardic_inspiration": {
        "label": "Bardic Inspiration",
        "kind": "condition",
        "target": "ally",
        "condition": "inspired",
        "uses": 2,
    },
    "dissonant_whispers": {
        "label": "Dissonant Whispers",
        "kind": "attack",
        "target": "enemy",
        "damage": "3d6",
        "spell_level": 1,
        "range": "any",
    },
    "faerie_fire": {
        "label": "Faerie Fire",
        "kind": "condition",
        "target": "enemy",
        "condition": "exposed",
        "spell_level": 1,
        "range": "any",
    },
    "shatter": {
        "label": "Shatter",
        "kind": "area_damage",
        "target": "enemy",
        "damage": "3d8",
        "spell_level": 2,
        "min_level": 3,
        "range": "any",
    },
    "death_save": {"label": "Death Save", "kind": "death_save", "target": "self"},
}


def spell_slots(character_class: str, level: int) -> dict[str, int]:
    if character_class in {"Cleric", "Wizard", "Bard"}:
        rows = {1: (2, 0, 0), 2: (3, 0, 0), 3: (4, 2, 0), 4: (4, 3, 0), 5: (4, 3, 2)}
    elif character_class == "Ranger" and level >= 2:
        rows = {2: (2, 0, 0), 3: (3, 0, 0), 4: (3, 0, 0), 5: (4, 2, 0)}
    else:
        return {}
    return {str(index): value for index, value in enumerate(rows[level], 1) if value}


def normalise_resources(character_class: str, level: int, stored: dict[str, Any]) -> dict[str, Any]:
    resources = dict(stored)
    maximum_slots = spell_slots(character_class, level)
    slots = dict(resources.get("spell_slots", maximum_slots))
    resources["spell_slots"] = {
        key: max(0, min(int(slots.get(key, value)), value)) for key, value in maximum_slots.items()
    }
    resources.setdefault("class_uses", {})
    for action_id in CLASSES[character_class]["actions"]:
        uses = COMBAT_ACTIONS.get(action_id, {}).get("uses")
        if uses:
            resources["class_uses"].setdefault(action_id, uses)
    return resources


def ability_modifier(score: int) -> int:
    return (score - 10) // 2


def proficiency_bonus(level: int) -> int:
    return 2 if level <= 4 else 3


def build_hero_sheet(
    ancestry: str,
    character_class: str,
    scores: dict[str, int] | None = None,
    *,
    apply_ancestry_bonuses: bool = True,
) -> dict[str, Any]:
    ancestry_data = ANCESTRIES.get(ancestry, ANCESTRIES["Human"])
    class_data = CLASSES.get(character_class, CLASSES["Fighter"])
    base = (
        dict(zip(ABILITIES, class_data["array"], strict=True))
        if scores is None
        else {ability: int(scores.get(ability, 10)) for ability in ABILITIES}
    )
    final = {
        ability: max(
            3,
            min(
                18,
                base[ability]
                + (ancestry_data["bonuses"].get(ability, 0) if apply_ancestry_bonuses else 0),
            ),
        )
        for ability in ABILITIES
    }
    hp = class_data["hit_die"] + ability_modifier(final["constitution"])
    primary = (
        "dexterity"
        if character_class in {"Rogue", "Ranger", "Bard"}
        else "wisdom"
        if character_class == "Cleric"
        else "intelligence"
        if character_class == "Wizard"
        else "strength"
    )
    return {
        **final,
        "level": 1,
        "max_hp": max(1, hp),
        "armor_class": class_data["ac"],
        "speed": ancestry_data["speed"],
        "skills": list(class_data["skills"]),
        "saves": list(class_data["saves"]),
        "inventory": list(class_data["equipment"]),
        "spells": [
            action
            for action in class_data["actions"]
            if action
            not in {
                "weapon_attack",
                "sneak_attack",
                "cunning_action",
                "second_wind",
                "action_surge",
            }
        ],
        "resources": normalise_resources(
            character_class,
            1,
            {
                "hit_dice": 1,
                "primary_ability": primary,
                "class_actions": list(class_data["actions"]),
            },
        ),
    }
