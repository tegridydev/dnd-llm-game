from __future__ import annotations

from sqlmodel import Session, select

from dndllm26.db.models import Character, Hero, Quest
from dndllm26.game.campaigns import create_campaign
from dndllm26.game.catalog import ABILITIES, build_hero_sheet
from dndllm26.game.progression import complete_quest, rest_party


def test_quest_milestone_levels_party_and_long_rest_restores_hp(engine) -> None:
    with Session(engine) as session:
        sheet = build_hero_sheet("Human", "Wizard")
        hero = Hero(
            name="Ilyra",
            ancestry="Human",
            character_class="Wizard",
            backstory="A scholar.",
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
        )
        session.add(hero)
        session.flush()
        campaign = create_campaign(
            session,
            title="Milestones",
            setting="Tower",
            tone="heroic",
            protagonist_id=hero.id or 0,
            companion_ids=[],
            lore_document_ids=[],
        )
        quest = session.exec(select(Quest).where(Quest.campaign_id == campaign.id)).first()
        assert quest is not None and quest.id is not None
        result = complete_quest(session, campaign.id or 0, quest.id)
        assert result["level"] == 2
        character = session.exec(
            select(Character).where(Character.campaign_id == campaign.id)
        ).first()
        assert character is not None
        assert character.level == 2
        character.current_hp = 1
        session.add(character)
        session.commit()
        rest_party(session, campaign.id or 0, "long", 0)
        session.refresh(character)
        assert character.current_hp == character.max_hp
