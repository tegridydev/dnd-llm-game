from __future__ import annotations

from dndllm26.game.world import _merge_npcs


def test_npc_memory_replaces_descriptor_and_plural_variants() -> None:
    merged = _merge_npcs(
        ("Soldiers (mismatched armor)", "Captain Rusk (watchful)"),
        ["Soldier (charging)", "Captain Rusk (wounded)"],
    )
    assert merged == ("Soldier (charging)", "Captain Rusk (wounded)")
