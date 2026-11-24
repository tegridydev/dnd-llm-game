from __future__ import annotations

from dndllm26.game.world import _merge_npcs


def test_npc_memory_replaces_descriptor_and_plural_variants() -> None:
    merged = _merge_npcs(
        ("Soldiers (mismatched armor)", "Captain Rusk (watchful)"),
        ["Soldier (charging)", "Captain Rusk (wounded)"],
    )
    assert merged == ("Soldier (charging)", "Captain Rusk (wounded)")


def test_full_memory_keeps_the_latest_discovery():
    from dndllm26.game.world import _merge_memory

    previous = tuple(f"Location clue {n}" for n in range(20))
    current = _merge_memory(previous, ["The librarian has a key."])
    assert len(current) == 20
    assert previous[0] not in current
    assert current[-1] == "The librarian has a key."
