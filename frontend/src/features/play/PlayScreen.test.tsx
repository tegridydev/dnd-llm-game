// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "vitest-axe";
import { afterEach, describe, expect, test, vi } from "vitest";

import type { CampaignDetail } from "../../types";
import { initialPlayState } from "./state";
import type { PlayState } from "./state";
import { PlayScreen } from "./PlayScreen";

afterEach(cleanup);

const detail: CampaignDetail = {
  campaign: {
    id: 1,
    title: "The Gate",
    setting: "A sealed ruin",
    tone: "heroic",
    milestone_points: 0,
    created_at: "2026-08-21T00:00:00Z",
    updated_at: "2026-08-21T00:00:00Z",
    archived_at: null
  },
  characters: [],
  turns: [{
    id: 1,
    campaign_id: 1,
    speaker: "DM",
    content: "The gate waits in silence.",
    created_at: "2026-08-21T00:00:00Z"
  }],
  world_state: {
    id: 1,
    campaign_id: 1,
    current_location: "Old Gate",
    active_objective: "Enter",
    scene_summary: "The gate is sealed.",
    choices_json: "[]",
    facts_json: "[]",
    npcs_json: "[]",
    updated_at: "2026-08-21T00:00:00Z"
  },
  choices: [],
  pending_roll: null,
  recoverable_operation: null,
  opening: { status: "complete", idempotency_key: null, error: null, retryable: false },
  last_roll: null,
  encounter: null,
  quests: [{ id: 1, title: "Open the Gate", objective: "Find the key", status: "active", milestone_reward: 1 }],
  turn_page: { has_more: true, next_cursor: "cursor", limit: 20 }
};

function renderScreen(
  mutationPending: string | null,
  streaming = false,
  stateOverrides: Partial<PlayState> = {},
  utilityDrawer: "party" | "journal" | null = mutationPending ? "journal" : null,
  screenDetail: CampaignDetail = detail
) {
  return render(
    <PlayScreen
      detail={screenDetail}
      state={{
        ...initialPlayState,
        phase: streaming ? "generating" : "ready",
        streaming,
        draft: streaming ? "The gate begins to open." : "",
        statusMessage: streaming ? "The DM is shaping the scene..." : "Ready",
        ...stateOverrides
      }}
      mutationPending={mutationPending}
      onSetAction={vi.fn()}
      onSubmitAction={vi.fn(async () => undefined)}
      onSubmitCombatAction={vi.fn(async () => undefined)}
      onResolvePendingRoll={vi.fn(async () => undefined)}
      onGenerateOpening={vi.fn(async () => undefined)}
      onRetry={vi.fn(async () => undefined)}
      onCancel={vi.fn()}
      onClearError={vi.fn()}
      onLoadOlderTurns={vi.fn(async () => undefined)}
      onRest={vi.fn(async () => undefined)}
      onCompleteQuest={vi.fn(async () => undefined)}
      onCreateCampaign={vi.fn()}
      onOpenParty={vi.fn()}
      onOpenJournal={vi.fn()}
      utilityDrawer={utilityDrawer}
      onCloseUtilityDrawer={vi.fn()}
      maxActionChars={2_000}
    />
  );
}

describe("PlayScreen recovery and accessibility", () => {
  test("shows an opening loader instead of an empty action prompt", () => {
    renderScreen(
      null,
      true,
      {
        draft: "",
        retry: { kind: "opening", campaignId: 1, idempotencyKey: "opening-test-key" }
      },
      null,
      {
        ...detail,
        turns: [],
        opening: { status: "processing", idempotency_key: "opening-test-key", error: null, retryable: false }
      }
    );
    expect(screen.getByLabelText("Opening scene generation")).toHaveTextContent(
      "The Dungeon Master is setting the stage"
    );
    expect(screen.queryByText("Describe what your character does to begin.")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Describe your character action")).not.toBeInTheDocument();
  });

  test("keeps token narration outside live regions", () => {
    const { container } = renderScreen(null, true);
    expect(container.querySelector(".scene-viewport")).not.toHaveAttribute("aria-live");
    expect(container.querySelector(".context-strip")).not.toHaveAttribute("aria-live");
    expect(container.querySelector(".status-pill")).toHaveAttribute("aria-live", "polite");
  });

  test("shows campaign context in one readable status strip", () => {
    renderScreen(null, false, {}, null, {
      ...detail,
      characters: [{
        id: 2, campaign_id: 1, role: "protagonist", name: "Mira Voss",
        ancestry: "Human", character_class: "Rogue", backstory: "A scout.", level: 1,
        strength: 10, dexterity: 16, constitution: 12,
        intelligence: 12, wisdom: 12, charisma: 10, max_hp: 9, current_hp: 9,
        armor_class: 14, speed: 30, skills_json: "[]", saves_json: "[]", spells_json: "[]",
        inventory_json: "[]", resources_json: "{}", conditions_json: "[]"
      }]
    });
    const strip = screen.getByRole("region", { name: "Campaign status" });
    expect(strip).toHaveTextContent("Mira Voss · 9/9 HP · AC 14");
    expect(strip).toHaveTextContent("Old Gate");
    expect(strip).toHaveTextContent("Enter");
  });

  test("uses distinct acting and target states without a disabled combat composer", async () => {
    const user = userEvent.setup();
    const combatDetail: CampaignDetail = {
      ...detail,
      encounter: {
        id: 3, name: "Battle with Bandits", status: "active", round_number: 2,
        current_combatant_id: 10, protagonist_combatant_id: 10,
        recent_events: ["Mira Voss hits Bandit for 6 damage."],
        combatants: [
          { id: 10, character_id: 2, name: "Mira Voss", side: "party", lane: "front",
            initiative: 11, max_hp: 9, current_hp: 9, armor_class: 14, conditions: [],
            death_successes: 0, death_failures: 0, defeated: false },
          { id: 11, character_id: null, name: "Bandit", side: "enemy", lane: "front",
            initiative: 10, max_hp: 11, current_hp: 5, armor_class: 12, conditions: [],
            death_successes: 0, death_failures: 0, defeated: false }
        ],
        legal_actions: [
          { id: "weapon_attack", label: "Weapon Attack", requires_target: true,
            target_type: "enemy", description: "Attack", resource: null },
          { id: "move_lane", label: "Change Lane", requires_target: false,
            target_type: "none", description: "Move", resource: null }
        ]
      }
    };
    renderScreen(null, false, {}, null, combatDetail);
    const mira = screen.getByRole("button", { name: /Mira Voss, 9 of 9 hit points, acting now/ });
    const bandit = screen.getByRole("button", { name: /Bandit, 5 of 11 hit points/ });
    expect(mira).toHaveAttribute("aria-current", "true");
    expect(mira).toHaveAttribute("aria-pressed", "false");
    expect(bandit).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByLabelText("Describe your character action")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Change Lane destination" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Weapon Attack/ })).toHaveTextContent("Target: Bandit");
    expect(screen.getByRole("log", { name: "Recent combat results" })).toHaveTextContent(
      "Mira Voss hits Bandit for 6 damage."
    );

    await user.click(mira);
    expect(mira).toHaveAttribute("aria-pressed", "true");
    expect(bandit).toHaveAttribute("aria-pressed", "false");
  });

  test("disables duplicate auxiliary mutations and exposes pending labels", () => {
    renderScreen("history");
    expect(screen.getByRole("button", { name: "Loading..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Short Rest" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Mark complete" })).toBeDisabled();
  });

  test("has no automated accessibility violations in the stable state", async () => {
    const { container } = renderScreen(null);
    const result = await axe(container);
    expect(result.violations).toEqual([]);
  });

  test("shows a retryable opening error instead of generic narration", () => {
    renderScreen(null, false, {}, null, {
      ...detail,
      turns: detail.turns.filter((turn) => turn.speaker !== "DM"),
      opening: {
        status: "failed",
        idempotency_key: "opening-retry-1",
        error: "The narrator returned no playable text. qwen3.5:9b returned an empty response.",
        retryable: true
      }
    });
    expect(screen.getByRole("alert")).toHaveTextContent("The narrator could not finish");
    expect(screen.getByRole("button", { name: "Retry opening" })).toBeInTheDocument();
    expect(screen.queryByText(/situation shifts/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Describe your character action")).not.toBeInTheDocument();
  });

  test("presents a pending roll as a labelled modal with focused action", () => {
    renderScreen(null, false, {
      phase: "roll_required",
      pendingRoll: {
        id: 4,
        campaign_id: 1,
        action_text: "Inspect the seal",
        formula: "1d20+3",
        ability: "Intelligence",
        skill: "Arcana",
        dc: 13,
        reason: "Understand the old ward.",
        narration: "The runes flare softly.",
        status: "pending"
      }
    });
    expect(screen.getByRole("dialog", { name: "Intelligence (Arcana) check" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Roll Dice" })).toHaveFocus();
  });

  test("blocks rest and quest mutations while a roll is pending", () => {
    renderScreen(null, false, {
      phase: "roll_required",
      pendingRoll: {
        id: 4,
        campaign_id: 1,
        action_text: "Inspect the seal",
        formula: "1d20+3",
        ability: "Intelligence",
        skill: "Arcana",
        dc: 13,
        reason: "Understand the old ward.",
        narration: "",
        status: "pending"
      }
    }, "journal");
    expect(screen.getByRole("button", { name: "Short Rest" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Long Rest" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Mark complete" })).toBeDisabled();
  });
});
