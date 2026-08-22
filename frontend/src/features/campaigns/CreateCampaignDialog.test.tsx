// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test, vi } from "vitest";

import { CreateCampaignDialog, type CampaignDraft } from "./CreateCampaignDialog";

function Harness() {
  const [draft, setDraft] = useState<CampaignDraft>({
    title: "The Shattered Gate",
    setting: "A frontier city above sealed ruins.",
    tone: "tense heroic fantasy"
  });
  return <CreateCampaignDialog
    draft={draft}
    heroes={[]}
    lore={[]}
    protagonistId={null}
    companionIds={[]}
    selectedLoreIds={[]}
    creating={false}
    onDraftChange={setDraft}
    onSelectProtagonist={vi.fn()}
    onToggleCompanion={vi.fn()}
    onToggleLore={vi.fn()}
    onCancel={() => undefined}
    onConfirm={vi.fn()}
  />;
}

test("typing controlled campaign fields does not return focus to the close button", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const title = screen.getByRole("textbox", { name: "Campaign title" });
  const tone = screen.getByRole("textbox", { name: "Tone" });
  const setting = screen.getByRole("textbox", { name: "Setting brief" });

  expect(title).toHaveFocus();
  await user.clear(title);
  await user.type(title, "Moonfall Keep");
  expect(title).toHaveFocus();
  expect(title).toHaveValue("Moonfall Keep");

  await user.clear(tone);
  await user.type(tone, "eerie mystery");
  expect(tone).toHaveFocus();
  await user.clear(setting);
  await user.type(setting, "A flooded citadel wakes beneath the marsh.");
  expect(setting).toHaveFocus();
});
