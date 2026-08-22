import {
  Activity,
  AlertTriangle,
  BookOpen,
  BrainCircuit,
  Check,
  CheckCircle2,
  Dice5,
  LoaderCircle,
  MapPin,
  RefreshCw,
  ScrollText,
  Send,
  Shield,
  Sparkles,
  Square,
  Users,
  X
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type RefObject } from "react";
import { Dialog } from "../../components/Dialog";
import { Drawer } from "../../components/Drawer";
import type { CampaignDetail, Character, Encounter, PendingRoll, Turn } from "../../types";
import type { PlayState } from "./state";

export function PlayScreen({
  detail,
  state,
  onSetAction,
  onSubmitAction,
  onSubmitCombatAction,
  onResolvePendingRoll,
  onGenerateOpening,
  onRetry,
  onCancel,
  onClearError,
  onLoadOlderTurns,
  onRest,
  onCompleteQuest,
  onCreateCampaign,
  onOpenParty,
  onOpenJournal,
  utilityDrawer,
  onCloseUtilityDrawer,
  mutationPending,
  maxActionChars
}: {
  detail: CampaignDetail | null;
  state: PlayState;
  onSetAction: (value: string) => void;
  onSubmitAction: (content: string) => Promise<void>;
  onSubmitCombatAction: (encounterId: number, actionId: string, targetId: number | null,
    destinationLane?: "front" | "back" | null) => Promise<void>;
  onResolvePendingRoll: () => Promise<void>;
  onGenerateOpening: () => Promise<void>;
  onRetry: () => Promise<void>;
  onCancel: () => void;
  onClearError: () => void;
  onLoadOlderTurns: () => Promise<void>;
  onRest: (kind: "short" | "long", hitDice?: number) => Promise<void>;
  onCompleteQuest: (questId: number) => Promise<void>;
  onCreateCampaign: () => void;
  onOpenParty: () => void;
  onOpenJournal: () => void;
  utilityDrawer: "party" | "journal" | null;
  onCloseUtilityDrawer: () => void;
  mutationPending: string | null;
  maxActionChars: number;
}) {
  const actionInput = useRef<HTMLTextAreaElement>(null);

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSubmitAction(state.action);
  }

  if (!detail && state.phase === "loading") {
    return (
      <section id="main-play" className="play loading-screen" aria-live="polite" tabIndex={-1}>
        <LoaderCircle className="spin" size={34} aria-hidden="true" />
        <h1>Loading campaign</h1>
      </section>
    );
  }

  if (!detail) {
    return (
      <section id="main-play" className="play empty-screen" tabIndex={-1}>
        {state.error && (
          <OperationError
            message={state.error}
            canRetry={Boolean(state.retry)}
            onRetry={onRetry}
            onDismiss={onClearError}
          />
        )}
        <div className="empty">
          <BookOpen size={42} aria-hidden="true" />
          <h1>Create or select a campaign</h1>
          <p>Your local DM, heroes, lore and campaign state will appear here.</p>
          <button type="button" className="primary inline-primary" onClick={onCreateCampaign}>
            Begin an adventure
          </button>
        </div>
      </section>
    );
  }

  const activeEncounter = detail.encounter?.status === "active" ? detail.encounter : null;
  const openingGenerating = state.streaming && state.retry?.kind === "opening";
  const openingReady = detail.opening.status === "complete";
  return (
    <section id="main-play" className={activeEncounter ? "play combat-mode" : "play"} aria-busy={state.streaming} tabIndex={-1}>
      {state.error && (
        <OperationError
          message={state.error}
          canRetry={Boolean(state.retry)}
          onRetry={onRetry}
          onDismiss={onClearError}
        />
      )}

      <CampaignHeader detail={detail} onOpenParty={onOpenParty} onOpenJournal={onOpenJournal} />
      <GameHud
        detail={detail}
        phase={state.phase}
        rollResult={state.rollResult}
        statusMessage={state.statusMessage}
      />

      <div className="game-layout">
        <div className="scene-column">
          {state.pendingRoll && (
            <RollPrompt
              pendingRoll={state.pendingRoll}
              streaming={state.streaming}
              onResolve={onResolvePendingRoll}
            />
          )}
          {!openingReady ? (
            openingGenerating ? <OpeningGenerationState /> : (
              <OpeningCard opening={detail.opening} onGenerate={onGenerateOpening} />
            )
          ) : (
            <SceneViewport
              draft={state.draft}
              phase={state.phase}
              streaming={state.streaming}
              turns={detail.turns}
            />
          )}
          {openingReady && (activeEncounter ? (
            <EncounterPanel encounter={activeEncounter}
              disabled={state.streaming || Boolean(state.pendingRoll) || Boolean(state.retry)}
              onAct={onSubmitCombatAction} />
          ) : (
            <QuickActions choices={state.choices} selectedAction={state.action}
              disabled={state.streaming || Boolean(state.pendingRoll) || Boolean(state.retry)}
              onSelect={(choice) => { onSetAction(cleanChoiceText(choice)); actionInput.current?.focus(); }} />
          ))}
          {openingReady && !activeEncounter && detail.encounter && (
            <EncounterOutcome encounter={detail.encounter} />
          )}
          {openingReady && !activeEncounter && (
            <ActionComposer
              inputRef={actionInput}
              action={state.action}
              pendingRoll={state.pendingRoll}
              streaming={state.streaming}
              blocked={Boolean(state.retry)}
              maxActionChars={maxActionChars}
              onSetAction={onSetAction}
              onSubmit={submit}
              onCancel={onCancel}
            />
          )}
        </div>

      </div>
      {utilityDrawer === "party" && <Drawer title="Party" eyebrow="Character Sheets" onClose={onCloseUtilityDrawer}>
        <PartyPanel characters={detail.characters} />
      </Drawer>}
      {utilityDrawer === "journal" && <Drawer title="Adventure Journal" eyebrow="World & Timeline" onClose={onCloseUtilityDrawer}>
        <div className="journal-stack">
          <WorldPanel detail={detail} mutationPending={mutationPending}
            interactionBlocked={state.streaming || Boolean(state.pendingRoll) || Boolean(state.retry)}
            onRest={onRest} onCompleteQuest={onCompleteQuest} />
          <AdventureLog draft={state.draft} streaming={state.streaming} turns={detail.turns}
            hasMore={detail.turn_page.has_more} loading={mutationPending === "history"}
            onLoadOlder={onLoadOlderTurns} />
        </div>
      </Drawer>}
    </section>
  );
}

function OpeningGenerationState() {
  return (
    <section className="opening-generation" aria-label="Opening scene generation" aria-busy="true">
      <div className="opening-generation-heading">
        <LoaderCircle className="spin" size={24} aria-hidden="true" />
        <div>
          <span className="eyebrow">Creating your opening scene</span>
          <h2>The Dungeon Master is setting the stage…</h2>
        </div>
      </div>
      <p>Establishing your location, immediate tension, and first meaningful choices.</p>
      <div className="opening-generation-lines" aria-hidden="true">
        <span /><span /><span />
      </div>
    </section>
  );
}

function OpeningCard({ opening, onGenerate }: {
  opening: CampaignDetail["opening"];
  onGenerate: () => Promise<void>;
}) {
  return (
    <section className="opening-card" role={opening.status === "failed" ? "alert" : undefined}>
      <BrainCircuit size={28} aria-hidden="true" />
      <div>
        <span className="eyebrow">Opening scene</span>
        <h2>{opening.status === "failed" ? "The narrator could not finish" : "Your adventure is ready to begin"}</h2>
        <p>{opening.error ?? "Generate the first playable scene with your selected local narrator."}</p>
      </div>
      <button type="button" className="primary inline-primary" onClick={() => void onGenerate()}>
        <RefreshCw size={16} />{opening.status === "failed" ? "Retry opening" : "Generate opening"}
      </button>
    </section>
  );
}

function OperationError({
  message,
  canRetry,
  onRetry,
  onDismiss
}: {
  message: string;
  canRetry: boolean;
  onRetry: () => Promise<void>;
  onDismiss: () => void;
}) {
  return (
    <div className="error operation-error" role="alert">
      <AlertTriangle size={18} aria-hidden="true" />
      <span>{message}</span>
      {canRetry && (
        <button type="button" className="secondary" onClick={() => void onRetry()}>
          <RefreshCw size={15} aria-hidden="true" /> Retry safely
        </button>
      )}
      {!canRetry && (
        <button type="button" className="icon-button" onClick={onDismiss} aria-label="Dismiss error">
          <X size={15} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

function CampaignHeader({ detail, onOpenParty, onOpenJournal }: {
  detail: CampaignDetail;
  onOpenParty: () => void;
  onOpenJournal: () => void;
}) {
  return (
    <header className="campaign-header">
      <div>
        <span className="eyebrow">Active Campaign</span>
        <h1>{detail.campaign.title}</h1>
        <p>{detail.campaign.setting}</p>
      </div>
      <div className="campaign-header-actions">
        <span>{detail.campaign.tone}</span>
        <button type="button" className="secondary" onClick={onOpenParty}><Users size={16} /> Party</button>
        <button type="button" className="secondary" onClick={onOpenJournal}><BookOpen size={16} /> Journal</button>
      </div>
    </header>
  );
}

function GameHud({
  detail,
  phase,
  rollResult,
  statusMessage,
}: {
  detail: CampaignDetail;
  phase: PlayState["phase"];
  rollResult: string;
  statusMessage: string;
}) {
  const busy = ["checking", "generating", "rolling", "loading"].includes(phase);
  const protagonist = detail.characters.find((character) => character.role === "protagonist");
  return (
    <section className="context-strip" aria-label="Campaign status">
      <div className={`context-status status-pill ${phase}`} role="status" aria-live="polite" aria-atomic="true">
        {busy ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />}
        <strong>{statusMessage}</strong>
      </div>
      <dl className="context-facts">
        <div>
          <Shield size={16} aria-hidden="true" />
          <dt>Hero</dt>
          <dd>{protagonist ? `${protagonist.name} · ${protagonist.current_hp}/${protagonist.max_hp} HP · AC ${protagonist.armor_class}` : "No protagonist"}</dd>
        </div>
        <div>
          <MapPin size={16} aria-hidden="true" />
          <dt>Location</dt>
          <dd>{detail.world_state.current_location}</dd>
        </div>
        <div className="context-objective">
          <ScrollText size={16} aria-hidden="true" />
          <dt>Objective</dt>
          <dd>{detail.world_state.active_objective}</dd>
        </div>
        <div>
          <Dice5 size={16} aria-hidden="true" />
          <dt>Last roll</dt>
          <dd>{rollResult}</dd>
        </div>
      </dl>
    </section>
  );
}

function SceneViewport({
  draft,
  phase,
  streaming,
  turns
}: {
  draft: string;
  phase: PlayState["phase"];
  streaming: boolean;
  turns: Turn[];
}) {
  const featured = streaming && draft ? { speaker: "DM", content: draft } : latestSceneTurn(turns);
  const contentRef = useRef<HTMLDivElement>(null);
  const featuredTurnId = [...turns].reverse().find((turn) => turn.speaker === "DM")?.id ?? 0;
  useEffect(() => {
    if (contentRef.current) contentRef.current.scrollTop = 0;
  }, [featuredTurnId, streaming]);
  return (
    <section className="scene-viewport">
      <div className="scene-header">
        <div>
          <span className="eyebrow">Current Scene</span>
          <h2>{featured?.speaker ?? "DM"}</h2>
        </div>
        {phase !== "ready" && phase !== "idle" && <div className={`model-indicator ${phase}`}>
          {["generating", "checking", "rolling"].includes(phase) ? (
            <LoaderCircle className="spin" size={16} aria-hidden="true" />
          ) : (
            <BrainCircuit size={16} aria-hidden="true" />
          )}
          <span>{phaseLabel(phase)}</span>
        </div>}
      </div>
      <div className="scene-content" ref={contentRef}>
        {featured ? (
          formatSceneText(featured.content).map((paragraph, index) => (
            <p key={`${paragraph.slice(0, 24)}-${index}`}>{paragraph}</p>
          ))
        ) : (
          <p>Describe what your character does to begin.</p>
        )}
      </div>
    </section>
  );
}

function RollPrompt({
  pendingRoll,
  streaming,
  onResolve
}: {
  pendingRoll: PendingRoll;
  streaming: boolean;
  onResolve: () => Promise<void>;
}) {
  const resuming = pendingRoll.status === "resolved";
  return (
    <Dialog titleId="dice-check-title" onClose={() => undefined} className="dice-dialog">
      <div className="roll-prompt">
        <div className="roll-emblem">
          <Dice5 size={30} aria-hidden="true" />
        </div>
        <div className="roll-copy">
          <span className="eyebrow">{resuming ? "Saved Dice Check" : "Dice Check"}</span>
          <strong id="dice-check-title">
            {pendingRoll.ability}
            {pendingRoll.skill ? ` (${pendingRoll.skill})` : ""} check
          </strong>
          <p>{pendingRoll.reason}</p>
          {pendingRoll.narration && <small>{pendingRoll.narration}</small>}
        </div>
        <div className="roll-target" aria-label="Roll target">
          <span>{pendingRoll.formula}</span>
          <span>DC {pendingRoll.dc}</span>
        </div>
        <button type="button" onClick={() => void onResolve()} disabled={streaming}>
          {streaming ? <LoaderCircle className="spin" size={20} /> : <Dice5 size={20} />}
          {streaming ? "Resolving..." : resuming ? "Resume Resolution" : "Roll Dice"}
        </button>
      </div>
    </Dialog>
  );
}

function PartyPanel({ characters }: { characters: Character[] }) {
  return (
    <section className="rail-panel">
      <div className="rail-title">
        <Shield size={16} aria-hidden="true" />
        <span>Party</span>
      </div>
      <div className="party">
        {characters.length === 0 && <p className="hint">No heroes joined this campaign.</p>}
        {characters.map((character) => (
          <article key={character.id}>
            <Shield size={18} aria-hidden="true" />
            <h2>{character.name}</h2>
            <p>
              Level {character.level} {character.ancestry} {character.character_class} · {character.role}
            </p>
            <small>HP {character.current_hp}/{character.max_hp} · AC {character.armor_class}</small>
            <div className="hp-track" aria-label={`${character.current_hp} of ${character.max_hp} hit points`}>
              <span style={{ width: `${Math.max(0, Math.min(100, character.current_hp / character.max_hp * 100))}%` }} />
            </div>
            <dl className="ability-grid">
              {[["STR", character.strength], ["DEX", character.dexterity], ["CON", character.constitution],
                ["INT", character.intelligence], ["WIS", character.wisdom], ["CHA", character.charisma]].map(([label, value]) =>
                <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
            </dl>
            <details><summary>Inventory & resources</summary>
              <p>{readStringList(character.inventory_json).join(", ") || "No inventory listed."}</p>
              <p>{formatResources(character.resources_json)}</p>
            </details>
          </article>
        ))}
      </div>
    </section>
  );
}

function WorldPanel({ detail, mutationPending, interactionBlocked, onRest, onCompleteQuest }: {
  detail: CampaignDetail;
  mutationPending: string | null;
  interactionBlocked: boolean;
  onRest: (kind: "short" | "long", hitDice?: number) => Promise<void>;
  onCompleteQuest: (questId: number) => Promise<void>;
}) {
  const activeQuest = detail.quests.find((quest) => quest.status === "active");
  const [confirmation, setConfirmation] = useState<{ kind: "short" | "long" | "quest"; questId?: number } | null>(null);
  const [hitDice, setHitDice] = useState(1);
  return (
    <section className="rail-panel">
      <div className="rail-title">
        <MapPin size={16} aria-hidden="true" />
        <span>World</span>
      </div>
      <dl className="world-grid">
        <dt>Location</dt>
        <dd>{detail.world_state.current_location}</dd>
        <dt>Objective</dt>
        <dd>{detail.world_state.active_objective}</dd>
        <dt>Scene</dt>
        <dd>{detail.world_state.scene_summary || detail.campaign.setting}</dd>
        {activeQuest && (
          <>
            <dt>Quest</dt>
            <dd>
              <strong>{activeQuest.title}</strong>
              <br />
              {activeQuest.objective}
              <button type="button" className="quest-complete"
                disabled={mutationPending !== null || interactionBlocked}
                onClick={() => setConfirmation({ kind: "quest", questId: activeQuest.id })}>
                {mutationPending === `quest-${activeQuest.id}` ? "Completing..." : "Mark complete"}
              </button>
            </dd>
          </>
        )}
      </dl>
      {detail.encounter?.status !== "active" && (
        <div className="rest-actions">
          <button type="button" disabled={mutationPending !== null || interactionBlocked} onClick={() => setConfirmation({ kind: "short" })}>
            {mutationPending === "rest-short" ? "Resting..." : "Short Rest"}
          </button>
          <button type="button" disabled={mutationPending !== null || interactionBlocked} onClick={() => setConfirmation({ kind: "long" })}>
            {mutationPending === "rest-long" ? "Resting..." : "Long Rest"}
          </button>
        </div>
      )}
      {confirmation && <Dialog titleId="confirm-world-action" onClose={() => setConfirmation(null)} className="confirm-dialog">
        <div className="modal-head"><div><span className="eyebrow">Confirm action</span>
          <h2 id="confirm-world-action">{confirmation.kind === "quest" ? "Complete this quest?" : `Take a ${confirmation.kind} rest?`}</h2></div>
          <button type="button" className="icon-button" onClick={() => setConfirmation(null)} aria-label="Cancel action"><X size={17} /></button></div>
        <p className="dialog-copy">{confirmation.kind === "quest"
          ? "This awards the quest milestone and cannot be undone."
          : confirmation.kind === "short" ? "The party will spend one hit die where available and recover class resources." : "The party will recover fully and advance the world clock."}</p>
        {confirmation.kind === "short" && <label className="rest-dice-field"><span>Hit dice to spend per eligible hero</span>
          <select value={hitDice} onChange={(event) => setHitDice(Number(event.target.value))}>
            {[0, 1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}</option>)}
          </select></label>}
        <div className="modal-actions"><button type="button" className="secondary" onClick={() => setConfirmation(null)}>Cancel</button>
          <button type="button" className="primary inline-primary" onClick={() => {
            const current = confirmation; setConfirmation(null);
            if (current.kind === "quest" && current.questId) void onCompleteQuest(current.questId);
            else if (current.kind !== "quest") void onRest(current.kind, current.kind === "short" ? hitDice : 0);
          }}>Confirm</button></div>
      </Dialog>}
    </section>
  );
}

function QuickActions({
  choices,
  selectedAction,
  disabled,
  onSelect
}: {
  choices: string[];
  selectedAction: string;
  disabled: boolean;
  onSelect: (value: string) => void;
}) {
  const fallback = [
    "Ask around for rumours about the sealed ruins.",
    "Look for a safe tavern and listen for trouble.",
    "Inspect the nearest landmark for unusual signs."
  ];
  const cleaned = choices.map(cleanChoiceText).filter(Boolean);
  const actions = cleaned.length ? cleaned : fallback;
  return (
    <section className="quick-actions" aria-label="Suggested actions">
      <div className="quick-title">
        <Sparkles size={16} aria-hidden="true" />
        <span>Suggested actions</span>
        <small>Choose one to edit</small>
      </div>
      {actions.map((text) => (
        <button type="button" key={text} disabled={disabled} onClick={() => onSelect(text)}
          className={cleanChoiceText(selectedAction) === text ? "selected" : ""}
          aria-pressed={cleanChoiceText(selectedAction) === text}>
          {cleanChoiceText(selectedAction) === text ? <Check size={15} aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
          {text}
        </button>
      ))}
    </section>
  );
}

function EncounterPanel({ encounter, disabled, onAct }: {
  encounter: Encounter;
  disabled: boolean;
  onAct: (encounterId: number, actionId: string, targetId: number | null,
    destinationLane?: "front" | "back" | null) => Promise<void>;
}) {
  const [targetId, setTargetId] = useState<number | null>(
    encounter.combatants.find((combatant) => combatant.side === "enemy" && !combatant.defeated)?.id ?? null
  );
  const [allyTargetId, setAllyTargetId] = useState<number | null>(encounter.protagonist_combatant_id);
  const [selectedSide, setSelectedSide] = useState<"party" | "enemy">("enemy");
  const [destinationLane, setDestinationLane] = useState<"front" | "back">("front");
  const enemies = encounter.combatants.filter((combatant) => combatant.side === "enemy" && !combatant.defeated);
  const protagonist = encounter.combatants.find((combatant) => combatant.id === encounter.protagonist_combatant_id);
  const allies = encounter.combatants.filter((combatant) => combatant.side === "party" && !combatant.defeated);
  useEffect(() => {
    setTargetId((current) => enemies.some((combatant) => combatant.id === current) ? current : enemies[0]?.id ?? null);
    setAllyTargetId((current) => allies.some((combatant) => combatant.id === current) ? current : allies[0]?.id ?? null);
    setDestinationLane(protagonist?.lane === "front" ? "back" : "front");
  }, [encounter.id, encounter.round_number, encounter.combatants, protagonist?.lane]);
  const selectedTargetId = selectedSide === "enemy" ? targetId : allyTargetId;
  const movementActions = new Set(["move_lane", "cunning_action", "misty_step"]);
  return (
    <section className="encounter-panel" aria-label="Active encounter">
      <div className="quick-title"><Shield size={16} /><span>{encounter.name} · Round {encounter.round_number}</span></div>
      <div className="initiative-strip" aria-label="Initiative order">
        {encounter.combatants.map((combatant) => <span className={combatant.id === encounter.current_combatant_id ? "current" : ""}
          key={combatant.id}>{combatant.initiative} · {combatant.name}</span>)}
      </div>
      <div className="battlefield">
        {(["back", "front"] as const).map((lane) => {
          const laneCombatants = encounter.combatants.filter((combatant) => combatant.lane === lane);
          return <div className={`combat-lane${laneCombatants.length ? "" : " empty"}`} key={lane}>
            <strong>{lane} lane</strong>
            <div className="combatants">{laneCombatants.map((combatant) => {
              const selected = selectedTargetId === combatant.id;
              const acting = encounter.current_combatant_id === combatant.id;
              return <button type="button" key={combatant.id} disabled={combatant.defeated}
                className={`combatant${selected ? " selected" : ""}${acting ? " acting" : ""}`}
                aria-label={`${combatant.name}, ${combatant.current_hp} of ${combatant.max_hp} hit points${acting ? ", acting now" : ""}`}
                aria-pressed={selected}
                aria-current={acting ? "true" : undefined}
                onClick={() => {
                  setSelectedSide(combatant.side);
                  if (combatant.side === "enemy") setTargetId(combatant.id);
                  else setAllyTargetId(combatant.id);
                }}>
                <span className="combatant-heading"><strong>{combatant.name}</strong>
                  {acting && <small className="acting-label">Acting</small>}
                  {selected && <small className="target-label">Target</small>}
                </span>
                <span>{combatant.lane} lane</span>
                <small>HP {combatant.current_hp}/{combatant.max_hp} · AC {combatant.armor_class}</small>
                {combatant.current_hp === 0 && combatant.side === "party" &&
                  <small>Death saves {combatant.death_successes}✓ / {combatant.death_failures}✕</small>}
                {combatant.conditions.length > 0 && <small>{combatant.conditions.join(", ")}</small>}
              </button>;
            })}</div>
          </div>;
        })}
      </div>
      {encounter.recent_events.length > 0 && (
        <div className="combat-round-log" role="log" aria-label="Recent combat results">
          <div><ScrollText size={14} aria-hidden="true" /><strong>Recent results</strong></div>
          <ul>{encounter.recent_events.map((event, index) => (
            <li key={`${index}-${event}`}>{event}</li>
          ))}</ul>
        </div>
      )}
      <div className="combat-actions">
        {encounter.legal_actions.map((action) => {
          const isMovement = movementActions.has(action.id);
          const actionTarget = action.target_type === "ally" ? allyTargetId : action.requires_target ? targetId : null;
          const targetName = encounter.combatants.find((combatant) => combatant.id === actionTarget)?.name;
          return <div className={`combat-action-control${isMovement ? " movement" : ""}`} key={action.id}>
            {isMovement && <label className="lane-choice"><span>Destination</span><select value={destinationLane}
              disabled={disabled} onChange={(event) => setDestinationLane(event.target.value as "front" | "back")}
              aria-label={`${action.label} destination`}>
              <option value="front">Front lane</option><option value="back">Back lane</option>
            </select></label>}
            <button type="button" disabled={disabled || (action.requires_target &&
            (action.target_type === "ally" ? !allyTargetId : !targetId))}
              title={`${action.description}${action.resource ? ` · ${action.resource}` : ""}`}
              onClick={() => void onAct(encounter.id, action.id,
                action.target_type === "ally" ? allyTargetId : action.requires_target ? targetId : null,
                isMovement
                ? destinationLane : null)}>
              <strong>{action.label}</strong>
              <small>{action.description}{action.resource ? ` · ${action.resource}` : ""}</small>
              {targetName && <small className="action-target">Target: {targetName}</small>}
            </button>
          </div>;
        })}
        {!encounter.legal_actions.length && enemies.length > 0 && <span className="hint">The party is resolving its turn.</span>}
      </div>
    </section>
  );
}

function EncounterOutcome({ encounter }: { encounter: Encounter }) {
  const labels: Record<string, string> = {
    victory: "Victory — the way forward is open.",
    defeat: "Defeat — the party survived with a lasting setback.",
    fled: "Escaped — the party broke away from danger."
  };
  return <section className={`encounter-outcome ${encounter.status}`}>
    <Shield size={17} /><strong>{labels[encounter.status] ?? "The encounter has ended."}</strong>
  </section>;
}

function AdventureLog({
  draft,
  streaming,
  turns,
  hasMore,
  loading,
  onLoadOlder
}: {
  draft: string;
  streaming: boolean;
  turns: Turn[];
  hasMore: boolean;
  loading: boolean;
  onLoadOlder: () => Promise<void>;
}) {
  const [expandedId, setExpandedId] = useState<number | "draft" | null>(null);
  return (
    <section className="rail-panel timeline-panel">
      <div className="rail-title">
        <ScrollText size={16} aria-hidden="true" />
        <span>Timeline</span>
      </div>
      <div className="log">
        {hasMore && (
          <button type="button" className="load-older" disabled={loading}
            onClick={() => void onLoadOlder()}>
            {loading ? "Loading..." : "Load older turns"}
          </button>
        )}
        {turns.slice(-12).map((turn) => (
          <button
            type="button"
            className={
              expandedId === turn.id
                ? `timeline-entry ${turn.speaker.toLowerCase()} expanded`
                : `timeline-entry ${turn.speaker.toLowerCase()}`
            }
            key={turn.id}
            onClick={() => setExpandedId(expandedId === turn.id ? null : turn.id)}
            aria-expanded={expandedId === turn.id}
          >
            <span>
              {turn.speaker === "DM" && <CheckCircle2 size={15} aria-hidden="true" />}
              {turn.speaker}
            </span>
            <p>{expandedId === turn.id ? turn.content : summariseTurn(turn.content)}</p>
          </button>
        ))}
        {streaming && draft && (
          <button
            type="button"
            className={expandedId === "draft" ? "timeline-entry dm expanded" : "timeline-entry dm"}
            onClick={() => setExpandedId(expandedId === "draft" ? null : "draft")}
            aria-expanded={expandedId === "draft"}
          >
            <span>DM · generating</span>
            <p>{draft}</p>
          </button>
        )}
      </div>
    </section>
  );
}

function ActionComposer({
  inputRef,
  action,
  pendingRoll,
  streaming,
  blocked,
  maxActionChars,
  onSetAction,
  onSubmit,
  onCancel
}: {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  action: string;
  pendingRoll: PendingRoll | null;
  streaming: boolean;
  blocked: boolean;
  maxActionChars: number;
  onSetAction: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onCancel: () => void;
}) {
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && action.trim() &&
      !streaming && !pendingRoll && !blocked) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }
  const remaining = maxActionChars - action.length;
  return (
    <form className="action-bar" onSubmit={onSubmit}>
      <label className="sr-only" htmlFor="player-action">
        Describe your character action
      </label>
      <textarea
        ref={inputRef}
        id="player-action"
        value={action}
        maxLength={maxActionChars}
        onChange={(event) => onSetAction(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={
          pendingRoll
            ? "Resolve the dice check to continue..."
            : blocked
              ? "Retry the interrupted operation to continue..."
              : "Describe what your character does..."
        }
        disabled={streaming || Boolean(pendingRoll) || blocked}
        rows={2}
      />
      <div className={remaining < 200 ? "composer-meta near-limit" : "composer-meta"}>
        <span>Draft saved locally · {navigator.platform.includes("Mac") ? "⌘" : "Ctrl"}+Enter to send</span>
        <span>{action.length.toLocaleString()} / {maxActionChars.toLocaleString()}</span>
      </div>
      {streaming ? (
        <button type="button" className="cancel-stream" onClick={onCancel} aria-label="Cancel generation">
          <Square size={17} aria-hidden="true" />
        </button>
      ) : (
        <button
          type="submit"
          disabled={Boolean(pendingRoll) || blocked || !action.trim()}
          aria-label="Submit action"
        >
          <Send size={18} aria-hidden="true" />
          <span>Send</span>
        </button>
      )}
    </form>
  );
}

function latestSceneTurn(turns: Turn[]): Pick<Turn, "speaker" | "content"> | null {
  return (
    [...turns].reverse().find((turn) => turn.speaker === "DM") ??
    [...turns].reverse().find((turn) => turn.speaker !== "System") ??
    null
  );
}

function formatSceneText(content: string): string[] {
  const withoutChoices = content.split(
    /\n\s*(?:(?:you have )?the following options|choices|options|what do you do)[?:]?\s*\n/i
  )[0] ?? content;
  const clean =
    withoutChoices
      .replace(/^\s*(?:here(?:'s| is)|this is|a possible|possible)\b[^:\n]*:\s*/i, "")
      .split(/\n?\s*(?:this message establishes|the message establishes|it establishes)\b/i)[0] ?? "";
  return clean
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

function cleanChoiceText(value: string): string {
  return value
    .replace(/^\s*(?:\d+[).:]|-|\*)\s+/, "")
    .replace(/\*\*/g, "")
    .trim();
}

function summariseTurn(content: string): string {
  const clean = content.replace(/\s+/g, " ").trim();
  return clean.length > 150 ? `${clean.slice(0, 147)}...` : clean;
}

function phaseLabel(phase: PlayState["phase"]): string {
  switch (phase) {
    case "checking":
      return "Rules referee";
    case "generating":
      return "DM generating";
    case "rolling":
      return "Resolving roll";
    case "roll_required":
      return "Awaiting roll";
    case "error":
      return "Recovery available";
    case "loading":
      return "Loading";
    default:
      return "Ready";
  }
}

function readStringList(value: string): string[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch { return []; }
}

function formatResources(value: string): string {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "No tracked resources.";
    const entries = Object.entries(parsed).map(([key, item]) => `${key.replaceAll("_", " ")}: ${String(item)}`);
    return entries.join(" · ") || "No tracked resources.";
  } catch { return "No tracked resources."; }
}
