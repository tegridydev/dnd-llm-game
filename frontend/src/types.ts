export type Campaign = {
  id: number;
  title: string;
  setting: string;
  tone: string;
  milestone_points: number;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  current_location?: string;
  last_activity_at?: string;
};

export type Character = {
  id: number;
  campaign_id: number;
  name: string;
  ancestry: string;
  character_class: string;
  backstory: string;
  inventory_json: string;
  role: "protagonist" | "companion";
  level: number;
  strength: number;
  dexterity: number;
  constitution: number;
  intelligence: number;
  wisdom: number;
  charisma: number;
  max_hp: number;
  current_hp: number;
  armor_class: number;
  speed: number;
  skills_json: string;
  saves_json: string;
  spells_json: string;
  resources_json: string;
  conditions_json: string;
};

export type Hero = {
  id: number;
  name: string;
  ancestry: string;
  character_class: string;
  backstory: string;
  inventory_json: string;
  level: number;
  strength: number;
  dexterity: number;
  constitution: number;
  intelligence: number;
  wisdom: number;
  charisma: number;
  max_hp: number;
  armor_class: number;
  speed: number;
  skills_json: string;
  saves_json: string;
  spells_json: string;
  resources_json: string;
  created_at: string;
  updated_at: string;
};

export type Turn = {
  id: number;
  campaign_id: number;
  speaker: "System" | "Player" | "DM" | "Roll";
  content: string;
  created_at: string;
};

export type WorldState = {
  id: number;
  campaign_id: number;
  current_location: string;
  active_objective: string;
  scene_summary: string;
  choices_json: string;
  facts_json: string;
  npcs_json: string;
  updated_at: string;
};

export type PendingRoll = {
  id: number;
  campaign_id: number;
  action_text: string;
  formula: string;
  ability: string;
  skill: string | null;
  dc: number;
  reason: string;
  narration: string;
  status: "pending" | "resolving" | "resolved" | "cancelled" | "failed";
};

export type TurnPage = {
  has_more: boolean;
  next_cursor: string | null;
  limit: number;
};

export type CampaignDetail = {
  campaign: Campaign;
  characters: Character[];
  turns: Turn[];
  world_state: WorldState;
  choices: string[];
  pending_roll: PendingRoll | null;
  recoverable_operation: RecoverableOperation | null;
  opening: CampaignOpening;
  encounter: Encounter | null;
  quests: Quest[];
  turn_page: TurnPage;
  last_roll: RollResult | null;
};

export type CampaignOpening = {
  status: "needed" | "processing" | "failed" | "complete";
  idempotency_key: string | null;
  error: string | null;
  retryable: boolean;
};

export type LoreDocument = {
  id: number;
  filename: string;
  status: "queued" | "indexing" | "ready" | "error" | "deleting";
  chunks: number;
  size_bytes: number;
  page_count: number;
  attempts: number;
  embed_model: string;
  created_at: string;
  updated_at: string;
  error: string | null;
};

export type HealthComponent = {
  status: "ok" | "degraded" | "error";
  detail: string | null;
};

export type WorkerStatus = HealthComponent & {
  running: boolean;
  active_document_id: number | null;
  queued_count: number;
};

export type Health = {
  status: "ok" | "degraded" | "error";
  database: HealthComponent;
  filesystem: HealthComponent;
  worker: WorkerStatus;
  ollama: HealthComponent;
  model_runtime: {
    narrator: ModelRuntimeComponent;
    utility: ModelRuntimeComponent;
    embeddings: ModelRuntimeComponent;
  };
  chat_model: string;
  utility_model: string;
  embed_model: string;
  request_max_chars: number;
};

export type ModelRuntimeComponent = {
  status: "unverified" | "healthy" | "failed";
  detail: string | null;
  updated_at: string | null;
};

export type ModelOption = { name: string; capabilities: string[] };
export type ModelSettings = {
  chat_model: string;
  utility_model: string;
  embed_model: string;
  narration_style: "focused" | "balanced" | "cinematic";
  models: ModelOption[];
  model_runtime: Health["model_runtime"];
  lore_document_count: number;
  reindex_queued: number;
};

export type HeroPayload = {
  name: string;
  ancestry: string;
  character_class: string;
  backstory: string;
  inventory: string[];
  strength?: number;
  dexterity?: number;
  constitution?: number;
  intelligence?: number;
  wisdom?: number;
  charisma?: number;
};

export type CampaignPayload = {
  title: string;
  setting: string;
  tone: string;
  protagonist_id: number;
  companion_ids: number[];
  lore_document_ids: number[];
};

export type ChoiceUpdate = {
  choices: string[];
  location: string;
  objective: string;
  summary: string;
};

export type RollResult = {
  id: number;
  pending_roll_id: number | null;
  formula: string;
  rolls: number[];
  modifier: number;
  total: number;
  dc: number | null;
  outcome: string;
  reason: string;
};

export type RecoverableOperation = {
  kind: "action" | "roll_resolution" | "combat_action";
  idempotency_key: string;
  action_text: string;
  error: string;
  pending_roll: PendingRoll | null;
  dice_roll: RollResult | null;
};

export type StreamPhase =
  | "checking_action"
  | "dm_streaming"
  | "utility_analyzing";

export type StreamDone = {
  status: "complete" | "roll_required" | "failed";
  authoritative_refresh: boolean;
};

export type StreamError = {
  code: string;
  message: string;
  retryable: boolean;
  detail?: string | null;
  dice_result_saved?: boolean;
};

export type StreamEventData = {
  stream_started: {
    operation: "action" | "roll_resolution" | "combat_action" | "campaign_opening";
    operation_id: number;
    idempotency_key: string;
    replayed: boolean;
  };
  phase: { status: StreamPhase };
  narration: { content: string };
  narration_delta: { content: string };
  narration_replace: { content: string };
  roll_required: PendingRoll;
  roll_result: RollResult;
  choices_updated: ChoiceUpdate;
  encounter_started: Encounter;
  encounter_updated: Encounter;
  combat_log: { events: string[] };
  party_recovered: { message: string };
  replay: { status: "complete" };
  error: StreamError;
  done: StreamDone;
};

export type Combatant = {
  id: number;
  character_id: number | null;
  name: string;
  side: "party" | "enemy";
  lane: "front" | "back";
  initiative: number;
  max_hp: number;
  current_hp: number;
  armor_class: number;
  conditions: string[];
  death_successes: number;
  death_failures: number;
  defeated: boolean;
};

export type CombatActionOption = {
  id: string;
  label: string;
  requires_target: boolean;
  target_type: "enemy" | "ally" | "self" | "party" | "none";
  description: string;
  resource: string | null;
};

export type Encounter = {
  id: number;
  name: string;
  status: "active" | "victory" | "defeat" | "fled";
  round_number: number;
  current_combatant_id: number | null;
  protagonist_combatant_id: number | null;
  legal_actions: CombatActionOption[];
  recent_events: string[];
  combatants: Combatant[];
};

export type Quest = {
  id: number;
  title: string;
  objective: string;
  status: "active" | "complete" | "failed";
  milestone_reward: number;
};

export type StreamEventName = keyof StreamEventData;

export type StreamEnvelope<T> = {
  protocol_version: 1;
  request_id: string;
  campaign_id: number;
  sequence: number;
  timestamp: string;
  data: T;
};

export type TypedStreamEvent<K extends StreamEventName = StreamEventName> =
  K extends StreamEventName
    ? { name: K; envelope: StreamEnvelope<StreamEventData[K]> }
    : never;
