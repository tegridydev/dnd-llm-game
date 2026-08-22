import {
  AlertTriangle, Archive, BookOpen, BrainCircuit, Check, Download, Menu,
  RefreshCw, ScrollText, Settings, Shield, UserRound, Users, X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Drawer } from "./components/Drawer";
import { CampaignSidebar } from "./features/campaigns/CampaignSidebar";
import { CreateCampaignDialog, type CampaignDraft } from "./features/campaigns/CreateCampaignDialog";
import { HeroManager } from "./features/heroes/HeroManager";
import { LorePanel } from "./features/lore/LorePanel";
import { PlayScreen } from "./features/play/PlayScreen";
import { useCampaignPlay } from "./hooks/useCampaignPlay";
import { deleteJson, downloadJson, errorMessage, getJson, getStatusJson, patchJson, postJson, uploadFile } from "./lib/api";
import type { Campaign, CampaignPayload, Health, Hero, HeroPayload, LoreDocument, ModelSettings } from "./types";
import "./styles/app.css";

const DEFAULT_CAMPAIGN: CampaignDraft = {
  title: "The Shattered Gate",
  setting: "A frontier city built above sealed ruins where old oaths are failing.",
  tone: "tense heroic fantasy"
};

type DrawerName = "campaigns" | "heroes" | "lore" | "runtime" | "settings" | "party" | "journal";

function savedCampaignId(): number | null {
  const value = window.localStorage.getItem("dndllm26.activeCampaign");
  const parsed = value ? Number(value) : NaN;
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export default function App() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [activeId, setActiveIdState] = useState<number | null>(savedCampaignId);
  const [health, setHealth] = useState<Health | null>(null);
  const [heroes, setHeroes] = useState<Hero[]>([]);
  const [lore, setLore] = useState<LoreDocument[]>([]);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [drawer, setDrawer] = useState<DrawerName | null>(null);
  const [railCollapsed, setRailCollapsed] = useState(
    () => window.localStorage.getItem("dndllm26.campaignRailCollapsed") === "true"
  );
  const [showCampaignDialog, setShowCampaignDialog] = useState(false);
  const [creatingCampaign, setCreatingCampaign] = useState(false);
  const [campaignDraft, setCampaignDraft] = useState<CampaignDraft>(DEFAULT_CAMPAIGN);
  const [protagonistId, setProtagonistId] = useState<number | null>(null);
  const [companionIds, setCompanionIds] = useState<number[]>([]);
  const [selectedLoreIds, setSelectedLoreIds] = useState<number[]>([]);
  const [pendingOpeningId, setPendingOpeningId] = useState<number | null>(null);

  const activeCampaign = campaigns.find((campaign) => campaign.id === activeId) ?? null;

  const setActiveId = useCallback((id: number | null) => {
    setActiveIdState(id);
    if (id) window.localStorage.setItem("dndllm26.activeCampaign", String(id));
    else window.localStorage.removeItem("dndllm26.activeCampaign");
  }, []);

  const refreshCampaigns = useCallback(async () => {
    const rows = await getJson<Campaign[]>("/campaigns?include_archived=true");
    setCampaigns(rows);
    setActiveIdState((current) => {
      if (current && rows.some((campaign) => campaign.id === current && !campaign.archived_at)) return current;
      const next = rows.find((campaign) => !campaign.archived_at)?.id ?? null;
      if (next) window.localStorage.setItem("dndllm26.activeCampaign", String(next));
      else window.localStorage.removeItem("dndllm26.activeCampaign");
      return next;
    });
  }, []);

  const refreshMetadata = useCallback(async () => {
    const results = await Promise.allSettled([
      getStatusJson<Health>("/health/ready", [200, 503]), getJson<Hero[]>("/heroes"),
      getJson<LoreDocument[]>("/lore")
    ] as const);
    const [status, heroRows, loreRows] = results;
    if (status.status === "fulfilled") setHealth(status.value);
    if (heroRows.status === "fulfilled") {
      setHeroes(heroRows.value);
      setProtagonistId((current) => current && heroRows.value.some((hero) => hero.id === current) ? current : null);
      setCompanionIds((current) => current.filter((id) => heroRows.value.some((hero) => hero.id === id)));
    }
    if (loreRows.status === "fulfilled") {
      setLore(loreRows.value);
      setSelectedLoreIds((current) => current.filter((id) =>
        loreRows.value.some((document) => document.id === id && document.status === "ready")
      ));
    }
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }, []);

  const refreshAll = useCallback(async () => {
    const results = await Promise.allSettled([refreshCampaigns(), refreshMetadata()]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }, [refreshCampaigns, refreshMetadata]);

  const play = useCampaignPlay(activeId, refreshAll);

  useEffect(() => {
    if (
      !pendingOpeningId ||
      play.detail?.campaign.id !== pendingOpeningId ||
      play.detail.opening.status !== "needed" ||
      play.state.streaming
    ) return;
    setPendingOpeningId(null);
    void play.generateOpening(pendingOpeningId);
  }, [pendingOpeningId, play.detail, play.generateOpening, play.state.streaming]);

  useEffect(() => {
    refreshAll().catch((error: unknown) => setGlobalError(errorMessage(error))).finally(() => setInitializing(false));
  }, [refreshAll]);

  const indexingActive = useMemo(
    () => lore.some((document) => document.status === "queued" || document.status === "indexing"), [lore]
  );
  useEffect(() => {
    const interval = window.setInterval(() => {
      refreshMetadata().catch((error: unknown) => setGlobalError(errorMessage(error)));
    }, indexingActive ? 2500 : 20_000);
    return () => window.clearInterval(interval);
  }, [indexingActive, refreshMetadata]);
  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 4000);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  function toggleRail() {
    setRailCollapsed((current) => {
      window.localStorage.setItem("dndllm26.campaignRailCollapsed", String(!current));
      return !current;
    });
  }

  async function createCampaign() {
    if (creatingCampaign || !protagonistId) return;
    setCreatingCampaign(true);
    setGlobalError(null);
    const payload: CampaignPayload = {
      ...campaignDraft,
      title: campaignDraft.title.trim(), setting: campaignDraft.setting.trim(), tone: campaignDraft.tone.trim(),
      protagonist_id: protagonistId, companion_ids: companionIds.filter((id) => id !== protagonistId),
      lore_document_ids: selectedLoreIds
    };
    try {
      const campaign = await postJson<Campaign>("/campaigns", payload);
      await refreshAll();
      setActiveId(campaign.id);
      setPendingOpeningId(campaign.id);
      setShowCampaignDialog(false);
      setProtagonistId(null); setCompanionIds([]); setSelectedLoreIds([]); setCampaignDraft(DEFAULT_CAMPAIGN);
      setNotice("Adventure created. Preparing the opening scene...");
    } catch (error) { setGlobalError(errorMessage(error)); }
    finally { setCreatingCampaign(false); }
  }

  async function createHero(payload: HeroPayload) {
    try { await postJson<Hero>("/heroes", payload); await refreshMetadata(); setNotice("Hero added to your library."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function updateHero(id: number, payload: HeroPayload) {
    try { await patchJson<Hero>(`/heroes/${id}`, payload); await refreshMetadata(); setNotice("Hero changes saved."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function deleteHero(id: number) {
    try { await deleteJson(`/heroes/${id}`); await refreshMetadata(); setNotice("Hero removed from the reusable library."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function uploadLore(file: File) {
    try { await uploadFile<LoreDocument>("/lore/upload", file); await refreshMetadata(); setNotice("Lore stored and queued for indexing."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function refreshLore(force = false) {
    try { await postJson(`/lore/refresh-index?force=${force ? "true" : "false"}`); await refreshMetadata(); setNotice(force ? "Failed lore queued again." : "Lore queue refreshed."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function deleteLore(id: number) {
    try { await deleteJson(`/lore/${id}`); setSelectedLoreIds((current) => current.filter((value) => value !== id)); await refreshMetadata(); setNotice("Lore document removed."); }
    catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }
  async function updateCampaign(id: number, changes: { title?: string; archived?: boolean }) {
    try {
      await patchJson<Campaign>(`/campaigns/${id}`, changes); await refreshCampaigns();
      setNotice(changes.archived === true ? "Campaign archived." : changes.archived === false ? "Campaign restored." : "Campaign renamed.");
    } catch (error) { setGlobalError(errorMessage(error)); throw error; }
  }

  return (
    <div className="app-shell">
      <a href="#main-play" className="skip-link">Skip to adventure</a>
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><Shield size={20} aria-hidden="true" /></span>
          <div><strong>DNDLLM26</strong><small>Local AI Dungeon Master</small></div></div>
        <button type="button" className="mobile-campaign-button" onClick={() => setDrawer("campaigns")}>
          <Menu size={18} aria-hidden="true" /><span>{activeCampaign?.title ?? "Campaigns"}</span>
        </button>
        <div className="top-actions" aria-label="Application tools">
          <button type="button" onClick={() => setDrawer("heroes")}><UserRound size={17} /><span>Heroes</span></button>
          <button type="button" onClick={() => setDrawer("lore")}><BookOpen size={17} /><span>Lore</span></button>
          <button type="button" onClick={() => setDrawer("runtime")} className={`runtime-button ${health?.status ?? "loading"}`}>
            <BrainCircuit size={17} /><span>{runtimeLabel(health)}</span>
          </button>
          <button type="button" className="icon-button" onClick={() => setDrawer("settings")} aria-label="Application settings"><Settings size={17} /></button>
        </div>
      </header>

      {globalError && <div className="global-error" role="alert"><AlertTriangle size={17} /><span>{globalError}</span>
        <button type="button" onClick={() => void refreshAll().then(() => setGlobalError(null)).catch((error) => setGlobalError(errorMessage(error)))}><RefreshCw size={15} /> Retry</button>
        <button type="button" className="icon-button" onClick={() => setGlobalError(null)} aria-label="Dismiss application error"><X size={15} /></button></div>}
      {notice && <div className="toast" role="status"><Check size={16} />{notice}</div>}

      <main className={railCollapsed ? "workspace rail-collapsed" : "workspace"}>
        <CampaignSidebar campaigns={campaigns} activeId={activeId} collapsed={railCollapsed}
          onSelect={(id) => { setActiveId(id); setDrawer(null); }} onCreate={() => setShowCampaignDialog(true)}
          onToggleCollapsed={toggleRail} onRestore={(id) => void updateCampaign(id, { archived: false })} />
        <PlayScreen detail={play.detail} state={initializing && !activeId ? { ...play.state, phase: "loading", statusMessage: "Loading your adventures..." } : play.state} onSetAction={play.setAction}
          onSubmitAction={play.submitAction} onSubmitCombatAction={play.submitCombatAction}
          onResolvePendingRoll={play.resolvePendingRoll} onRetry={play.retry} onCancel={play.cancel}
          onGenerateOpening={() => play.generateOpening()}
          onClearError={() => play.dispatch({ type: "clear_error" })} onLoadOlderTurns={play.loadOlderTurns}
          onRest={play.rest} onCompleteQuest={play.completeQuest} onCreateCampaign={() => setShowCampaignDialog(true)}
          onOpenParty={() => setDrawer("party")} onOpenJournal={() => setDrawer("journal")}
          utilityDrawer={drawer === "party" || drawer === "journal" ? drawer : null}
          onCloseUtilityDrawer={() => setDrawer(null)} mutationPending={play.mutationPending}
          maxActionChars={health?.request_max_chars ?? 2000} />
      </main>

      <nav className="mobile-nav" aria-label="Primary navigation">
        <button type="button" onClick={() => setDrawer("campaigns")}><ScrollText size={18} /><span>Campaigns</span></button>
        <button type="button" onClick={() => setDrawer("party")} disabled={!activeCampaign}><Users size={18} /><span>Party</span></button>
        <button type="button" onClick={() => setDrawer("journal")} disabled={!activeCampaign}><BookOpen size={18} /><span>Journal</span></button>
        <button type="button" onClick={() => setDrawer("heroes")}><UserRound size={18} /><span>Library</span></button>
      </nav>

      {drawer === "campaigns" && <Drawer title="Campaigns" eyebrow="Adventure Library" side="left" onClose={() => setDrawer(null)}>
        <CampaignSidebar campaigns={campaigns} activeId={activeId} collapsed={false}
          onSelect={(id) => { setActiveId(id); setDrawer(null); }} onCreate={() => { setDrawer(null); setShowCampaignDialog(true); }}
          onToggleCollapsed={() => undefined} onRestore={(id) => void updateCampaign(id, { archived: false })} />
      </Drawer>}
      {drawer === "heroes" && <Drawer title="Hero Library" eyebrow="Reusable Characters" onClose={() => setDrawer(null)}><HeroManager heroes={heroes} onCreate={createHero} onUpdate={updateHero} onDelete={deleteHero} /></Drawer>}
      {drawer === "lore" && <Drawer title="Lore Library" eyebrow="Campaign Sources" onClose={() => setDrawer(null)}><LorePanel lore={lore} worker={health?.worker ?? null} onRefresh={refreshLore} onUpload={uploadLore} onDelete={deleteLore} /></Drawer>}
      {drawer === "runtime" && <Drawer title="Local Runtime" eyebrow="System Status" onClose={() => setDrawer(null)}><RuntimeDetails health={health} onRefresh={refreshMetadata} /></Drawer>}
      {drawer === "settings" && <Drawer title="Settings" eyebrow="Models & Adventure" onClose={() => setDrawer(null)}>
        <ApplicationSettings campaign={activeCampaign} onModelsSaved={async (message) => { await refreshMetadata(); setNotice(message); }}
          onError={(error) => setGlobalError(errorMessage(error))}
          onRename={activeCampaign ? (title) => updateCampaign(activeCampaign.id, { title }) : undefined}
          onArchive={activeCampaign ? async () => { await updateCampaign(activeCampaign.id, { archived: true }); setDrawer(null); } : undefined}
          onExport={activeCampaign ? () => downloadJson(`/campaigns/${activeCampaign.id}/export`, `campaign-${activeCampaign.id}.json`) : undefined} />
      </Drawer>}

      {showCampaignDialog && <CreateCampaignDialog draft={campaignDraft} heroes={heroes} lore={lore}
        protagonistId={protagonistId} companionIds={companionIds} selectedLoreIds={selectedLoreIds}
        creating={creatingCampaign} onDraftChange={setCampaignDraft}
        onSelectProtagonist={(id) => { setProtagonistId(id); setCompanionIds((current) => current.filter((value) => value !== id)); }}
        onToggleCompanion={(id) => setCompanionIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])}
        onToggleLore={(id) => setSelectedLoreIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])}
        onCancel={() => setShowCampaignDialog(false)} onConfirm={() => void createCampaign()} />}
    </div>
  );
}

function runtimeLabel(health: Health | null): string {
  if (!health) return "Connecting";
  if (health.model_runtime.narrator.status === "failed") return "Narrator needs attention";
  if (health.status === "error" || health.ollama.status !== "ok") return "Runtime degraded";
  return health.model_runtime.narrator.status === "healthy" ? "Local AI ready" : "Local AI available";
}

function RuntimeDetails({ health, onRefresh }: { health: Health | null; onRefresh: () => Promise<void> }) {
  const worker = health?.worker;
  const components = health ? [["Database", health.database], ["Storage", health.filesystem], ["Ollama", health.ollama], ["Lore worker", health.worker]] as const : [];
  return <div className="runtime-details"><div className={`runtime-summary ${health?.status ?? "loading"}`}><BrainCircuit size={24} /><div>
    <strong>{health?.status === "ok" ? "Everything is ready" : health ? "Some services need attention" : "Checking local services"}</strong><span>Your campaign data remains on this machine.</span></div></div>
    {components.map(([label, component]) => <article className="runtime-component" key={label}><span className={`status-dot ${component.status}`} /><div><strong>{label}</strong><span>{component.detail ?? component.status}</span></div><b>{component.status}</b></article>)}
    {health && Object.entries(health.model_runtime).map(([role, component]) => <article className="runtime-component" key={role}><span className={`status-dot ${component.status}`} /><div><strong>{role === "narrator" ? "Narrator generation" : role === "utility" ? "Rules and memory" : "Lore embeddings"}</strong><span>{component.detail ?? (component.status === "unverified" ? "Installed; not exercised in this session" : component.status)}</span></div><b>{component.status}</b></article>)}
    {health && <dl className="model-list"><dt>Dungeon Master</dt><dd>{health.chat_model}</dd><dt>Rules utility</dt><dd>{health.utility_model}</dd><dt>Embeddings</dt><dd>{health.embed_model}</dd></dl>}
    <div className="worker-note"><span className={`status-dot ${worker?.running ? "ok" : "error"}`} />{worker?.running ? worker.active_document_id ? `Indexing lore document ${worker.active_document_id}` : `${worker.queued_count} lore documents queued` : "Lore indexer unavailable"}</div>
    <button type="button" className="secondary" onClick={() => void onRefresh()}><RefreshCw size={16} />Refresh status</button></div>;
}

function CampaignSettings({ campaign, onRename, onArchive, onExport }: { campaign: Campaign; onRename: (title: string) => Promise<void>; onArchive: () => Promise<void>; onExport: () => Promise<void> }) {
  const [title, setTitle] = useState(campaign.title);
  const [pending, setPending] = useState<string | null>(null);
  const [confirmArchive, setConfirmArchive] = useState(false);
  async function run(kind: string, action: () => Promise<void>) { setPending(kind); try { await action(); } catch { /* Parent surfaces the actionable error. */ } finally { setPending(null); } }
  return <div className="settings-stack"><label><span>Campaign name</span><input value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} /></label>
    <button type="button" className="primary" disabled={!title.trim() || title.trim() === campaign.title || pending !== null} onClick={() => void run("rename", () => onRename(title.trim()))}><Check size={16} />{pending === "rename" ? "Saving..." : "Save name"}</button>
    <section className="settings-card"><Download size={20} /><div><strong>Export campaign</strong><p>Download the story, party, world, quests, and encounters as JSON. Lore PDFs are not included.</p></div><button type="button" className="secondary" disabled={pending !== null} onClick={() => void run("export", onExport)}>{pending === "export" ? "Preparing..." : "Export JSON"}</button></section>
    <section className="settings-card archive-card"><Archive size={20} /><div><strong>Archive campaign</strong><p>Hide this adventure without deleting its story or recovery state.</p></div>
      {!confirmArchive ? <button type="button" className="secondary" onClick={() => setConfirmArchive(true)}>Archive...</button> : <div className="confirm-actions"><button type="button" className="secondary" onClick={() => setConfirmArchive(false)}>Cancel</button><button type="button" className="danger-button" disabled={pending !== null} onClick={() => void run("archive", onArchive)}>Confirm archive</button></div>}</section></div>;
}

function ApplicationSettings({ campaign, onModelsSaved, onError, onRename, onArchive, onExport }: {
  campaign: Campaign | null;
  onModelsSaved: (message: string) => Promise<void>;
  onError: (error: unknown) => void;
  onRename?: (title: string) => Promise<void>;
  onArchive?: () => Promise<void>;
  onExport?: () => Promise<void>;
}) {
  const [saved, setSaved] = useState<ModelSettings | null>(null);
  const [draft, setDraft] = useState<ModelSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [confirmReindex, setConfirmReindex] = useState(false);
  useEffect(() => {
    getJson<ModelSettings>("/settings/models")
      .then((value) => { setSaved(value); setDraft(value); })
      .catch(onError)
      .finally(() => setLoading(false));
  }, []); // Settings load once per drawer mount.
  const dirty = Boolean(saved && draft && ["chat_model", "utility_model", "embed_model", "narration_style"]
    .some((key) => saved[key as keyof ModelSettings] !== draft[key as keyof ModelSettings]));
  const embedChanged = Boolean(saved && draft && saved.embed_model !== draft.embed_model);
  const completion = draft?.models.filter((model) => model.capabilities.includes("completion")) ?? [];
  const embeddings = draft?.models.filter((model) => model.capabilities.includes("embedding")) ?? [];
  async function save(confirm = false) {
    if (!draft || !dirty || saving) return;
    if (embedChanged && draft.lore_document_count > 0 && !confirm) { setConfirmReindex(true); return; }
    setSaving(true);
    try {
      const value = await patchJson<ModelSettings>("/settings/models", {
        chat_model: draft.chat_model, utility_model: draft.utility_model,
        embed_model: draft.embed_model, narration_style: draft.narration_style,
        confirm_lore_reindex: confirm
      });
      setSaved(value); setDraft(value); setConfirmReindex(false);
      await onModelsSaved(value.reindex_queued ? `${value.reindex_queued} lore documents queued for reindexing.` : "AI settings saved.");
    } catch (error) { onError(error); }
    finally { setSaving(false); }
  }
  function options(models: { name: string }[], current: string) {
    const available = models.some((model) => model.name === current);
    return <>{!available && <option value={current}>{current} (unavailable)</option>}
      {models.map((model) => <option value={model.name} key={model.name}>{model.name}</option>)}</>;
  }
  return <div className="settings-stack">
    <section className="settings-section"><div><span className="eyebrow">Local AI</span><h3>Model roles</h3><p>Installed, compatible Ollama models only. Changes apply from the next turn.</p></div>
      {loading && <span className="hint">Loading installed models...</span>}
      {!loading && draft && <div className="model-settings-grid">
        <div className="model-role-health" role="status">Narrator: <strong>{draft.model_runtime.narrator.status}</strong>{draft.model_runtime.narrator.detail && <span>{draft.model_runtime.narrator.detail}</span>}</div>
        <label><span>Dungeon Master</span><small>Writes scenes and resolves outcomes.</small><select value={draft.chat_model} onChange={(event) => setDraft({ ...draft, chat_model: event.target.value })}>{options(completion, draft.chat_model)}</select></label>
        <label><span>Rules and memory</span><small>Classifies actions and maintains world state.</small><select value={draft.utility_model} onChange={(event) => setDraft({ ...draft, utility_model: event.target.value })}>{options(completion, draft.utility_model)}</select></label>
        <label><span>Embeddings</span><small>Searches indexed campaign lore.</small><select value={draft.embed_model} onChange={(event) => { setConfirmReindex(false); setDraft({ ...draft, embed_model: event.target.value }); }}>{options(embeddings, draft.embed_model)}</select></label>
        <fieldset><legend>Narration style</legend><div className="style-options">{(["focused", "balanced", "cinematic"] as const).map((style) => <label key={style}><input type="radio" name="narration-style" checked={draft.narration_style === style} onChange={() => setDraft({ ...draft, narration_style: style })} /><span>{style}</span></label>)}</div><small>Balanced adapts normal turns to roughly 90–160 words.</small></fieldset>
      </div>}
      {confirmReindex && draft && <div className="reindex-warning" role="alert"><AlertTriangle size={18} /><div><strong>Reindex {draft.lore_document_count} lore documents?</strong><p>Lore search will be temporarily unavailable while compatible indexes are rebuilt.</p></div><div className="confirm-actions"><button className="secondary" type="button" onClick={() => setConfirmReindex(false)}>Cancel</button><button className="primary" type="button" onClick={() => void save(true)}>Save and reindex</button></div></div>}
      <button type="button" className="primary" disabled={!dirty || saving || confirmReindex} onClick={() => void save()}>{saving ? "Saving..." : "Save AI settings"}</button>
    </section>
    {campaign && onRename && onArchive && onExport && <><div className="settings-divider" /><section><span className="eyebrow">Current adventure</span><h3>{campaign.title}</h3></section><CampaignSettings key={campaign.id} campaign={campaign} onRename={onRename} onArchive={onArchive} onExport={onExport} /></>}
  </div>;
}
