import { AlertTriangle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useId, useState } from "react";
import { errorMessage, getJson, patchJson } from "../../lib/api";
import type { ModelSettings } from "../../types";

type Draft = Pick<ModelSettings, "chat_model" | "utility_model" | "embed_model" | "narration_style">;

function ModelInput({ label, value, models, hint, onChange, disabled }: {
  label: string;
  value: string;
  models: ModelSettings["models"];
  hint: string;
  onChange: (value: string) => void;
  disabled: boolean;
}) {
  const id = useId();
  return (
    <label>
      <span id={id + "-label"}>{label}</span>
      <input aria-labelledby={id + "-label"} aria-describedby={id + "-hint"} list={id} value={value} maxLength={160} disabled={disabled}
        onChange={(event) => onChange(event.target.value)} autoComplete="off" spellCheck={false} />
      <datalist id={id}>{models.map(({ name }) => <option value={name} key={name} />)}</datalist>
      <small id={id + "-hint"}>{hint}</small>
    </label>
  );
}

export function ModelSettingsPanel({ onSaved }: { onSaved: (message: string) => void }) {
  const [saved, setSaved] = useState<ModelSettings | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [confirm, setConfirm] = useState(false);

  const discover = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError("");
    try {
      const value = await getJson<ModelSettings>(`/settings/models${signal ? "" : "?refresh=true"}`, signal);
      if (signal?.aborted) return;
      setSaved(value);
      setDraft((current) => current ?? value);
    } catch (failure) {
      if (!signal?.aborted) setError(errorMessage(failure));
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void discover(controller.signal);
    return () => controller.abort();
  }, [discover]);

  const dirty = Boolean(saved && draft && (
    saved.chat_model !== draft.chat_model || saved.utility_model !== draft.utility_model ||
    saved.embed_model !== draft.embed_model || saved.narration_style !== draft.narration_style
  ));

  function change(field: keyof Draft, value: string) {
    setConfirm(false);
    setDraft((current) => current ? { ...current, [field]: value } : current);
  }

  async function save(approved = false) {
    if (!draft || !saved || !draft.chat_model.trim() || !dirty || saving) return;
    if (draft.embed_model.trim() && draft.embed_model !== saved.embed_model && saved.lore_document_count && !approved) {
      setConfirm(true);
      return;
    }
    setSaving(true);
    setError("");
    try {
      const value = await patchJson<ModelSettings>("/settings/models", {
        chat_model: draft.chat_model.trim(),
        utility_model: draft.utility_model.trim(),
        embed_model: draft.embed_model.trim(),
        narration_style: draft.narration_style,
        confirm_lore_reindex: approved
      });
      setSaved(value);
      setDraft(value);
      setConfirm(false);
      onSaved(value.reindex_queued ? `${value.reindex_queued} lore documents queued for reindexing.` : "Model settings saved.");
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setSaving(false);
    }
  }

  const provider = saved?.provider;
  return (
    <section className="settings-section">
      <div className="model-heading">
        <div><h3>AI models</h3><p>{provider?.label ?? "Model connection"}{provider && (provider.remote ? " · Hosted" : " · Local")}</p></div>
        <button type="button" className="secondary" disabled={loading || saving}
          onClick={() => void discover()} aria-label="Refresh model list"><RefreshCw size={16} aria-hidden="true" /> Refresh</button>
      </div>
      {provider?.remote && <div className="connection-note"><AlertTriangle size={16} aria-hidden="true" /><p>Gameplay prompts, including retrieved lore passages, go to {provider.label}. {provider.embedding_remote ? "Lore indexing also sends PDF text to the configured embedding provider." : "Lore embeddings stay local or disabled."}</p></div>}
      {!provider?.remote && provider?.embedding_remote && <div className="connection-note"><AlertTriangle size={16} aria-hidden="true" /><p>Lore indexing sends PDF text to {provider.embedding_label}.</p></div>}
      <p className="hint">Switch provider and configure API keys in the server’s .env file, then restart.</p>
      {error && <p className="error" role="alert">{error}</p>}
      {loading && !saved && <p role="status">Connecting to model settings…</p>}
      {saved?.discovery_error && <p className="hint" role="status">{saved.discovery_error} You can enter model IDs manually.</p>}
      {saved && draft && <div className="model-settings-grid">
        <ModelInput label="Dungeon Master" value={draft.chat_model} models={saved.models}
          onChange={(value) => change("chat_model", value)} disabled={saving} hint="Writes the story and narrates outcomes." />
        <ModelInput label="Rules and memory" value={draft.utility_model} models={saved.models}
          onChange={(value) => change("utility_model", value)} disabled={saving} hint="Leave blank to use the Dungeon Master model." />
        <ModelInput label="Lore embeddings" value={draft.embed_model} models={saved.embedding_models}
          onChange={(value) => change("embed_model", value)} disabled={saving || !provider?.embedding_label}
          hint={provider?.embedding_label ? `${provider.embedding_label} · leave blank to pause lore indexing.` : "Disabled in server configuration."} />
        <fieldset><legend>Narration length</legend><div className="style-options">
          {(["focused", "balanced", "cinematic"] as const).map((style) => <label key={style}>
            <input type="radio" name="narration-style" checked={draft.narration_style === style}
              disabled={saving} onChange={() => change("narration_style", style)} /><span>{style}</span>
          </label>)}
        </div></fieldset>
      </div>}
      {confirm && saved ? <div className="reindex-warning" role="alert">
        <AlertTriangle size={18} aria-hidden="true" /><div><strong>Reindex {saved.lore_document_count} lore documents?</strong>
          <p>Search resumes as the new indexes become ready.</p></div>
        <div className="confirm-actions">
          <button type="button" className="secondary" disabled={saving} onClick={() => setConfirm(false)}>Cancel</button>
          <button type="button" className="primary" disabled={saving} onClick={() => void save(true)}>Save and reindex</button>
        </div>
      </div> : <button type="button" className="primary"
        disabled={!dirty || saving || !draft?.chat_model.trim()} onClick={() => void save()}>
        {saving ? "Saving…" : "Save AI settings"}
      </button>}
    </section>
  );
}
