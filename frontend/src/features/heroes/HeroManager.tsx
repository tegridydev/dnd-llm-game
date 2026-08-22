import { Edit3, Plus, Save, Shield, Trash2, UserPlus, X } from "lucide-react";
import { useState } from "react";
import { Dialog } from "../../components/Dialog";
import type { Hero, HeroPayload } from "../../types";

const EMPTY_HERO: HeroPayload = {
  name: "",
  ancestry: "Human",
  character_class: "Fighter",
  backstory: "",
  inventory: []
};

function parseInventory(value: string): string[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

export function HeroManager({
  heroes,
  onCreate,
  onUpdate,
  onDelete
}: {
  heroes: Hero[];
  onCreate: (payload: HeroPayload) => Promise<void>;
  onUpdate: (id: number, payload: HeroPayload) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}) {
  const [draft, setDraft] = useState<HeroPayload>(EMPTY_HERO);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [showForm, setShowForm] = useState(false);

  function edit(hero: Hero) {
    setEditingId(hero.id);
    setShowForm(true);
    setConfirmDeleteId(null);
    setDraft({
      name: hero.name,
      ancestry: hero.ancestry,
      character_class: hero.character_class,
      backstory: hero.backstory,
      inventory: parseInventory(hero.inventory_json),
      strength: hero.strength,
      dexterity: hero.dexterity,
      constitution: hero.constitution,
      intelligence: hero.intelligence,
      wisdom: hero.wisdom,
      charisma: hero.charisma
    });
  }

  function reset() {
    setEditingId(null);
    setDraft(EMPTY_HERO);
    setShowForm(false);
  }

  async function save() {
    if (!draft.name.trim() || saving) return;
    const payload: HeroPayload = {
      ...draft,
      name: draft.name.trim(),
      ancestry: draft.ancestry.trim() || "Human",
      character_class: draft.character_class.trim() || "Fighter",
      backstory: draft.backstory.trim() || "An adventurer looking for a reason to risk everything.",
      inventory: draft.inventory.map((item) => item.trim()).filter(Boolean)
    };
    setSaving(true);
    try {
      if (editingId) await onUpdate(editingId, payload);
      else await onCreate(payload);
      reset();
    } catch {
      // The application-level error banner owns mutation failures.
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel hero-manager">
      <div className="panel-title">
        <UserPlus size={16} aria-hidden="true" />
        <span>Hero Library</span>
        <button type="button" className="icon-button hero-add" onClick={() => setShowForm(true)}
          aria-label="Create hero"><Plus size={15} /></button>
      </div>
      <div className="hero-list">
        {heroes.length === 0 && <p className="empty-note">No reusable heroes yet. Create one to begin an adventure.</p>}
        {heroes.map((hero) => (
          <article className="hero-row" key={hero.id}>
            <Shield size={16} aria-hidden="true" />
            <div>
              <strong>{hero.name}</strong>
              <small>
                {hero.ancestry} {hero.character_class}
              </small>
            </div>
            <button
              type="button"
              className="icon-button"
              onClick={() => edit(hero)}
              aria-label={`Edit ${hero.name}`}
            >
              <Edit3 size={14} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="icon-button danger"
              onClick={() => setConfirmDeleteId(hero.id)}
              aria-label={`Delete ${hero.name}`}
            >
              <Trash2 size={14} aria-hidden="true" />
            </button>
          </article>
        ))}
      </div>

      {showForm && <><div className="form-stack">
        <label>
          <span>Hero name</span>
          <input
            value={draft.name}
            maxLength={100}
            onChange={(event) => setDraft({ ...draft, name: event.target.value })}
          />
        </label>
        <div className="hero-form-grid">
          <label>
            <span>Ancestry</span>
            <select
              value={draft.ancestry}
              onChange={(event) => setDraft({ ...draft, ancestry: event.target.value })}
            >
              {['Human', 'Elf', 'Dwarf', 'Halfling'].map((value) => <option key={value}>{value}</option>)}
            </select>
          </label>
          <label>
            <span>Class</span>
            <select
              value={draft.character_class}
              onChange={(event) => setDraft({ ...draft, character_class: event.target.value })}
            >
              {['Fighter', 'Rogue', 'Cleric', 'Wizard', 'Ranger', 'Bard'].map((value) => <option key={value}>{value}</option>)}
            </select>
          </label>
        </div>
        <label>
          <span>Backstory</span>
          <textarea
            value={draft.backstory}
            maxLength={4000}
            onChange={(event) => setDraft({ ...draft, backstory: event.target.value })}
          />
        </label>
        <label>
          <span>Starting inventory · separate items with commas</span>
          <input
            value={draft.inventory.join(", ")}
            onChange={(event) =>
              setDraft({
                ...draft,
                inventory: event.target.value.split(",").map((item) => item.trim())
              })
            }
          />
        </label>
        {draft.inventory.some(Boolean) && <div className="inventory-chips" aria-label="Inventory preview">
          {draft.inventory.filter(Boolean).map((item, index) => <span key={`${item}-${index}`}>{item}</span>)}
        </div>}
        <p className="hint">Ability scores, proficiencies, equipment and class actions are prepared automatically.</p>
      </div>

      <div className="hero-actions">
        {editingId && (
          <button type="button" className="secondary" onClick={reset} disabled={saving}>
            <X size={16} aria-hidden="true" /> Cancel
          </button>
        )}
        <button
          type="button"
          className="primary"
          onClick={save}
          disabled={!draft.name.trim() || saving}
        >
          <Save size={16} aria-hidden="true" /> {editingId ? "Save Hero" : "Add Hero"}
        </button>
      </div></>}
      {confirmDeleteId !== null && (() => {
        const hero = heroes.find((item) => item.id === confirmDeleteId);
        return <Dialog titleId="delete-hero-title" onClose={() => setConfirmDeleteId(null)} className="confirm-dialog">
          <div className="modal-head"><div><span className="eyebrow">Remove from library</span><h2 id="delete-hero-title">Delete {hero?.name ?? "this hero"}?</h2></div>
            <button type="button" className="icon-button" onClick={() => setConfirmDeleteId(null)} aria-label="Cancel deletion"><X size={17} /></button></div>
          <p className="dialog-copy">Existing campaign characters are preserved, but this reusable hero cannot be selected for new adventures.</p>
          <div className="modal-actions"><button type="button" className="secondary" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
            <button type="button" className="danger-button" onClick={async () => { try { await onDelete(confirmDeleteId); setConfirmDeleteId(null); } catch { /* Parent reports the error. */ } }}><Trash2 size={15} />Delete hero</button></div>
        </Dialog>;
      })()}
    </section>
  );
}
