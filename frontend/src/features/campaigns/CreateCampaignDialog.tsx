import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, LoaderCircle, Shield, Sparkles, X } from "lucide-react";
import { useState } from "react";
import { Dialog } from "../../components/Dialog";
import type { CampaignPayload, Hero, LoreDocument } from "../../types";

export type CampaignDraft = Omit<CampaignPayload, "protagonist_id" | "companion_ids" | "lore_document_ids">;

export function CreateCampaignDialog({ draft, heroes, lore, protagonistId, companionIds,
  selectedLoreIds, creating, onDraftChange, onSelectProtagonist, onToggleCompanion,
  onToggleLore, onCancel, onConfirm }: {
  draft: CampaignDraft; heroes: Hero[]; lore: LoreDocument[]; protagonistId: number | null;
  companionIds: number[]; selectedLoreIds: number[]; creating: boolean;
  onDraftChange: (draft: CampaignDraft) => void; onSelectProtagonist: (id: number) => void;
  onToggleCompanion: (id: number) => void; onToggleLore: (id: number) => void;
  onCancel: () => void; onConfirm: () => void;
}) {
  const [step, setStep] = useState(1);
  const conceptValid = Boolean(draft.title.trim() && draft.setting.trim() && draft.tone.trim());
  const canCreate = conceptValid && Boolean(protagonistId);
  const protagonist = heroes.find((hero) => hero.id === protagonistId);
  return <Dialog titleId="create-campaign-title" onClose={creating ? () => undefined : onCancel} className="campaign-dialog">
    <div className="modal-head"><div><span className="eyebrow">New adventure · Step {step} of 3</span>
      <h2 id="create-campaign-title">{step === 1 ? "Shape your world" : step === 2 ? "Choose your party" : "Review your adventure"}</h2></div>
      <button type="button" className="icon-button" onClick={onCancel} disabled={creating} aria-label="Close campaign setup"><X size={17} /></button></div>
    <div className="step-progress" aria-label={`Step ${step} of 3`}><span className={step >= 1 ? "active" : ""} /><span className={step >= 2 ? "active" : ""} /><span className={step >= 3 ? "active" : ""} /></div>

    {step === 1 && <div className="wizard-page"><p className="step-intro">Give your Dungeon Master a strong starting point. You can keep the example or make it entirely yours.</p>
      <div className="campaign-form-grid"><label><span>Campaign title</span><input autoFocus data-dialog-initial-focus value={draft.title} maxLength={120} onChange={(event) => onDraftChange({ ...draft, title: event.target.value })} /></label>
        <label><span>Tone</span><input value={draft.tone} maxLength={160} placeholder="Hopeful, eerie, swashbuckling..." onChange={(event) => onDraftChange({ ...draft, tone: event.target.value })} /></label>
        <label className="full-field"><span>Setting brief</span><textarea value={draft.setting} maxLength={2000} placeholder="Where does this story begin, and what makes it interesting?" onChange={(event) => onDraftChange({ ...draft, setting: event.target.value })} /></label></div></div>}

    {step === 2 && <div className="wizard-page party-step"><section><div className="picker-section-title"><Shield size={14} /> Your protagonist</div>
      <p className="hint">This is the character you control directly.</p><div className="hero-picker-list">{heroes.length === 0 && <p className="empty-note">Create a hero in the Hero Library before starting a campaign.</p>}
        {heroes.map((hero) => <HeroChoice key={hero.id} hero={hero} selected={protagonistId === hero.id} onClick={() => onSelectProtagonist(hero.id)} />)}</div></section>
      {protagonistId && heroes.length > 1 && <section><div className="picker-section-title">Optional companions</div><p className="hint">The Dungeon Master controls companions during the adventure.</p>
        <div className="hero-picker-list compact">{heroes.filter((hero) => hero.id !== protagonistId).map((hero) => <HeroChoice key={hero.id} hero={hero} selected={companionIds.includes(hero.id)} onClick={() => onToggleCompanion(hero.id)} />)}</div></section>}</div>}

    {step === 3 && <div className="wizard-page review-page"><section className="review-card"><Sparkles size={22} /><div><span className="eyebrow">Campaign</span><h3>{draft.title}</h3><p>{draft.setting}</p><small>{draft.tone}</small></div></section>
      <section className="review-card"><Shield size={22} /><div><span className="eyebrow">Party</span><h3>{protagonist?.name ?? "Choose a protagonist"}</h3><p>{protagonist ? `${protagonist.ancestry} ${protagonist.character_class}` : "Return to the previous step to choose your hero."}</p><small>{companionIds.length ? `${companionIds.length} companion${companionIds.length === 1 ? "" : "s"}` : "Solo adventure"}</small></div></section>
      <section className="lore-review"><div className="picker-section-title"><BookOpen size={14} /> Optional campaign lore</div><p className="hint">Ready PDFs help the DM stay grounded in your setting.</p>
        <div className="hero-picker-list compact">{lore.length === 0 && <p className="empty-note">No lore is indexed. You can add it later from the Lore Library.</p>}
          {lore.map((document) => { const selected = selectedLoreIds.includes(document.id); const ready = document.status === "ready"; return <button type="button" className={selected ? "hero-select selected" : "hero-select"} key={document.id} disabled={!ready} onClick={() => onToggleLore(document.id)} aria-pressed={selected}>
            <strong>{document.filename}</strong><span>{ready ? `${document.chunks} indexed passages` : document.status}</span>{selected && <CheckCircle2 size={17} />}</button>; })}</div></section>
      {creating && <div className="creation-progress" role="status"><LoaderCircle className="spin" size={22} /><div><strong>Creating your world...</strong><span>The local DM is preparing the opening scene. This can take a moment.</span></div></div>}</div>}

    <div className="modal-actions wizard-actions"><button type="button" className="secondary" disabled={creating} onClick={() => step === 1 ? onCancel() : setStep(step - 1)}>{step > 1 && <ArrowLeft size={16} />}{step === 1 ? "Cancel" : "Back"}</button>
      {step < 3 ? <button type="button" className="primary inline-primary" disabled={(step === 1 && !conceptValid) || (step === 2 && !protagonistId)} onClick={() => setStep(step + 1)}>Continue <ArrowRight size={16} /></button>
        : <button type="button" className="primary inline-primary" disabled={!canCreate || creating} onClick={onConfirm}>{creating ? <LoaderCircle className="spin" size={17} /> : <BookOpen size={17} />}{creating ? "Preparing adventure..." : "Begin Adventure"}</button>}</div>
  </Dialog>;
}

function HeroChoice({ hero, selected, onClick }: { hero: Hero; selected: boolean; onClick: () => void }) {
  return <button type="button" className={selected ? "hero-select selected" : "hero-select"} onClick={onClick} aria-pressed={selected}>
    <strong>{hero.name}</strong><span>Level {hero.level} · {hero.ancestry} {hero.character_class}</span><small>{hero.backstory}</small>{selected && <CheckCircle2 size={17} />}
  </button>;
}
