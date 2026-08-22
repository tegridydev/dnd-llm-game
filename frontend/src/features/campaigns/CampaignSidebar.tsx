import { ArchiveRestore, ChevronLeft, ChevronRight, Plus, ScrollText } from "lucide-react";
import type { Campaign } from "../../types";

export function CampaignSidebar({
  campaigns,
  activeId,
  collapsed,
  onSelect,
  onCreate,
  onToggleCollapsed,
  onRestore
}: {
  campaigns: Campaign[];
  activeId: number | null;
  collapsed: boolean;
  onSelect: (id: number) => void;
  onCreate: () => void;
  onToggleCollapsed: () => void;
  onRestore: (id: number) => void;
}) {
  const active = campaigns.filter((campaign) => !campaign.archived_at);
  const archived = campaigns.filter((campaign) => campaign.archived_at);
  return (
    <nav className={collapsed ? "campaign-rail collapsed" : "campaign-rail"} aria-label="Campaigns">
      <div className="rail-controls">
        <span className="rail-label"><ScrollText size={16} aria-hidden="true" /><span>Campaigns</span></span>
        <button type="button" className="icon-button" onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expand campaign rail" : "Collapse campaign rail"}>
          {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
      </div>
      <button type="button" className="primary new-campaign" onClick={onCreate}>
        <Plus size={17} aria-hidden="true" />
        <span>New Campaign</span>
      </button>
      <div className="campaign-list">
        {active.length === 0 && <p className="hint">No active campaigns yet.</p>}
        {active.map((campaign) => (
          <button
            type="button"
            className={campaign.id === activeId ? "campaign active" : "campaign"}
            key={campaign.id}
            onClick={() => onSelect(campaign.id)}
            aria-current={campaign.id === activeId ? "page" : undefined}
          >
            <ScrollText size={18} aria-hidden="true" />
            <strong>{campaign.title}</strong>
            <span>{campaign.current_location ?? campaign.tone}</span>
            <small title={new Date(campaign.last_activity_at ?? campaign.updated_at).toLocaleString()}>
              {relativeActivity(campaign.last_activity_at ?? campaign.updated_at)} · {campaign.tone}
            </small>
          </button>
        ))}
      </div>
      {archived.length > 0 && (
        <details className="archived-campaigns">
          <summary>Archived <span>{archived.length}</span></summary>
          {archived.map((campaign) => (
            <div className="archived-row" key={campaign.id}>
              <span>{campaign.title}</span>
              <button type="button" className="icon-button" onClick={() => onRestore(campaign.id)}
                aria-label={`Restore ${campaign.title}`}>
                <ArchiveRestore size={15} aria-hidden="true" />
              </button>
            </div>
          ))}
        </details>
      )}
    </nav>
  );
}

function relativeActivity(value: string): string {
  const elapsed = Date.now() - new Date(value).getTime();
  if (!Number.isFinite(elapsed) || elapsed < 60_000) return "Just now";
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return days < 30 ? `${days}d ago` : new Date(value).toLocaleDateString();
}
