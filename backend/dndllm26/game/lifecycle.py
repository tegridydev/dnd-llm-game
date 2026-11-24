from __future__ import annotations

from sqlmodel import Session

from dndllm26.core.errors import ConflictError, NotFoundError
from dndllm26.db.models import Campaign


def playable_campaign(session: Session, campaign_id: int) -> Campaign:
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        raise NotFoundError("Campaign not found.")
    if campaign.archived_at is not None:
        raise ConflictError("Restore this archived campaign before playing.")
    return campaign
