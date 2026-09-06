from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AuditBase(BaseModel):
    title: str
    description: Optional[str] = None
    vendor_name: Optional[str] = None
    status: str = Field(default="pending")
    metadata: Optional[Dict[str, Any]] = None


class AuditCreate(AuditBase):
    pass


class AuditResponse(AuditBase):
    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True
