from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.supabase import supabase


router = APIRouter(
    prefix="/semantic",
    tags=["Semantic Validation"],
)


class SemanticValidationRequest(BaseModel):
    vendor: str
    platform: str
    raw_pattern: str

    security_control: str
    security_category: str

    approved_value: Any = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


@router.post("/validate")
def validate_semantic_mapping(
    request: SemanticValidationRequest,
):
    """
    Human-in-the-loop approval endpoint.

    An administrator reviews an AI-generated semantic mapping
    and approves it for future reuse.
    """

    try:
        existing = (
            supabase
            .table("semantic_mappings")
            .select("id")
            .eq("vendor", request.vendor)
            .eq("platform", request.platform)
            .eq("raw_pattern", request.raw_pattern)
            .eq("status", "APPROVED")
            .limit(1)
            .execute()
        )

        if existing.data:
            return {
                "success": True,
                "message": "Mapping is already approved.",
                "mapping_id": existing.data[0]["id"],
            }

        result = (
            supabase
            .table("semantic_mappings")
            .insert(
                {
                    "vendor": request.vendor,
                    "platform": request.platform,
                    "raw_pattern": request.raw_pattern,
                    "security_control": request.security_control,
                    "security_category": request.security_category,
                    "approved_value": request.approved_value,
                    "confidence": request.confidence,
                    "status": "APPROVED",
                }
            )
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to save semantic mapping.",
            )

        return {
            "success": True,
            "message": "Semantic mapping approved and stored.",
            "mapping": result.data[0],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic validation failed: {str(exc)}",
        )