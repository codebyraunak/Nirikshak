from fastapi import APIRouter, HTTPException

from app.services.remediation_engine import generate_remediation


router = APIRouter(
    prefix="/remediation",
    tags=["Remediation"],
)


@router.post("/{finding_id}/generate")
def generate_finding_remediation(
    finding_id: str,
    finding: dict,
):
    """
    Generate a remediation proposal for a compliance finding.

    No configuration is automatically applied.
    Human approval is always required.
    """

    try:
        result = generate_remediation(finding)

        result["finding_id"] = finding_id

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Remediation generation failed: {str(exc)}",
        )