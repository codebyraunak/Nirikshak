from typing import Any


def generate_remediation(finding: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a remediation proposal from a real compliance finding.

    This function ONLY proposes a change.
    It does NOT modify any device configuration.
    """

    control_id = finding.get("control_id")
    actual = finding.get("actual")
    expected = finding.get("expected")
    evidence = finding.get("evidence")

    raw_config = None

    if isinstance(evidence, dict):
        raw_config = evidence.get("raw")

    if not raw_config:
        raw_config = "Configuration evidence unavailable"

    proposed_config = raw_config

    if actual is not None and expected is not None:
        proposed_config = raw_config.replace(
            str(actual),
            str(expected),
            1,
        )

    return {
        "success": True,
        "control_id": control_id,
        "before": raw_config,
        "proposed_change": proposed_config,
        "actual": actual,
        "expected": expected,
        "requires_human_approval": True,
        "auto_apply": False,
        "status": "PROPOSED",
    }