from typing import Any

from app.core.supabase import supabase


# ---------------------------------------------------------
# Helper: normalize values
# ---------------------------------------------------------

def normalize_value(value: Any) -> Any:
    """
    Normalize values so comparisons between database rules
    and semantic-model values are predictable.
    """

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered == "true":
            return True

        if lowered == "false":
            return False

        if lowered in {"null", "none"}:
            return None

        # Convert numeric strings to numbers
        try:
            if "." in lowered:
                return float(lowered)

            return int(lowered)

        except ValueError:
            return value.strip()

    return value


# ---------------------------------------------------------
# Rule evaluator
# ---------------------------------------------------------

def evaluate_rule(
    actual: Any,
    operator: str,
    expected: Any,
) -> bool:

    actual = normalize_value(actual)
    expected = normalize_value(expected)

    operator = operator.lower().strip()

    if operator == "equals":
        return actual == expected

    if operator == "not_equals":
        return actual != expected

    if operator == "less_than_or_equal":
        return actual <= expected

    if operator == "greater_than_or_equal":
        return actual >= expected

    if operator == "less_than":
        return actual < expected

    if operator == "greater_than":
        return actual > expected

    if operator == "present":
        return actual is not None

    if operator == "absent":
        return actual is None

    raise ValueError(
        f"Unsupported compliance operator: {operator}"
    )


# ---------------------------------------------------------
# Find semantic object
# ---------------------------------------------------------

def find_semantic_object(
    semantic_objects: list[dict],
    control_id: str,
):
    """
    Find the semantic object corresponding to a security
    control.
    """

    for obj in semantic_objects:

        if obj.get("control_id") == control_id:
            return obj

    return None


# ---------------------------------------------------------
# Build evidence
# ---------------------------------------------------------

def build_evidence(
    semantic_object: dict | None,
):
    """
    Return configuration evidence when the control is
    explicitly represented.

    Missing controls intentionally have no fabricated
    configuration line.
    """

    if not semantic_object:
        return None

    evidence = semantic_object.get(
        "evidence"
    )

    if not evidence:
        return None

    return {
        "line": evidence.get("line"),
        "raw": evidence.get("raw"),
    }


# ---------------------------------------------------------
# Missing-control evaluation
# ---------------------------------------------------------

def evaluate_missing_control(
    operator: str,
    expected: Any,
):
    """
    Decide the result when a configuration control is not
    explicitly represented in the semantic model.

    Security logic:

    - equals false:
        absence normally means the feature is disabled
        → PASS

    - equals true:
        absence means the required feature is not enabled
        → FAIL

    - present:
        absence → FAIL

    - absent:
        absence → PASS

    - numeric comparisons:
        absence cannot satisfy the requirement
        → FAIL

    This does NOT fabricate configuration evidence.
    """

    operator = operator.lower().strip()
    expected = normalize_value(expected)

    if operator == "equals":

        if expected is False:
            return True

        if expected is True:
            return False

    if operator == "not_equals":

        if expected is False:
            return False

        if expected is True:
            return True

    if operator == "present":
        return False

    if operator == "absent":
        return True

    # For numeric requirements such as:
    # SESSION_IDLE_TIMEOUT <= 10
    #
    # If the configuration does not contain the control,
    # we cannot prove compliance.
    return False


# ---------------------------------------------------------
# Main compliance assessment
# ---------------------------------------------------------

def assess_semantic_model(
    semantic_model: dict,
    framework: str = "CIS",
) -> dict:

    semantic_objects = semantic_model.get(
        "semantic_objects",
        [],
    )

    # -----------------------------------------------------
    # Load framework rules from Supabase
    # -----------------------------------------------------

    response = (
        supabase
        .table("framework_rules")
        .select("*")
        .eq("framework", framework)
        .execute()
    )

    rules = response.data or []

    assessments = []
    findings = []

    passed = 0
    failed = 0

    # -----------------------------------------------------
    # Evaluate every framework rule
    # -----------------------------------------------------

    for index, rule in enumerate(rules, start=1):

        control_id = rule.get(
            "control_id"
        )

        operator = rule.get(
            "operator",
            "equals",
        )

        expected = rule.get(
            "expected_value"
        )

        severity = rule.get(
            "severity",
            "MEDIUM",
        )

        reference = rule.get(
            "reference"
        )

        semantic_object = find_semantic_object(
            semantic_objects,
            control_id,
        )

        # -------------------------------------------------
        # Control is explicitly represented
        # -------------------------------------------------

        if semantic_object:

            actual = semantic_object.get(
                "value"
            )

            is_compliant = evaluate_rule(
                actual=actual,
                operator=operator,
                expected=expected,
            )

            evidence = build_evidence(
                semantic_object
            )

            status = (
                "PASS"
                if is_compliant
                else "FAIL"
            )

        # -------------------------------------------------
        # Control is missing
        # -------------------------------------------------

        else:

            actual = None

            is_compliant = evaluate_missing_control(
                operator=operator,
                expected=expected,
            )

            evidence = None

            status = (
                "PASS"
                if is_compliant
                else "FAIL"
            )

        # -------------------------------------------------
        # Counters
        # -------------------------------------------------

        if status == "PASS":
            passed += 1

        else:
            failed += 1

        # -------------------------------------------------
        # Assessment object
        # -------------------------------------------------

        assessment = {
            "control_id": control_id,
            "framework": framework,
            "operator": operator,
            "actual": actual,
            "expected": expected,
            "status": status,
            "severity": severity,
            "evidence": evidence,
        }

        if reference:
            assessment["reference"] = reference

        assessments.append(
            assessment
        )

        # -------------------------------------------------
        # Create finding for failures
        # -------------------------------------------------

        if status == "FAIL":

            finding = {
                "finding_id": f"F-{len(findings) + 1:04d}",
                "framework": framework,
                "control_id": control_id,
                "severity": severity,
                "status": "OPEN",
                "actual": actual,
                "expected": expected,
                "operator": operator,
                "evidence": evidence,
            }

            if reference:
                finding["reference"] = reference

            findings.append(
                finding
            )

    # -----------------------------------------------------
    # Compliance percentage
    # -----------------------------------------------------

    total_assessed = len(
        assessments
    )

    if total_assessed > 0:

        compliance_percentage = round(
            (passed / total_assessed) * 100,
            2,
        )

    else:

        compliance_percentage = 0.0

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {
        "framework": framework,
        "rules_loaded": len(rules),
        "total_assessed": total_assessed,
        "passed": passed,
        "failed": failed,
        "compliance_percentage": compliance_percentage,
        "assessments": assessments,
        "findings": findings,
    }