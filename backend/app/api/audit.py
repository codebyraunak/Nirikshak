from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion import read_config_file
from app.services.vendor_detection import detect_vendor
from app.services.parser import parse_cisco_config
from app.services.compliance_engine import assess_semantic_model
from app.services.semantic_memory import find_approved_mapping
from app.ai.inference import qwen


router = APIRouter(
    prefix="/audits",
    tags=["Audits"],
)


# ---------------------------------------------------------
# Find unknown security-related configuration lines
# ---------------------------------------------------------

def get_unrecognized_security_lines(
    config_text: str,
    semantic_model: dict,
) -> list[dict]:
    """
    Find security-related configuration lines that were not
    already represented by the deterministic parser.

    These candidate lines are sent to Qwen for semantic
    interpretation.
    """

    recognized_lines = set()

    for obj in semantic_model.get(
        "semantic_objects",
        []
    ):
        evidence = obj.get(
            "evidence",
            {}
        )

        if "line" in evidence:
            recognized_lines.add(
                evidence["line"]
            )

    candidates = []

    security_keywords = (
        "aaa",
        "telnet",
        "http",
        "https",
        "ssh",
        "login",
        "authentication",
        "authorization",
        "account",
        "timeout",
        "management",
        "snmp",
        "logging",
        "ntp",
        "password",
        "enable",
        "privilege",
        "access",
    )

    lines = config_text.splitlines()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):

        line = raw_line.strip()

        if not line:
            continue

        # Already understood by deterministic parser
        if line_number in recognized_lines:
            continue

        # Ignore comments
        if line.startswith("!"):
            continue

        # Only send likely security-related unknown
        # configuration lines to Qwen.
        if any(
            keyword in line.lower()
            for keyword in security_keywords
        ):

            previous_lines = [
                l.strip()
                for l in lines[
                    max(0, line_number - 3):
                    line_number - 1
                ]
                if l.strip()
            ]

            next_lines = [
                l.strip()
                for l in lines[
                    line_number:
                    line_number + 2
                ]
                if l.strip()
            ]

            candidates.append(
                {
                    "line": line_number,
                    "text": line,
                    "previous_lines": previous_lines,
                    "next_lines": next_lines,
                }
            )

    return candidates


# ---------------------------------------------------------
# Audit endpoint
# ---------------------------------------------------------

@router.post("/analyze")
async def analyze_configuration(
    file: UploadFile = File(...),
    framework: str = "CIS",
):

    try:

        # -------------------------------------------------
        # 1. Read configuration
        # -------------------------------------------------

        config_text = await read_config_file(
            file
        )

        line_count = len(
            config_text.splitlines()
        )

        # -------------------------------------------------
        # 2. Detect vendor
        # -------------------------------------------------

        vendor_info = detect_vendor(
            config_text
        )

        vendor = vendor_info["vendor"]

        # -------------------------------------------------
        # 3. Deterministic parsing
        # -------------------------------------------------

        semantic_model = None

        if vendor == "Cisco":

            semantic_model = parse_cisco_config(
                config_text
            )

        else:

            return {
                "success": True,
                "filename": file.filename,
                "line_count": line_count,
                "vendor_detection": vendor_info,
                "parser": "not_available",
                "semantic_model": None,
                "compliance": None,
                "ai_interpretations": [],
                "message": (
                    f"Vendor '{vendor}' detected, "
                    "but a deterministic parser is "
                    "not implemented yet."
                ),
            }

        # -------------------------------------------------
        # 4. AI fallback + Adaptive Knowledge Base
        # -------------------------------------------------

        ai_interpretations = []

        unknown_lines = (
            get_unrecognized_security_lines(
                config_text,
                semantic_model,
            )
        )

        for candidate in unknown_lines:

            try:

                # -----------------------------------------
                # 4A. Check previously approved mappings
                # -----------------------------------------

                approved_mapping = find_approved_mapping(
                    vendor=vendor,
                    platform=vendor_info["platform"],
                    raw_pattern=candidate["text"],
                )

                if approved_mapping:

                    # -----------------------------------------
                    # Reuse human-approved mapping
                    # -----------------------------------------

                    semantic_model[
                        "semantic_objects"
                    ].append(
                        {
                            "control_id": approved_mapping[
                                "security_control"
                            ],
                            "category": approved_mapping[
                                "security_category"
                            ],
                            "value": approved_mapping[
                                "approved_value"
                            ],
                            "evidence": {
                                "line": candidate["line"],
                                "raw": candidate["text"],
                            },
                            "source": "human_approved_mapping",
                        }
                    )

                    semantic_model[
                        "object_count"
                    ] = len(
                        semantic_model[
                            "semantic_objects"
                        ]
                    )

                    ai_interpretations.append(
                        {
                            "security_control": approved_mapping[
                                "security_control"
                            ],
                            "value": approved_mapping[
                                "approved_value"
                            ],
                            "security_category": approved_mapping[
                                "security_category"
                            ],
                            "security_meaning": (
                                "Previously approved human "
                                "mapping reused from the "
                                "adaptive knowledge base."
                            ),
                            "confidence": approved_mapping.get(
                                "confidence",
                                1.0,
                            ),
                            "evidence": [
                                {
                                    "line": candidate["line"],
                                    "text": candidate["text"],
                                }
                            ],
                            "requires_human_validation": False,
                            "source": "human_approved_mapping",
                            "input_line": candidate["line"],
                            "input_text": candidate["text"],
                        }
                    )

                    # IMPORTANT:
                    # Do not call Qwen for an already
                    # approved mapping.
                    continue

                # -----------------------------------------
                # 4B. No approved mapping → call Qwen
                # -----------------------------------------

                result = qwen.interpret(
                    vendor=vendor,
                    platform=vendor_info[
                        "platform"
                    ],
                    section=(
                        "global configuration"
                    ),
                    previous_lines=candidate[
                        "previous_lines"
                    ],
                    current_line=candidate[
                        "text"
                    ],
                    next_lines=candidate[
                        "next_lines"
                    ],
                )

                # -----------------------------------------
                # 4C. Attach trusted backend metadata
                # -----------------------------------------

                result["source"] = "qwen"

                # Always trust the original
                # configuration line.
                result["input_line"] = (
                    candidate["line"]
                )

                result["input_text"] = (
                    candidate["text"]
                )

                # -----------------------------------------
                # 4D. Store AI interpretation
                # -----------------------------------------

                ai_interpretations.append(
                    result
                )

                # -----------------------------------------
                # 4E. Add HIGH-CONFIDENCE Qwen result
                #     to semantic model.
                #
                # Low-confidence results remain outside
                # the semantic model until human validation.
                # -----------------------------------------

                confidence = result.get(
                    "confidence",
                    0.0,
                )

                requires_validation = result.get(
                    "requires_human_validation",
                    True,
                )

                security_control = result.get(
                    "security_control"
                )

                evidence = result.get(
                    "evidence",
                    [],
                )

                if (
                    security_control
                    and security_control != "UNKNOWN"
                    and confidence >= 0.85
                    and not requires_validation
                    and evidence
                ):

                    semantic_model[
                        "semantic_objects"
                    ].append(
                        {
                            "control_id": security_control,
                            "category": result.get(
                                "security_category",
                                "unknown",
                            ),
                            "value": result.get(
                                "value"
                            ),
                            "evidence": {
                                "line": candidate[
                                    "line"
                                ],
                                "raw": candidate[
                                    "text"
                                ],
                            },
                            "source": "qwen",
                            "confidence": confidence,
                        }
                    )

                    semantic_model[
                        "object_count"
                    ] = len(
                        semantic_model[
                            "semantic_objects"
                        ]
                    )

            except Exception as ai_error:

                # -----------------------------------------
                # 4F. AI / memory error
                # -----------------------------------------

                ai_interpretations.append(
                    {
                        "source": "qwen",
                        "input_line": candidate[
                            "line"
                        ],
                        "input_text": candidate[
                            "text"
                        ],
                        "error": str(
                            ai_error
                        ),
                    }
                )

        # -------------------------------------------------
        # 5. Update final semantic object count
        #
        # IMPORTANT:
        # Do NOT loop through ai_interpretations here.
        #
        # Qwen results were already added to the semantic
        # model in Section 4E.
        #
        # Human-approved mappings were already added in
        # Section 4A.
        # -------------------------------------------------

        semantic_model[
            "object_count"
        ] = len(
            semantic_model[
                "semantic_objects"
            ]
        )

        # -------------------------------------------------
        # 6. Deterministic compliance assessment
        # -------------------------------------------------

        assessment = assess_semantic_model(
            semantic_model,
            framework,
        )

        # -------------------------------------------------
        # 7. Final response
        # -------------------------------------------------

        return {
            "success": True,

            "filename": file.filename,

            "line_count": line_count,

            "vendor_detection": vendor_info,

            "parser": semantic_model.get(
                "parser"
            ),

            "semantic_model": semantic_model,

            "ai_interpretations": (
                ai_interpretations
            ),

            "compliance": assessment,

            "message": (
                "Configuration security "
                "assessment completed successfully "
                "using deterministic parsing with "
                "AI fallback."
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Configuration analysis failed: "
                f"{str(exc)}"
            ),
        )