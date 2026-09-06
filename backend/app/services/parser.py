import re
from typing import Any


def parse_cisco_config(config_text: str) -> dict[str, Any]:
    """
    Deterministic parser for Cisco IOS / IOS-XE configuration.

    Converts raw Cisco configuration into a vendor-neutral
    Security Semantic Model.

    This parser does NOT decide compliance.
    It only extracts security-relevant configuration.
    """

    lines = config_text.splitlines()

    semantic_objects = []

    current_interface = None
    current_acl = None

    for line_number, raw_line in enumerate(lines, start=1):

        line = raw_line.strip()

        if not line:
            continue

        # ---------------------------------------------------------
        # HOSTNAME
        # ---------------------------------------------------------
        match = re.match(r"^hostname\s+(\S+)", line, re.IGNORECASE)

        if match:
            semantic_objects.append({
                "control_id": "HOSTNAME",
                "category": "device_identity",
                "value": match.group(1),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # SSH VERSION
        # ---------------------------------------------------------
        match = re.match(
            r"^ip\s+ssh\s+version\s+(\d+)",
            line,
            re.IGNORECASE
        )

        if match:
            semantic_objects.append({
                "control_id": "SSH_VERSION",
                "category": "session_security",
                "value": int(match.group(1)),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # SSH TIMEOUT
        # ---------------------------------------------------------
        match = re.match(
            r"^ip\s+ssh\s+timeout\s+(\d+)",
            line,
            re.IGNORECASE
        )

        if match:
            semantic_objects.append({
                "control_id": "SSH_TIMEOUT",
                "category": "session_security",
                "value": int(match.group(1)),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # SSH AUTHENTICATION RETRIES
        # ---------------------------------------------------------
        match = re.match(
            r"^ip\s+ssh\s+authentication-retries\s+(\d+)",
            line,
            re.IGNORECASE
        )

        if match:
            semantic_objects.append({
                "control_id": "SSH_AUTHENTICATION_RETRIES",
                "category": "authentication",
                "value": int(match.group(1)),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # HTTP SERVER
        # ---------------------------------------------------------
        if re.match(
            r"^ip\s+http\s+server$",
            line,
            re.IGNORECASE
        ):
            semantic_objects.append({
                "control_id": "HTTP_MANAGEMENT",
                "category": "management_access",
                "value": True,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # HTTPS SERVER
        # ---------------------------------------------------------
        if re.match(
            r"^ip\s+http\s+secure-server$",
            line,
            re.IGNORECASE
        ):
            semantic_objects.append({
                "control_id": "HTTPS_MANAGEMENT",
                "category": "management_access",
                "value": True,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # NO HTTP SERVER
        # ---------------------------------------------------------
        if re.match(
            r"^no\s+ip\s+http\s+server$",
            line,
            re.IGNORECASE
        ):
            semantic_objects.append({
                "control_id": "HTTP_MANAGEMENT",
                "category": "management_access",
                "value": False,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # NO HTTPS SERVER
        # ---------------------------------------------------------
        if re.match(
            r"^no\s+ip\s+http\s+secure-server$",
            line,
            re.IGNORECASE
        ):
            semantic_objects.append({
                "control_id": "HTTPS_MANAGEMENT",
                "category": "management_access",
                "value": False,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # INTERFACE
        # ---------------------------------------------------------
        match = re.match(
            r"^interface\s+(.+)",
            line,
            re.IGNORECASE
        )

        if match:
            current_interface = match.group(1)

            semantic_objects.append({
                "control_id": "INTERFACE",
                "category": "network_interface",
                "value": current_interface,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # INTERFACE SHUTDOWN
        # ---------------------------------------------------------
        if current_interface and line.lower() == "shutdown":

            semantic_objects.append({
                "control_id": "INTERFACE_SHUTDOWN",
                "category": "network_interface",
                "value": True,
                "context": {
                    "interface": current_interface
                },
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # ACCESS LIST
        # ---------------------------------------------------------
        match = re.match(
            r"^access-list\s+(\d+)\s+(.+)",
            line,
            re.IGNORECASE
        )

        if match:

            acl_number = int(match.group(1))
            acl_rule = match.group(2)

            semantic_objects.append({
                "control_id": "ACCESS_LIST_RULE",
                "category": "network_access_control",
                "value": {
                    "acl": acl_number,
                    "rule": acl_rule
                },
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # LOGGING HOST
        # ---------------------------------------------------------
        match = re.match(
            r"^logging\s+host\s+(\S+)",
            line,
            re.IGNORECASE
        )

        if match:

            semantic_objects.append({
                "control_id": "REMOTE_LOGGING",
                "category": "logging_monitoring",
                "value": match.group(1),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # NTP SERVER
        # ---------------------------------------------------------
        match = re.match(
            r"^ntp\s+server\s+(\S+)",
            line,
            re.IGNORECASE
        )

        if match:

            semantic_objects.append({
                "control_id": "NTP_SERVER",
                "category": "time_synchronization",
                "value": match.group(1),
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

        # ---------------------------------------------------------
        # ENABLE SECRET
        # ---------------------------------------------------------
        if re.match(
            r"^enable\s+secret\s+",
            line,
            re.IGNORECASE
        ):

            semantic_objects.append({
                "control_id": "ENABLE_SECRET",
                "category": "authentication",
                "value": True,
                "evidence": {
                    "line": line_number,
                    "raw": raw_line
                }
            })
            continue

    return {
        "vendor": "Cisco",
        "platform": "IOS/IOS-XE",
        "parser": "deterministic",
        "semantic_objects": semantic_objects,
        "object_count": len(semantic_objects)
    }