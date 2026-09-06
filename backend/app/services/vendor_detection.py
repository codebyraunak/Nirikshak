import re


def detect_vendor(config_text: str) -> dict:
    """
    Detect the network device vendor from configuration syntax.

    Supported MVP vendors:
    - Cisco IOS / IOS-XE
    - Juniper Junos / SRX
    - Fortinet FortiGate
    """

    text = config_text.lower()

    scores = {
        "Cisco": 0,
        "Juniper": 0,
        "Fortinet": 0,
    }

    # Cisco indicators
    cisco_patterns = [
        r"^version\s+\d+\.\d+",
        r"^hostname\s+\S+",
        r"^interface\s+(gigabitethernet|fastethernet|ten-gigabitethernet)",
        r"^ip\s+ssh\s+version",
        r"^router\s+(bgp|ospf|eigrp)",
        r"^access-list\s+\d+",
    ]

    # Juniper indicators
    juniper_patterns = [
        r"^set\s+system",
        r"^set\s+interfaces",
        r"^set\s+security",
        r"^set\s+protocols",
        r"^set\s+routing-options",
        r"^set\s+firewall",
    ]

    # Fortinet indicators
    fortinet_patterns = [
        r"^config\s+system\s+global",
        r"^set\s+hostname",
        r"^config\s+firewall\s+policy",
        r"^config\s+system\s+interface",
        r"^config\s+vpn",
        r"^config\s+router",
    ]

    lines = [line.strip() for line in text.splitlines()]

    for line in lines:
        if any(re.search(pattern, line) for pattern in cisco_patterns):
            scores["Cisco"] += 1

        if any(re.search(pattern, line) for pattern in juniper_patterns):
            scores["Juniper"] += 1

        if any(re.search(pattern, line) for pattern in fortinet_patterns):
            scores["Fortinet"] += 1

    vendor = max(scores, key=scores.get)
    highest_score = scores[vendor]

    if highest_score == 0:
        return {
            "vendor": "Unknown",
            "platform": "Unknown",
            "confidence": 0.0,
            "scores": scores,
        }

    confidence = min(0.99, 0.60 + (highest_score * 0.08))

    platform_map = {
        "Cisco": "IOS/IOS-XE",
        "Juniper": "Junos/SRX",
        "Fortinet": "FortiGate",
    }

    return {
        "vendor": vendor,
        "platform": platform_map[vendor],
        "confidence": round(confidence, 2),
        "scores": scores,
    }