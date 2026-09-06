from app.core.supabase import supabase


def find_approved_mapping(
    vendor: str,
    platform: str,
    raw_pattern: str,
):
    """
    Look for a previously human-approved semantic mapping.
    """

    result = (
        supabase
        .table("semantic_mappings")
        .select("*")
        .eq("vendor", vendor)
        .eq("platform", platform)
        .eq("raw_pattern", raw_pattern.strip())
        .eq("status", "APPROVED")
        .limit(1)
        .execute()
    )

    if result.data:
        return result.data[0]

    return None