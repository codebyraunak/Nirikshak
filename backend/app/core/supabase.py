from supabase import create_client, Client
from app.core.config import settings


def get_supabase_client() -> Client:
    if not settings.SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is missing from environment variables.")

    if not settings.SUPABASE_KEY:
        raise RuntimeError("SUPABASE_KEY is missing from environment variables.")

    if not settings.SUPABASE_URL.startswith("https://"):
        raise RuntimeError(
            "SUPABASE_URL must start with https:// "
            f"(current value starts with: {settings.SUPABASE_URL[:20]!r})"
        )

    return create_client(
        settings.SUPABASE_URL.rstrip("/"),
        settings.SUPABASE_KEY,
    )


supabase: Client = get_supabase_client()