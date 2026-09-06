from pathlib import Path


ALLOWED_EXTENSIONS = {".conf", ".cfg", ".txt"}


def validate_config_file(filename: str) -> None:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported configuration file. "
            "Only .conf, .cfg and .txt files are supported."
        )


async def read_config_file(file) -> str:
    validate_config_file(file.filename)

    content = await file.read()

    if not content:
        raise ValueError("Configuration file is empty.")

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")