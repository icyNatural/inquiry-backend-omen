from pathlib import Path, PureWindowsPath


def resolve_note_path(root: Path, relative_path: str) -> Path:
    """Accept only Markdown paths contained in the configured notes directory.

    Windows path syntax is checked on every platform so the same API policy
    applies in development, CI, and the Windows runtime. The notes directory
    must remain writable only by trusted local users (resolution is not an
    operating-system sandbox against concurrent filesystem changes).
    """
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\x00" in relative_path
        or "\\" in relative_path
        or ":" in relative_path
    ):
        raise ValueError("Invalid note path")

    supplied = Path(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        supplied.is_absolute()
        or windows.drive
        or windows.root
        or ".." in supplied.parts
        or supplied.suffix.lower() != ".md"
    ):
        raise ValueError("Invalid note path")

    try:
        base = root.resolve()
        candidate = (base / supplied).resolve()
        candidate.relative_to(base)
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError("Invalid note path") from error
    if candidate == base:
        raise ValueError("Invalid note path")
    return candidate
