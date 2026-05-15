"""Artifact persistence helpers for generated media."""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import detect_image_mime, ensure_dir

_DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", re.DOTALL)
_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_GENERATE_IMAGE_TOOL_NAME = "generate_image"
_AUTOSEND_ATTACHMENT_EXTENSIONS = frozenset({
    ".csv",
    ".doc",
    ".docx",
    ".pdf",
    ".ppt",
    ".pptx",
    ".txt",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".zip",
})
_AUTOSEND_ATTACHMENT_RE = re.compile(
    r"(?P<path>(?:~|/)[^\n\r<>\"'`]*?\."
    r"(?:xlsm|xlsx|xls|docx|doc|pptx|ppt|csv|pdf|txt|zip))"
    r"(?=$|[\s<>\")\]}'`*,，。;；:：])",
    re.IGNORECASE,
)
_WRITE_TOOL_SUCCESS_RE = re.compile(
    r"Successfully (?:wrote \d+ characters to|created|edited) (?P<path>[^\n\r]+)",
    re.IGNORECASE,
)
_FILENAME_ATTACHMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{2,}\."
    r"(?:xlsm|xlsx|xls|docx|doc|pptx|ppt|csv|pdf|txt|zip))",
    re.IGNORECASE,
)
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,}")
_ATTACHMENT_INTENT_RE = re.compile(
    r"\b(?:excel|xlsx|xls|spreadsheet|file|document|pdf|csv|attachment|send)\b",
    re.IGNORECASE,
)


class ArtifactError(ValueError):
    """Raised when an artifact cannot be safely decoded or stored."""


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode a base64 image data URL and return ``(bytes, mime)``."""
    match = _DATA_IMAGE_RE.match(data_url.strip())
    if match is None:
        raise ArtifactError("expected a base64 image data URL")

    declared_mime, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ArtifactError("invalid base64 image payload") from exc

    detected_mime = detect_image_mime(raw)
    if detected_mime is None:
        raise ArtifactError("unsupported or unrecognized image data")
    if declared_mime != detected_mime:
        declared_mime = detected_mime
    return raw, declared_mime


def _safe_relative_dir(save_dir: str) -> Path:
    normalized = save_dir.replace("\\", "/").strip("/")
    if not normalized:
        raise ArtifactError("save_dir must not be empty")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ArtifactError("save_dir must be a safe relative path")
    return Path(*rel.parts)


def _artifact_root(save_dir: str) -> Path:
    media_root = get_media_dir().resolve()
    root = (media_root / _safe_relative_dir(save_dir)).resolve()
    try:
        root.relative_to(media_root)
    except ValueError as exc:
        raise ArtifactError("artifact directory escapes media root") from exc
    return root


def store_generated_image_artifact(
    data_url: str,
    *,
    prompt: str,
    model: str,
    source_images: list[str] | None = None,
    save_dir: str = "generated",
    provider: str = "openrouter",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist a generated image and sidecar metadata under the media root."""
    raw, mime = decode_image_data_url(data_url)
    ext = _MIME_EXTENSIONS.get(mime)
    if ext is None:
        raise ArtifactError(f"unsupported image MIME type: {mime}")

    now = created_at or datetime.now().astimezone()
    day_dir = ensure_dir(_artifact_root(save_dir) / now.strftime("%Y-%m-%d"))
    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    image_path = day_dir / f"{artifact_id}{ext}"
    metadata_path = day_dir / f"{artifact_id}.json"

    image_path.write_bytes(raw)
    metadata: dict[str, Any] = {
        "id": artifact_id,
        "path": str(image_path),
        "mime": mime,
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "source_images": list(source_images or []),
        "created_at": now.isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def generated_image_tool_result(artifacts: list[dict[str, Any]]) -> str:
    """Return the compact structured result exposed to the LLM."""
    return json.dumps(
        {
            "artifacts": artifacts,
            "next_step": (
                "Use these artifact paths as reference_images for follow-up edits. "
                "For the current chat, reply naturally; the runtime attaches generated images automatically. "
                "Do not call message just to announce or resend them. Keep raw paths internal unless the user asks for debug details."
            ),
        },
        ensure_ascii=False,
    )


def _extract_text_payload(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts) if parts else None
    return None


def generated_image_paths_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Collect generated image artifact paths from generate_image tool results."""
    paths: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != _GENERATE_IMAGE_TOOL_NAME:
            continue
        payload = _extract_text_payload(message.get("content"))
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        artifacts = data.get("artifacts") if isinstance(data, dict) else None
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if isinstance(path, str) and path and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _valid_local_attachment_path(path: str, *, workspace: str | Path) -> str | None:
    candidate = path.strip().strip("*")
    if not candidate:
        return None

    file_path = Path(candidate).expanduser()
    if not file_path.is_absolute():
        file_path = Path(workspace).expanduser() / file_path
    resolved = file_path.resolve(strict=False)

    if resolved.suffix.lower() not in _AUTOSEND_ATTACHMENT_EXTENSIONS:
        return None
    if not resolved.is_file():
        return None

    # HTTP/API channel uploads (e.g. WhatsApp PDFs saved under media/api/) are inbound
    # conversation context — never treat them as outbound "deliverables" for link/media APIs.
    try:
        api_inbox = get_media_dir("api").resolve(strict=False)
        if api_inbox.exists() and (_is_under(resolved, api_inbox) or resolved == api_inbox):
            return None
    except OSError:
        pass

    workspace_root = Path(workspace).expanduser().resolve(strict=False)
    media_root = get_media_dir().resolve(strict=False)
    if not (
        _is_under(resolved, workspace_root)
        or resolved == workspace_root
        or _is_under(resolved, media_root)
        or resolved == media_root
    ):
        return None

    return str(resolved)


def _iter_attachment_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in _AUTOSEND_ATTACHMENT_RE.finditer(text):
        refs.append(match.group("path"))
    for match in _WRITE_TOOL_SUCCESS_RE.finditer(text):
        refs.append(match.group("path"))
    for match in _FILENAME_ATTACHMENT_RE.finditer(text):
        refs.append(match.group("name"))
    return refs


def _attachment_roots(workspace: str | Path) -> list[Path]:
    roots: list[Path] = []
    for root in (Path(workspace).expanduser(), get_media_dir()):
        resolved = root.resolve(strict=False)
        if resolved not in roots and resolved.exists():
            roots.append(resolved)
    return roots


def _searchable_tokens(text: str) -> list[str]:
    if not _ATTACHMENT_INTENT_RE.search(text):
        return []

    tokens: list[str] = []
    seen: set[str] = set()
    for match in _SEARCH_TOKEN_RE.finditer(text):
        token = match.group(0).strip("._-")
        lowered = token.lower()
        if lowered in {
            "excel", "xlsx", "xls", "file", "document", "send", "please",
            "quotation", "download", "workspace", "home", "nanobot",
        }:
            continue
        # Avoid broad natural-language tokens; quote numbers/codes are useful.
        if not (any(ch.isdigit() for ch in token) or "-" in token or "_" in token):
            continue
        if lowered not in seen:
            tokens.append(token)
            seen.add(lowered)
    return tokens


def discover_attachment_paths_from_text(
    text: str,
    *,
    workspace: str | Path,
    max_results: int = 5,
) -> list[str]:
    """Find existing local attachments referenced by a user/assistant turn."""
    paths: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path or path in seen:
            return
        paths.append(path)
        seen.add(path)

    for ref in _iter_attachment_refs(text):
        add(_valid_local_attachment_path(ref, workspace=workspace))

    if len(paths) >= max_results:
        return paths[:max_results]

    tokens = _searchable_tokens(text)
    if not tokens:
        return paths

    candidates: list[Path] = []
    lowered_tokens = [token.lower() for token in tokens]
    for root in _attachment_roots(workspace):
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in _AUTOSEND_ATTACHMENT_EXTENSIONS:
                continue
            name = candidate.name.lower()
            stem = candidate.stem.lower()
            if any(token in name or token in stem for token in lowered_tokens):
                candidates.append(candidate)

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        add(_valid_local_attachment_path(str(candidate), workspace=workspace))
        if len(paths) >= max_results:
            break

    return paths


def generated_attachment_paths_from_messages(
    messages: list[dict[str, Any]],
    *,
    workspace: str | Path,
    final_content: str | None = None,
) -> list[str]:
    """Collect local files that should be attached to the final channel reply.

    Generated images are stored as structured tool artifacts. Other files, like
    Excel quotes or PDFs, are commonly surfaced by tools or the final response as
    local paths; attach those only when they exist under the workspace/media roots.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path or path in seen:
            return
        paths.append(path)
        seen.add(path)

    for image_path in generated_image_paths_from_messages(messages):
        add(_valid_local_attachment_path(image_path, workspace=workspace) or image_path)

    texts: list[str] = []
    if final_content:
        texts.append(final_content)

    for message in messages:
        role = message.get("role")
        name = message.get("name")
        if role == "assistant":
            payload = _extract_text_payload(message.get("content"))
        elif role == "tool" and name in {"write_file", "edit_file"}:
            payload = _extract_text_payload(message.get("content"))
        else:
            payload = None
        if payload:
            texts.append(payload)

    for text in texts:
        for path in discover_attachment_paths_from_text(text, workspace=workspace):
            add(path)

    return paths
