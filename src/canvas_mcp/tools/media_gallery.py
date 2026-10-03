"""Read-only Kaltura Media Gallery tools launched through Canvas."""

from __future__ import annotations

import json

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..core.untrusted_content import fence_untrusted, fence_untrusted_inline
from ..core.validation import validate_params
from ..integrations.kaltura import (
    KalturaIntegrationError,
    open_media_gallery,
    vtt_to_text,
)


def register_media_gallery_tools(mcp: FastMCP) -> None:
    """Register Canvas-authenticated Kaltura Media Gallery read tools."""

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @validate_params
    async def list_course_media(
        course_id: int,
        external_tool_id: int = 5826,
        max_entries: int = 100,
    ) -> str:
        """List recordings in a course's Kaltura Media Gallery.

        Uses Canvas's sessionless LTI launch with the caller's existing Canvas
        API credential; no Kaltura password, admin secret, browser cookie, or KS
        needs to be supplied. ``external_tool_id`` defaults to UCSD's Kaltura
        1.3 tool (5826). Returns Kaltura entry IDs suitable for
        ``get_media_transcript``.

        Args:
            course_id: Canvas course ID.
            external_tool_id: Canvas external-tool ID for Kaltura Media Gallery.
            max_entries: Maximum recordings to return (1-500).
        """
        try:
            session = await open_media_gallery(
                course_id, external_tool_id=external_tool_id
            )
            async with session:
                entries, total_count = await session.list_entries(
                    max_entries=max_entries
                )
                payload = {
                    "courseId": course_id,
                    "externalToolId": external_tool_id,
                    "galleryId": session.gallery_id,
                    "count": len(entries),
                    "totalCount": total_count,
                    "truncated": total_count is not None and len(entries) < total_count,
                    "entries": [
                        {
                            "entryId": entry.entry_id,
                            "title": fence_untrusted_inline(
                                entry.title, "Kaltura Media Gallery title"
                            ),
                            "duration": entry.duration,
                        }
                        for entry in entries
                    ],
                }
                return json.dumps(payload, indent=2)
        except KalturaIntegrationError as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @validate_params
    async def get_media_transcript(
        course_id: int,
        entry_id: str,
        external_tool_id: int = 5826,
        language_code: str = "en",
        include_timestamps: bool = False,
    ) -> str:
        """Fetch a Kaltura Media Gallery transcript for one recording.

        Authenticates to Kaltura through Canvas LTI, extracts the media page's
        playback KS in memory, discovers the caption asset, and downloads the
        transcript. By default returns readable text; set ``include_timestamps``
        to return WebVTT cues. The temporary KS is never returned.

        Args:
            course_id: Canvas course ID.
            entry_id: Kaltura media entry ID from ``list_course_media``.
            external_tool_id: Canvas external-tool ID for Kaltura Media Gallery.
            language_code: Preferred caption language code (default ``en``).
            include_timestamps: Return WebVTT instead of plain transcript text.
        """
        try:
            session = await open_media_gallery(
                course_id, external_tool_id=external_tool_id
            )
            async with session:
                transcript = await session.get_transcript(
                    entry_id, language_code=language_code
                )
                body = (
                    transcript.vtt
                    if include_timestamps
                    else vtt_to_text(transcript.vtt)
                )
                if not body.strip():
                    return json.dumps(
                        {"error": "Kaltura returned an empty caption transcript."}
                    )
                title = fence_untrusted_inline(transcript.title, "Kaltura media title")
                label = fence_untrusted_inline(
                    transcript.caption.label or "",
                    "Kaltura caption label",
                )
                fenced_body = fence_untrusted(
                    body,
                    "Kaltura Media Gallery transcript",
                )
                lines = [
                    f"Entry ID: {transcript.entry_id}",
                    f"Gallery ID: {transcript.gallery_id}",
                    f"Title: {title}",
                    f"Caption asset ID: {transcript.caption.asset_id}",
                    f"Caption language: {transcript.caption.language or transcript.caption.language_code or 'unknown'}",
                    f"Caption label: {label}",
                    f"Segment fallback used: {'yes' if transcript.used_segment_fallback else 'no'}",
                    "",
                    fenced_body,
                ]
                return "\n".join(lines)
        except KalturaIntegrationError as exc:
            return json.dumps({"error": str(exc)})
