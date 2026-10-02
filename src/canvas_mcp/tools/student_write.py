"""Tier 1 student write tools (#170).

These are the first tools that let an agent act *on* Canvas on a student's
behalf rather than only read. Four properties are load-bearing:

1. **No identity override on the wire.** The submit endpoint
   (``POST /courses/:id/assignments/:id/submissions``) is not structurally
   self-scoped: Canvas accepts ``submission[user_id]`` there when the token
   carries grading permission, and a real person can hold mixed student and TA
   enrollments. So rather than trusting the tool profile, every outbound write
   body is checked against an identity-override denylist immediately before it
   is sent (``assert_no_identity_override``).
2. **Operator ceiling.** A tool absent from ``STUDENT_WRITE_TOOLS`` is never
   registered, so it never enters the MCP tool list. The default is empty.
3. **Instructor agency for course-scoped writes.** Within that ceiling, a
   per-course policy can further restrict writes, and it is re-checked immediately
   before the write itself, not merely during the preview. The personal-calendar
   writer is self-scoped instead and never targets a course calendar. See
   ``core/course_policy.py``.
4. **Confirmation bound to content.** ``submit_assignment`` will not submit on
   a bare boolean. The preview issues a short-lived, single-use token bound to
   the target, the payload hash and the observed attempt number, so an agent
   cannot submit without first surfacing a preview, and cannot submit something
   other than what was previewed.

Group assignments are refused in Tier 1: a submission to a group assignment
becomes the whole group's submission, affecting students who never consented,
and those shared-attempt semantics deserve their own decision.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from typing import Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..core.cache import get_course_id
from ..core.client import (
    fetch_all_paginated_results,
    make_canvas_request,
    upload_file_to_storage,
)
from ..core.config import get_config
from ..core.course_policy import (
    assert_no_identity_override,
    check_student_write_allowed,
)
from ..core.credentials import is_http_request_active
from ..core.dates import format_date
from ..core.file_validation import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    detect_mime_type,
    sanitize_filename,
)
from ..core.untrusted_content import (
    FENCE_LEAK_ERROR,
    contains_fence_markers,
    fence_untrusted,
    fence_untrusted_inline,
)
from ..core.validation import coerce_canvas_id, validate_params
from ..core.write_confirmation import ConfirmationGuard, unconfirmed_write_warning
from ..core.write_outcome import RequestFailure, WriteOutcome

# Submission types this tool supports. Quiz and discussion types are absent by
# design: quiz-taking is a separate academic-integrity decision behind its own
# flag, and discussion participation already has dedicated tools.
_SUPPORTED_TYPES = ("online_text_entry", "online_url", "online_upload")

# These tools are self-scoped by a hard-coded "/submissions/self" path suffix. That
# only holds while assignment_id cannot end the path early, so a non-numeric ID is
# refused outright rather than passed to Canvas.
_INVALID_ASSIGNMENT_ID = (
    "Error: assignment_id must be a numeric Canvas assignment ID. "
    "Use list_assignments to find it."
)

_INVALID_CALENDAR_EVENT_ID = (
    "Error: event_id must be a numeric Canvas calendar event ID. "
    "Use list_my_calendar_events to find it."
)


def _parse_event_datetime(value: str, field_name: str) -> tuple[str | None, datetime | None, str | None]:
    """Validate an offset-aware ISO-8601 event timestamp.

    Canvas accepts ISO-8601 datetimes. Requiring an explicit UTC offset avoids
    silently interpreting a school deadline in the server's local timezone.
    """
    raw = value.strip()
    if not raw:
        return None, None, f"Error: {field_name} cannot be empty"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return (
            None,
            None,
            f"Error: {field_name} must be an ISO-8601 datetime, for example "
            "2026-10-02T16:00:00-07:00",
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return (
            None,
            None,
            f"Error: {field_name} must include a timezone offset, for example "
            "2026-10-02T16:00:00-07:00",
        )
    return parsed.isoformat(), parsed, None


def _same_event_instant(left: Any, right: datetime) -> bool:
    """Compare Canvas-returned ISO timestamps by absolute instant."""
    if not isinstance(left, str):
        return False
    try:
        parsed = datetime.fromisoformat(left.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return False
    return parsed.timestamp() == right.timestamp()


def _parse_calendar_date(
    value: str, field_name: str
) -> tuple[str | None, date | None, str | None]:
    """Validate a YYYY-MM-DD date used for calendar listing windows."""
    raw = value.strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        return None, None, f"Error: {field_name} must be YYYY-MM-DD"
    return parsed.isoformat(), parsed, None


async def _personal_calendar_context_code() -> tuple[str | None, str | None]:
    """Return the authenticated user's personal Canvas calendar context code."""
    profile = await make_canvas_request("get", "/users/self/profile")
    if not isinstance(profile, dict) or "error" in profile:
        detail = profile.get("error") if isinstance(profile, dict) else profile
        return None, f"Could not read your Canvas profile: {detail}"
    raw_user_id = profile.get("id")
    if not isinstance(raw_user_id, (str, int)):
        return None, "Canvas returned no usable user ID"
    user_id = coerce_canvas_id(raw_user_id)
    if user_id is None:
        return None, "Canvas returned no usable user ID"
    return f"user_{user_id}", None


async def _get_my_calendar_event_record(
    event_id: str | int,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Fetch one event and prove it belongs to the authenticated user's calendar."""
    validated_event_id = coerce_canvas_id(event_id)
    if validated_event_id is None:
        return None, None, _INVALID_CALENDAR_EVENT_ID

    context_code, context_error = await _personal_calendar_context_code()
    if context_error:
        return None, None, f"❌ {context_error}."
    assert context_code is not None

    event = await make_canvas_request("get", f"/calendar_events/{validated_event_id}")
    if not isinstance(event, dict) or "error" in event:
        detail = event.get("error") if isinstance(event, dict) else event
        return (
            context_code,
            None,
            f"❌ Could not read calendar event {validated_event_id}: {detail}",
        )
    if event.get("context_code") != context_code:
        return (
            context_code,
            None,
            "❌ Refusing that calendar event because it is not on your personal "
            "Canvas calendar.",
        )
    return context_code, event, None


def _format_personal_calendar_event(
    event: dict[str, Any], *, include_description: bool
) -> str:
    """Format one personal event while fencing all user-authored text."""
    title = fence_untrusted_inline(
        str(event.get("title") or "Untitled"), "calendar event title"
    )
    lines = [
        f"Event ID: {event.get('id', 'N/A')}",
        f"Title: {title}",
        f"Start: {event.get('start_at') or event.get('all_day_date') or 'N/A'}",
        f"End: {event.get('end_at') or 'N/A'}",
        f"All day: {bool(event.get('all_day'))}",
        f"State: {event.get('workflow_state', 'N/A')}",
    ]
    if event.get("location_name"):
        lines.append(
            "Location: "
            + fence_untrusted_inline(
                str(event["location_name"]), "calendar event location"
            )
        )
    if event.get("location_address"):
        lines.append(
            "Address: "
            + fence_untrusted_inline(
                str(event["location_address"]), "calendar event address"
            )
        )
    if include_description and event.get("description"):
        lines.append(
            "Description:\n"
            + fence_untrusted(
                str(event["description"]), "calendar event description"
            )
        )
    return "\n".join(lines)

# Whole-request upload bounds. These exist on top of the per-file limit in
# core/file_validation, which on its own would allow an unlimited number of
# maximum-size files in a single call. Both are checked before any file content
# is decoded or read.
_MAX_UPLOAD_FILES = 20
_MAX_TOTAL_UPLOAD_BYTES = DEFAULT_MAX_FILE_SIZE_BYTES


def _too_large_message() -> str:
    limit_mb = _MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)
    return (
        f"❌ Those files total more than the {limit_mb} MB allowed for one "
        "submission. Submit fewer or smaller files, or upload them in Canvas."
    )

# How long a confirmation token stays valid. Long enough for a human to read a
# preview and answer, short enough that course state cannot drift far.
_CONFIRM_TTL_SECONDS = 300

# Reuse the shared per-process nonce, mismatch-burn and monotonic token clock.
# A second guard below deduplicates identical payload/attempt fingerprints even
# across fresh preview tokens. Neither guard is shared across workers.
_SUBMISSION_GUARD = ConfirmationGuard(
    ttl_seconds=_CONFIRM_TTL_SECONDS, nothing_done="Nothing was submitted."
)
_redeemed: dict[str, float] = {}  # fingerprint -> monotonic retention deadline
_active: set[str] = set()  # claims cannot expire while their owner is awaiting I/O


def reset_pending_confirmations() -> None:
    """Discard confirmation state (used by tests with no active submissions)."""
    _SUBMISSION_GUARD.reset()
    _redeemed.clear()
    _active.clear()


def _purge_redeemed() -> None:
    now = time.monotonic()
    for fingerprint in [
        f for f, expiry in _redeemed.items() if expiry < now and f not in _active
    ]:
        _redeemed.pop(fingerprint, None)


def _reserve_confirmation(fingerprint: str, token: str) -> bool:
    """Claim both nonce and fingerprint, without yielding to the event loop.

    Caller must check the content binding first. A live fingerprint owner is
    retained even beyond the preview TTL, so a fresh token cannot overlap it.
    """
    _purge_redeemed()
    if fingerprint in _redeemed or not _SUBMISSION_GUARD.reserve(token):
        return False
    _redeemed[fingerprint] = time.monotonic() + _CONFIRM_TTL_SECONDS
    _active.add(fingerprint)
    return True


def _release_confirmation(fingerprint: str, token: str) -> None:
    """Owner-only: no submission was attempted. A nonce burn remains terminal."""
    _active.discard(fingerprint)
    _redeemed.pop(fingerprint, None)
    _SUBMISSION_GUARD.release(token)


def _finish_confirmation(fingerprint: str) -> None:
    """Retire the active owner; retain uncertain/completed attempts for one TTL."""
    _active.discard(fingerprint)
    if fingerprint in _redeemed:
        _redeemed[fingerprint] = time.monotonic() + _CONFIRM_TTL_SECONDS


def _issue_token(fingerprint: str, now: float | None = None) -> str:
    return _SUBMISSION_GUARD.issue(fingerprint, now=now)


def _check_token(token: str, fingerprint: str) -> str | None:
    error = _SUBMISSION_GUARD.check(token, fingerprint)
    if error:
        return error
    _purge_redeemed()
    if fingerprint in _redeemed:
        return (
            "❌ That confirmation was already used. Nothing was submitted. "
            "Run the preview again."
        )
    return None


def _caller_identity() -> str:
    return _SUBMISSION_GUARD.caller_identity()


class _PreparedFile:
    """A file staged for upload, with its bytes already resolved.

    Holds bytes rather than a path because the two ingress modes differ: a
    stdio caller names a local file, while an HTTP caller must inline the
    content. Normalizing early keeps the upload path identical for both.
    """

    def __init__(self, name: str, content: bytes, mime_type: str) -> None:
        self.name = name
        self.content = content
        self.mime_type = mime_type

    @property
    def size(self) -> int:
        return len(self.content)


def _fingerprint(
    course_id: str,
    assignment_id: str,
    submission_type: str,
    payload_digest: str,
    attempt: int,
    allowed_attempts: int | None = None,
) -> str:
    """Bind a confirmation to exactly what was previewed, and to who previewed it.

    Including the observed attempt number means a submission that lands between
    preview and confirm invalidates the token rather than silently consuming a
    second attempt.

    Including the caller identity matters on a hosted server, where each request
    carries its own Canvas token: without it, a token issued to one student could
    be redeemed by another whose attempt number happened to match, so the
    confirmation would no longer authorize the account that saw the preview.

    Including the attempt *limit* as well as the count matters because an
    instructor can change ``allowed_attempts`` in between. A preview that said
    "unlimited" could otherwise be confirmed against a freshly capped assignment
    and spend what is now the final attempt, with the student having agreed to
    something different.
    """
    raw = (
        f"{_caller_identity()}|{course_id}|{assignment_id}|"
        f"{submission_type}|{payload_digest}|{attempt}|{allowed_attempts}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _digest_payload(
    body: str | None,
    url: str | None,
    comment: str | None,
    files: list[_PreparedFile],
) -> str:
    """Hash the exact content that would be submitted.

    Every field is length-prefixed rather than concatenated, because plain
    concatenation is ambiguous: a file named ``a.txt`` holding ``XPAYLOAD``
    would hash identically to one named ``a.txtX`` holding ``PAYLOAD``. That
    would let a token approve content other than what was previewed, which is
    precisely the guarantee this digest exists to provide.

    ``comment`` is covered too. The preview displays it, so a confirmation that
    did not commit to it could swap in text the student never saw before it
    reached their instructor.
    """
    hasher = hashlib.sha256()

    def absorb(chunk: bytes) -> None:
        hasher.update(len(chunk).to_bytes(8, "big"))
        hasher.update(chunk)

    absorb((body or "").encode())
    absorb((url or "").encode())
    absorb((comment or "").encode())
    absorb(len(files).to_bytes(8, "big"))
    for prepared in files:
        absorb(prepared.name.encode())
        absorb(prepared.content)
    return hasher.hexdigest()




def _describe_attempts(assignment: dict, submission: dict) -> str:
    """Render attempt usage for the preview.

    Canvas encodes "unlimited" as ``allowed_attempts = -1`` (and often omits the
    field), which is the detail worth stating plainly: the student needs to know
    whether proceeding spends a scarce resource.
    """
    allowed = assignment.get("allowed_attempts")
    used = submission.get("attempt") or 0

    if allowed is None or allowed == -1:
        return f"Attempts: {used} used, unlimited allowed"

    remaining = allowed - used
    warning = "  ⚠️  This is your LAST attempt." if remaining <= 1 else ""
    return f"Attempts: {used} of {allowed} used, {remaining} remaining.{warning}"


def _decoded_size(encoded: str) -> int:
    """How many bytes a base64 string will decode to, computed without decoding.

    Used to reject an oversized upload before allocating it. The naive
    ``len(encoded) // 4 * 3`` overshoots by the number of padding characters, so
    a file whose decoded size is exactly the documented limit would be refused;
    subtracting the trailing '=' makes this exact for well-formed input.
    """
    padding = len(encoded) - len(encoded.rstrip("="))
    return max(0, len(encoded) // 4 * 3 - padding)


def _normalize_extensions(raw: list[str] | None) -> frozenset[str] | None:
    """Normalize an assignment's ``allowed_extensions`` into ``{'.pdf', ...}``.

    Canvas stores these without leading dots and with inconsistent case. An
    empty or absent list means the assignment does not restrict types.
    """
    if not raw:
        return None
    return frozenset(
        f".{str(ext).strip().lstrip('.').lower()}" for ext in raw if str(ext).strip()
    )


def _check_submission_name(
    name: str, allowed_extensions: frozenset[str] | None
) -> tuple[str, str | None]:
    """Sanitize a filename and check it against the ASSIGNMENT's own rules.

    Returns ``(safe_name, error)``.

    Deliberately no global extension allowlist. The instructor's
    ``allowed_extensions`` on the assignment is the legitimate statement of what
    that assignment accepts; a separate hard-coded list can only disagree with
    it, and the one previously used here rejected ordinary student work such as
    ``.heic`` photos and ``.tex`` sources. Sanitization is still applied, since
    a malicious *path* is a real attack whereas an unusual extension is not.
    """
    safe_name = sanitize_filename(name)
    if not safe_name or safe_name in (".", ".."):
        return "", f"❌ '{name}' is not a usable filename."

    if allowed_extensions is not None:
        extension = os.path.splitext(safe_name)[1].lower()
        if extension not in allowed_extensions:
            accepted = ", ".join(sorted(allowed_extensions))
            return "", (
                f"❌ This assignment does not accept "
                f"'{extension or 'files without an extension'}'. "
                f"It accepts: {accepted}"
            )
    return safe_name, None


def _prepare_files(
    file_paths: list[str] | None,
    file_contents: list[dict[str, str]] | None,
    allowed_extensions: frozenset[str] | None = None,
) -> tuple[list[_PreparedFile], str | None]:
    """Resolve either ingress mode into raw bytes.

    Returns ``(files, error)``.

    ``file_paths`` reads the *server's* filesystem, which is correct for a
    local stdio server and a serious disclosure hole for a shared HTTP one: a
    remote caller could name any file the server process can read and upload it
    into their own Canvas submission. It is therefore refused outright over HTTP
    transport, where callers must inline content instead.
    """
    prepared: list[_PreparedFile] = []

    if file_paths and is_http_request_active():
        return [], (
            "Error: 'file_paths' reads files from the server and is only "
            "available on a local (stdio) server. On this hosted server, pass "
            "the file with 'file_contents' as base64 instead."
        )

    # Bound the whole request, not just each file. A per-file cap alone lets a
    # caller send an unlimited NUMBER of maximum-size files, and the preview
    # decodes and holds them all before any confirmation is required, so one
    # authenticated request could exhaust a shared server's memory.
    total_files = len(file_paths or []) + len(file_contents or [])
    if total_files > _MAX_UPLOAD_FILES:
        return [], (
            f"❌ Too many files ({total_files}). At most {_MAX_UPLOAD_FILES} may "
            "be submitted at once."
        )

    running_bytes = 0

    for path in file_paths or []:
        if not os.path.isfile(path):
            return [], f"❌ Cannot submit '{path}': no such file."
        try:
            file_size = os.path.getsize(path)
        except OSError as exc:
            return [], f"❌ Could not read '{path}': {exc}"
        if file_size > DEFAULT_MAX_FILE_SIZE_BYTES:
            return [], f"❌ '{path}' exceeds the maximum upload size."
        running_bytes += file_size
        if running_bytes > _MAX_TOTAL_UPLOAD_BYTES:
            return [], _too_large_message()

        safe_name, name_error = _check_submission_name(
            os.path.basename(path), allowed_extensions
        )
        if name_error:
            return [], name_error
        try:
            with open(path, "rb") as handle:
                content = handle.read()
        except OSError as exc:
            return [], f"❌ Could not read '{path}': {exc}"
        prepared.append(
            _PreparedFile(safe_name, content, detect_mime_type(safe_name))
        )

    for entry in file_contents or []:
        name = str(entry.get("name") or "").strip()
        encoded = entry.get("content_base64")
        if not name or not encoded:
            return [], "Error: each file_contents entry needs 'name' and 'content_base64'"

        # Bound sizes BEFORE decoding, so an oversized request is rejected
        # without ever allocating the buffer it describes.
        encoded = encoded.strip()
        approx_size = _decoded_size(encoded)
        if approx_size > DEFAULT_MAX_FILE_SIZE_BYTES:
            return [], f"❌ '{name}' exceeds the maximum upload size."
        running_bytes += approx_size
        if running_bytes > _MAX_TOTAL_UPLOAD_BYTES:
            return [], _too_large_message()

        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return [], f"❌ '{name}' is not valid base64."

        # Validate the name the CLIENT actually supplied. An earlier version
        # checked a temp file's random basename instead, which meant the
        # sanitized result was discarded and a name like "../essay.pdf" reached
        # Canvas untouched.
        safe_name, name_error = _check_submission_name(name, allowed_extensions)
        if name_error:
            return [], name_error
        if len(content) > DEFAULT_MAX_FILE_SIZE_BYTES:
            return [], f"❌ '{name}' exceeds the maximum upload size."

        prepared.append(
            _PreparedFile(safe_name, content, detect_mime_type(safe_name))
        )

    return prepared, None


async def _final_preflight(
    course_id: str,
    assignment_id: str,
    submission_type: str,
    payload_digest: str,
    confirmed_fingerprint: str,
) -> str | None:
    """Re-verify EVERY precondition immediately before the submit call.

    Returns an error message if anything has changed, or None if the write may
    proceed.

    This exists because arbitrary time passes between the earlier checks and the
    POST: the policy read, and for uploads a multi-step round trip per file. Any
    precondition checked earlier can have changed in that window, so all of them
    are re-checked here rather than only the attempt count. Three extra requests
    on a rare, irreversible, attempt-consuming operation is a trade worth making.

    If the state cannot be re-read, that counts as changed. Proceeding would mean
    submitting without the guarantee the student was promised.
    """
    # The instructor may have revoked agent writes while uploads were running,
    # and the earlier grant may have been served from a cache that has since
    # expired. The authoritative check is the last one before the write.
    allowed, reason = await check_student_write_allowed(course_id, "submit_assignment")
    if not allowed:
        return f"❌ Submission blocked. {reason}"

    assignment = await make_canvas_request(
        "get", f"/courses/{course_id}/assignments/{assignment_id}"
    )
    submission = await make_canvas_request(
        "get", f"/courses/{course_id}/assignments/{assignment_id}/submissions/self"
    )
    if (
        not isinstance(assignment, dict)
        or "error" in assignment
        or not isinstance(submission, dict)
        or "error" in submission
    ):
        return (
            "❌ Could not re-check your attempt count just before submitting, so "
            "nothing was submitted. Try again shortly."
        )

    # The assignment may have become a group assignment in the meantime, which
    # the tool refuses outright: submitting would bind classmates who never
    # agreed to it and spend a shared attempt.
    if assignment.get("group_category_id"):
        return (
            "❌ This became a group assignment while the submission was being "
            "prepared, so nothing was submitted. Agent-assisted submission is "
            "not supported for group assignments. Please submit it in Canvas."
        )

    current = _fingerprint(
        course_id,
        assignment_id,
        submission_type,
        payload_digest,
        submission.get("attempt") or 0,
        assignment.get("allowed_attempts"),
    )
    if current != confirmed_fingerprint:
        return (
            "❌ Your submission state changed while this was being prepared "
            "(another submission landed, or the attempt limit changed). Nothing "
            "was submitted. Check get_my_submission, then preview again."
        )
    return None


async def _upload_one(
    course_id: str, assignment_id: str, prepared: _PreparedFile
) -> tuple[str | None, str | None]:
    """Run Canvas's 3-step upload for one file. Returns ``(file_id, error)``.

    Step 1 targets ``/submissions/self/files``, which *is* structurally
    self-scoped: the slot Canvas hands back belongs to the calling user's own
    submission and cannot be redirected at another student.
    """
    slot = await make_canvas_request(
        "post",
        f"/courses/{course_id}/assignments/{assignment_id}/submissions/self/files",
        data={
            "name": prepared.name,
            "size": prepared.size,
            "content_type": prepared.mime_type,
        },
        use_form_data=True,
    )
    if isinstance(slot, dict) and "error" in slot:
        return None, f"❌ Failed to request an upload slot for '{prepared.name}': {slot['error']}"

    upload_url = slot.get("upload_url")
    if not upload_url:
        return None, f"❌ Canvas returned no upload URL for '{prepared.name}'."

    # Step 2 writes the bytes through a temp file, because the storage helper
    # takes a path. The bytes are passed through verbatim: no decoding, no
    # transcoding, no content inspection, no OCR. Whether they are a JPEG, a
    # PDF or a zip is not this server's business.
    handle_fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(prepared.content)
        stored = await upload_file_to_storage(
            upload_url=upload_url,
            upload_params=slot.get("upload_params", {}),
            file_path=temp_path,
            filename=prepared.name,
            content_type=prepared.mime_type,
        )
    finally:
        os.unlink(temp_path)

    if isinstance(stored, dict) and "error" in stored:
        return None, f"❌ Upload failed for '{prepared.name}': {stored['error']}"

    # Canvas storage answers in more than one shape. A 200/201 whose body is
    # empty or non-JSON yields {"success": true} with no id, and a redirect
    # confirmation can nest the file under "attachment". Check each documented
    # shape before concluding the upload produced nothing usable.
    file_id = (
        stored.get("id")
        or (stored.get("attachment") or {}).get("id")
        or (stored.get("file") or {}).get("id")
    )
    if not file_id:
        if stored.get("success"):
            return None, (
                f"❌ '{prepared.name}' uploaded, but Canvas returned no file ID "
                "to attach it with, so the submission was not sent. Check "
                "whether the file appears in Canvas before retrying."
            )
        return None, f"❌ Canvas did not return a file ID for '{prepared.name}'."
    return str(file_id), None


def register_student_write_tools(mcp: FastMCP) -> None:
    """Register Tier 1 student tools.

    ``get_my_submission`` is read-only and always registered. The write tools
    register only when the operator has named them in ``STUDENT_WRITE_TOOLS``,
    so an unlisted tool never becomes visible to an agent at all.
    """
    enabled = get_config().student_write_tools

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @validate_params
    async def get_my_submission(
        course_identifier: str | int,
        assignment_id: str | int,
    ) -> str:
        """Get your own submission for an assignment, including attempts used.

        Args:
            course_identifier: Course code or Canvas ID
            assignment_id: Canvas assignment ID
        """
        validated_assignment_id = coerce_canvas_id(assignment_id)
        if validated_assignment_id is None:
            return _INVALID_ASSIGNMENT_ID
        assignment_id = validated_assignment_id

        course_id = await get_course_id(course_identifier)
        if not course_id:
            return f"Error: Could not find course {course_identifier}"

        submission = await make_canvas_request(
            "get",
            f"/courses/{course_id}/assignments/{assignment_id}/submissions/self",
            params={"include[]": ["submission_comments", "assignment"]},
        )
        if isinstance(submission, dict) and "error" in submission:
            return f"Error fetching submission: {submission['error']}"

        assignment = submission.get("assignment") or {}
        lines = [
            # Assignment name and submission comments are author-controlled
            # (teacher/peer feedback) — fenced (issue 239).
            f"Submission for: {fence_untrusted_inline(assignment.get('name', f'Assignment {assignment_id}'), 'assignment name')}",
            f"Status: {submission.get('workflow_state', 'unsubmitted')}",
        ]

        if submission.get("submitted_at"):
            lines.append(f"Submitted: {format_date(submission['submitted_at'])}")
        else:
            lines.append("Submitted: not yet")

        if assignment.get("due_at"):
            lines.append(f"Due: {format_date(assignment['due_at'])}")
        if assignment.get("lock_at"):
            lines.append(f"Locks: {format_date(assignment['lock_at'])}")

        lines.append(_describe_attempts(assignment, submission))

        if submission.get("grade") is not None:
            lines.append(f"Grade: {submission['grade']}")

        comments = submission.get("submission_comments") or []
        if comments:
            lines.append(f"\nComments ({len(comments)}):")
            for comment in comments:
                author = comment.get("author_name")
                prefix = (
                    f"{fence_untrusted_inline(author, 'comment author')}: "
                    if author else ""
                )
                lines.append(
                    f"• {prefix}"
                    f"{fence_untrusted(comment.get('comment', ''), 'submission comment')}"
                )

        return "\n".join(lines)

    if "submit_assignment" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=False))
        @validate_params
        async def submit_assignment(
            course_identifier: str | int,
            assignment_id: str | int,
            submission_type: str,
            body: str | None = None,
            url: str | None = None,
            file_paths: list[str] | None = None,
            file_contents: list[dict[str, str]] | None = None,
            comment: str | None = None,
            confirmation_token: str | None = None,
        ) -> str:
            """Submit one of YOUR OWN assignments. Consumes an attempt.

            Two-step by design. Call it without a confirmation_token to get a
            preview of exactly what would be sent plus a token; show that preview
            to the student, and only after they approve it call again passing the
            token to actually submit.
            The token expires, is single-use, and is void if the content or the
            attempt count changed since the preview.

            Args:
                course_identifier: Course code or Canvas ID
                assignment_id: Canvas assignment ID
                submission_type: online_text_entry, online_url, or online_upload
                body: HTML/text content for online_text_entry
                url: URL for online_url
                file_paths: Local file paths (local stdio servers only, any file type)
                file_contents: Inline files as [{"name": ..., "content_base64": ...}]
                comment: Optional comment to include with the submission
                confirmation_token: Token from the preview call; omit to preview
            """
            if submission_type not in _SUPPORTED_TYPES:
                return (
                    f"Error: submission_type must be one of "
                    f"{', '.join(_SUPPORTED_TYPES)} (got '{submission_type}')"
                )

            validated_assignment_id = coerce_canvas_id(assignment_id)
            if validated_assignment_id is None:
                return _INVALID_ASSIGNMENT_ID
            assignment_id = validated_assignment_id

            course_id = await get_course_id(course_identifier)
            if not course_id:
                return f"Error: Could not find course {course_identifier}"

            allowed, reason = await check_student_write_allowed(
                course_id, "submit_assignment"
            )
            if not allowed:
                return f"❌ Submission blocked. {reason}"

            # Backstop for issue 239: never publish our provenance markers into
            # a submission body or its comment.
            if contains_fence_markers(body or "") or contains_fence_markers(comment or ""):
                return FENCE_LEAK_ERROR

            if submission_type == "online_text_entry" and not body:
                return "Error: online_text_entry requires 'body'"
            if submission_type == "online_url" and not url:
                return "Error: online_url requires 'url'"
            if submission_type == "online_upload" and not (file_paths or file_contents):
                return "Error: online_upload requires 'file_paths' or 'file_contents'"

            assignment = await make_canvas_request(
                "get", f"/courses/{course_id}/assignments/{assignment_id}"
            )
            if isinstance(assignment, dict) and "error" in assignment:
                return f"Error fetching assignment: {assignment['error']}"

            # A group submission becomes the whole group's submission and
            # consumes a shared attempt, affecting students who never consented.
            # That needs its own decision, so Tier 1 declines rather than guess.
            if assignment.get("group_category_id"):
                return (
                    "❌ This is a group assignment. Agent-assisted submission is "
                    "not supported for group assignments, because it would submit "
                    "on behalf of your whole group. Please submit it in Canvas."
                )

            if submission_type not in (assignment.get("submission_types") or []):
                return (
                    f"❌ This assignment does not accept '{submission_type}'. "
                    f"It accepts: {', '.join(assignment.get('submission_types') or []) or 'nothing'}"
                )

            submission = await make_canvas_request(
                "get",
                f"/courses/{course_id}/assignments/{assignment_id}/submissions/self",
            )
            # Attempt state is not optional context here: it is what the preview
            # reports and what the confirmation commits to. Substituting zero on
            # a failed read would show the student a false attempt count and
            # make the drift check vacuous, so stop instead.
            if not isinstance(submission, dict) or "error" in submission:
                detail = (
                    submission.get("error")
                    if isinstance(submission, dict)
                    else "unexpected response from Canvas"
                )
                return (
                    "❌ Could not read your current submission state, so the "
                    f"attempt count is unknown: {detail}\n"
                    "Nothing was submitted. Try again shortly."
                )
            attempt = submission.get("attempt") or 0

            prepared, prep_error = _prepare_files(
                file_paths,
                file_contents,
                _normalize_extensions(assignment.get("allowed_extensions")),
            )
            if prep_error:
                return prep_error

            digest = _digest_payload(body, url, comment, prepared)
            fingerprint = _fingerprint(
                course_id,
                str(assignment_id),
                submission_type,
                digest,
                attempt,
                assignment.get("allowed_attempts"),
            )

            if not confirmation_token:
                token = _issue_token(fingerprint)

                preview = [
                    "📋 Submission preview — NOTHING has been submitted yet.",
                    "",
                    f"Assignment: {assignment.get('name', assignment_id)}",
                    f"Type: {submission_type}",
                ]
                if assignment.get("due_at"):
                    preview.append(f"Due: {format_date(assignment['due_at'])}")
                if assignment.get("lock_at"):
                    preview.append(f"Locks: {format_date(assignment['lock_at'])}")
                preview.append(_describe_attempts(assignment, submission))
                preview.append("")

                if submission_type == "online_text_entry":
                    # Shown in full, deliberately. The token authorizes the whole
                    # body, so truncating here would ask the student to confirm
                    # text they were never shown — which is exactly the thing
                    # this preview exists to prevent.
                    text = body or ""
                    preview.append(f"Content ({len(text)} chars):\n{text}")
                elif submission_type == "online_url":
                    preview.append(f"URL: {url}")
                else:
                    preview.append("Files:")
                    for item in prepared:
                        preview.append(f"• {item.name} ({item.mime_type}, {item.size} bytes)")
                if comment:
                    preview.append(f"\nComment: {comment}")

                preview.append(
                    "\n➡️  Show this to the student. Only after they approve it, "
                    f"submit by calling again with confirmation_token='{token}' "
                    "and identical content.\n"
                    "This consumes an attempt and cannot be undone."
                )
                return "\n".join(preview)

            token_error = _check_token(confirmation_token, fingerprint)
            if token_error:
                return token_error

            # Claim the confirmation BEFORE any awaited work. File uploads sit
            # between here and the submit call, so two overlapping confirmations
            # could otherwise both pass validation during those uploads and both
            # submit, spending two attempts. Reserving first makes that
            # impossible; every path that ends without submitting releases it
            # again, so a failed upload still does not cost a fresh preview.
            if not _reserve_confirmation(fingerprint, confirmation_token):
                return (
                    "❌ That confirmation was already used. Nothing was "
                    "submitted. Run the preview again."
                )

            try:
                # Re-check policy at the moment of the write, so an instructor's
                # change between preview and confirm takes effect.
                allowed, reason = await check_student_write_allowed(
                    course_id, "submit_assignment"
                )
                if not allowed:
                    _release_confirmation(fingerprint, confirmation_token)
                    return f"❌ Submission blocked. {reason}"

                data: dict[str, Any] = {"submission[submission_type]": submission_type}
                if submission_type == "online_text_entry":
                    data["submission[body]"] = body
                elif submission_type == "online_url":
                    data["submission[url]"] = url
                else:
                    file_ids = []
                    for item in prepared:
                        file_id, upload_error = await _upload_one(
                            course_id, str(assignment_id), item
                        )
                        if upload_error:
                            # Release the claim: nothing was submitted, so the
                            # student can retry without re-previewing.
                            _release_confirmation(fingerprint, confirmation_token)
                            return f"{upload_error}\nNothing was submitted."
                        file_ids.append(file_id)
                    data["submission[file_ids][]"] = file_ids

                if comment:
                    data["comment[text_comment]"] = comment

                assert_no_identity_override(data)

                # Re-verify every precondition immediately before the write. See
                # _final_preflight: arbitrary time has passed since the earlier
                # checks, so policy, group status and attempt state are all rechecked
                # rather than trusted.
                preflight_error = await _final_preflight(
                    course_id, str(assignment_id), submission_type, digest, fingerprint
                )
                if preflight_error:
                    _release_confirmation(fingerprint, confirmation_token)
                    return preflight_error

                # The claim taken above stands from here on. Even if this call
                # errors, the token is not released: Canvas may have accepted the
                # submission and only lost the reply, and a blind retry would spend
                # a second attempt.
                response = await make_canvas_request(
                    "post",
                    f"/courses/{course_id}/assignments/{assignment_id}/submissions",
                    data=data,
                    use_form_data=True,
                )
                if isinstance(response, dict) and "error" in response:
                    return (
                        f"❌ Submission failed: {response['error']}\n"
                        "Check get_my_submission before retrying — if Canvas accepted "
                        "it and only the reply was lost, retrying would spend another "
                        "attempt."
                    )

                lines = ["✅ Submitted.", f"Assignment: {assignment.get('name', assignment_id)}"]
                if response.get("submitted_at"):
                    lines.append(f"Submitted at: {format_date(response['submitted_at'])}")
                if response.get("attempt"):
                    lines.append(f"Attempt: {response['attempt']}")
                if response.get("late"):
                    lines.append("⚠️  Canvas marked this submission LATE.")
                return "\n".join(lines)
            finally:
                _finish_confirmation(fingerprint)

    if "comment_on_my_submission" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False))
        @validate_params
        async def comment_on_my_submission(
            course_identifier: str | int,
            assignment_id: str | int,
            comment: str,
        ) -> str:
            """Add a comment to YOUR OWN submission.

            Args:
                course_identifier: Course code or Canvas ID
                assignment_id: Canvas assignment ID
                comment: The comment text
            """
            if not comment.strip():
                return "Error: comment cannot be empty"

            # Backstop for issue 239: never publish our provenance markers.
            if contains_fence_markers(comment):
                return FENCE_LEAK_ERROR

            validated_assignment_id = coerce_canvas_id(assignment_id)
            if validated_assignment_id is None:
                return _INVALID_ASSIGNMENT_ID
            assignment_id = validated_assignment_id

            course_id = await get_course_id(course_identifier)
            if not course_id:
                return f"Error: Could not find course {course_identifier}"

            allowed, reason = await check_student_write_allowed(
                course_id, "comment_on_my_submission"
            )
            if not allowed:
                return f"❌ Comment blocked. {reason}"

            data = {"comment[text_comment]": comment}
            assert_no_identity_override(data)

            response = await make_canvas_request(
                "put",
                f"/courses/{course_id}/assignments/{assignment_id}/submissions/self",
                data=data,
                use_form_data=True,
            )
            if isinstance(response, dict) and "error" in response:
                return f"❌ Comment failed: {response['error']}"

            return "✅ Comment added to your submission."

    if "mark_module_item_done" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=True))
        @validate_params
        async def mark_module_item_done(
            course_identifier: str | int,
            module_id: str | int,
            item_id: str | int,
        ) -> str:
            """Mark a module item done for YOURSELF.

            Args:
                course_identifier: Course code or Canvas ID
                module_id: Canvas module ID
                item_id: Canvas module item ID
            """
            course_id = await get_course_id(course_identifier)
            if not course_id:
                return f"Error: Could not find course {course_identifier}"

            allowed, reason = await check_student_write_allowed(
                course_id, "mark_module_item_done"
            )
            if not allowed:
                return f"❌ Update blocked. {reason}"

            item_endpoint = (
                f"/courses/{course_id}/modules/{module_id}/items/{item_id}"
            )

            # The /done PUT only has a visible effect on items whose
            # completion requirement is must_mark_done; for anything else
            # Canvas accepts the request and changes nothing (#221), so a
            # bare 200 is not evidence the item was marked.
            item = await make_canvas_request("get", item_endpoint)
            if not isinstance(item, dict) or "error" in item:
                detail = item.get("error") if isinstance(item, dict) else item
                return f"❌ Could not read module item: {detail}"

            requirement = item.get("completion_requirement")
            if not isinstance(requirement, dict) or requirement.get("type") != "must_mark_done":
                have = (
                    f"a '{requirement.get('type')}' completion requirement"
                    if isinstance(requirement, dict)
                    else "no completion requirement"
                )
                return (
                    f"❌ '{item.get('title', item_id)}' cannot be marked done: it has "
                    f"{have}, not 'must_mark_done'. Canvas accepts the request but "
                    "nothing changes. Only items the instructor configured with a "
                    "'Mark as done' requirement support this."
                )

            if requirement.get("completed"):
                return "✅ Module item is already marked done."

            response = await make_canvas_request(
                "put",
                f"{item_endpoint}/done",
            )
            if isinstance(response, dict) and "error" in response:
                return f"❌ Could not mark item done: {response['error']}"

            # Confirm the write actually landed before claiming success.
            after = await make_canvas_request("get", item_endpoint)
            confirmed = (
                isinstance(after, dict)
                and isinstance(after.get("completion_requirement"), dict)
                and after["completion_requirement"].get("completed")
            )
            if not confirmed:
                return unconfirmed_write_warning(
                    "the module item was marked done",
                    {
                        "Item": item.get("title", item_id),
                        "Module": module_id,
                        "Course": course_id,
                    },
                    "Canvas accepted the request but the item still shows as not "
                    "done. Check the module in Canvas and retry.",
                )

            return "✅ Module item marked done."

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @validate_params
    async def list_my_calendar_events(
        start_date: str | None = None,
        end_date: str | None = None,
        all_events: bool = False,
        include_description: bool = False,
        max_events: int = 100,
    ) -> str:
        """List events on YOUR OWN personal Canvas calendar.

        By default Canvas returns events for today. Supply start_date/end_date as
        YYYY-MM-DD for a range, or all_events=true to ignore the date window.
        Course/group calendar events are excluded by a hard-coded personal
        context derived from the authenticated Canvas user.
        """
        if max_events < 1 or max_events > 200:
            return "Error: max_events must be between 1 and 200"

        normalized_start: str | None = None
        normalized_end: str | None = None
        start_obj: date | None = None
        end_obj: date | None = None
        if start_date is not None:
            normalized_start, start_obj, error = _parse_calendar_date(
                start_date, "start_date"
            )
            if error:
                return error
        if end_date is not None:
            normalized_end, end_obj, error = _parse_calendar_date(
                end_date, "end_date"
            )
            if error:
                return error
        if start_obj is not None and end_obj is not None and end_obj < start_obj:
            return "Error: end_date cannot be earlier than start_date"

        context_code, context_error = await _personal_calendar_context_code()
        if context_error:
            return f"❌ {context_error}."
        assert context_code is not None

        params: dict[str, Any] = {
            "type": "event",
            "context_codes[]": [context_code],
        }
        if all_events:
            params["all_events"] = True
        else:
            if normalized_start is not None:
                params["start_date"] = normalized_start
            if normalized_end is not None:
                params["end_date"] = normalized_end

        events = await fetch_all_paginated_results("/calendar_events", params=params)
        if isinstance(events, dict) and "error" in events:
            return f"❌ Could not list your calendar events: {events['error']}"
        if not isinstance(events, list):
            return "❌ Canvas returned an unexpected calendar response."

        personal = [
            event
            for event in events
            if isinstance(event, dict) and event.get("context_code") == context_code
        ]
        if not personal:
            return "No personal Canvas calendar events found for that window."

        personal.sort(
            key=lambda event: str(
                event.get("start_at") or event.get("all_day_date") or ""
            )
        )
        shown = personal[:max_events]
        blocks = [
            _format_personal_calendar_event(
                event, include_description=include_description
            )
            for event in shown
        ]
        header = f"Your personal Canvas calendar events ({len(shown)} shown"
        if len(personal) > len(shown):
            header += f", {len(personal) - len(shown)} more not shown"
        header += "):"
        return header + "\n\n" + "\n\n---\n\n".join(blocks)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @validate_params
    async def get_my_calendar_event(event_id: str | int) -> str:
        """Get one event from YOUR OWN personal Canvas calendar by event ID."""
        _, event, error = await _get_my_calendar_event_record(event_id)
        if error:
            return error
        assert event is not None
        return _format_personal_calendar_event(event, include_description=True)

    if "create_my_calendar_event" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False))
        @validate_params
        async def create_my_calendar_event(
            title: str,
            start_at: str,
            end_at: str | None = None,
            description: str | None = None,
            all_day: bool = False,
            location_name: str | None = None,
            location_address: str | None = None,
            time_zone_edited: str | None = None,
        ) -> str:
            """Create an event on YOUR OWN personal Canvas calendar.

            The calendar context is derived from the authenticated Canvas user;
            callers cannot choose a course, group, account, or another user's
            calendar. Matching title/start events are detected before writing so
            semester-sync workflows can be rerun without duplicating the same
            deadline event.

            Args:
                title: Short event title
                start_at: Offset-aware ISO-8601 start time
                end_at: Optional offset-aware ISO-8601 end time
                description: Optional event description (Canvas accepts HTML)
                all_day: Whether Canvas should display the event as all-day
                location_name: Optional location label
                location_address: Optional street/location address
                time_zone_edited: Optional IANA/Rails timezone name recorded by Canvas
            """
            title = title.strip()
            if not title:
                return "Error: title cannot be empty"
            if len(title) > 255:
                return "Error: title cannot exceed 255 characters"

            text_fields = (
                title,
                description,
                location_name,
                location_address,
                time_zone_edited,
            )
            if any(value and contains_fence_markers(value) for value in text_fields):
                return FENCE_LEAK_ERROR

            normalized_start, start_dt, start_error = _parse_event_datetime(
                start_at, "start_at"
            )
            if start_error:
                return start_error
            assert normalized_start is not None and start_dt is not None

            normalized_end: str | None = None
            end_dt: datetime | None = None
            if all_day and end_at is not None:
                return (
                    "Error: end_at is not supported when all_day=true. "
                    "Create a single-day all-day event without end_at."
                )
            if end_at is not None:
                normalized_end, end_dt, end_error = _parse_event_datetime(
                    end_at, "end_at"
                )
                if end_error:
                    return end_error
                assert end_dt is not None
                if end_dt.timestamp() < start_dt.timestamp():
                    return "Error: end_at cannot be earlier than start_at"

            profile = await make_canvas_request("get", "/users/self/profile")
            if not isinstance(profile, dict) or "error" in profile:
                detail = profile.get("error") if isinstance(profile, dict) else profile
                return f"❌ Could not read your Canvas profile: {detail}"
            raw_user_id = profile.get("id")
            if not isinstance(raw_user_id, (str, int)):
                return "❌ Canvas returned no usable user ID; no calendar event was created."
            user_id = coerce_canvas_id(raw_user_id)
            if user_id is None:
                return "❌ Canvas returned no usable user ID; no calendar event was created."
            context_code = f"user_{user_id}"

            # Make repeat syncs safe. We look only at the caller's own calendar on
            # the relevant date and refuse a matching title/start event. If an end
            # time was supplied for a timed event, it must match too.
            window_end = end_dt or start_dt
            existing = await fetch_all_paginated_results(
                "/calendar_events",
                params={
                    "type": "event",
                    # Canvas may normalize an offset-aware late-night deadline to
                    # the next UTC date. Search one day on either side so that a
                    # repeat still finds the event after that normalization.
                    "start_date": (start_dt - timedelta(days=1)).date().isoformat(),
                    "end_date": (window_end + timedelta(days=1)).date().isoformat(),
                    "context_codes[]": [context_code],
                },
            )
            if isinstance(existing, dict) and "error" in existing:
                return (
                    "❌ Could not check your existing calendar events, so no event "
                    f"was created: {existing['error']}"
                )
            if isinstance(existing, list):
                for event in existing:
                    if not isinstance(event, dict):
                        continue
                    if event.get("context_code") != context_code:
                        continue
                    if event.get("title") != title:
                        continue
                    if all_day:
                        if event.get("all_day") is not True:
                            continue
                        if event.get("all_day_date") != start_dt.date().isoformat():
                            continue
                    elif not _same_event_instant(event.get("start_at"), start_dt):
                        continue
                    if end_dt is not None and not _same_event_instant(
                        event.get("end_at"), end_dt
                    ):
                        continue
                    event_id = event.get("id", "unknown")
                    return (
                        "✅ Matching personal calendar event already exists; nothing "
                        f"was created.\nEvent ID: {event_id}\nStart: {normalized_start}"
                    )

            data: dict[str, Any] = {
                "calendar_event[context_code]": context_code,
                "calendar_event[title]": title,
                "calendar_event[start_at]": normalized_start,
                "calendar_event[all_day]": all_day,
            }
            if normalized_end is not None:
                data["calendar_event[end_at]"] = normalized_end
            if description is not None:
                data["calendar_event[description]"] = description
            if location_name is not None:
                data["calendar_event[location_name]"] = location_name
            if location_address is not None:
                data["calendar_event[location_address]"] = location_address
            if time_zone_edited is not None:
                data["calendar_event[time_zone_edited]"] = time_zone_edited

            response = await make_canvas_request(
                "post", "/calendar_events", data=data, use_form_data=True
            )
            if isinstance(response, RequestFailure):
                if response.outcome is WriteOutcome.MAY_HAVE_WRITTEN:
                    return unconfirmed_write_warning(
                        "the personal calendar event was created",
                        {"Title": title, "Start": normalized_start},
                        "Canvas did not provide a reliable response after the write may "
                        "have reached the server. Check your Canvas calendar before "
                        "retrying to avoid a duplicate.",
                    )
                return f"❌ Calendar event creation failed: {response['error']}"
            if not isinstance(response, dict) or "error" in response:
                detail = response.get("error") if isinstance(response, dict) else response
                return f"❌ Calendar event creation failed: {detail}"

            raw_event_id = response.get("id")
            event_id = (
                coerce_canvas_id(raw_event_id)
                if isinstance(raw_event_id, (str, int))
                else None
            )
            if event_id is None:
                return unconfirmed_write_warning(
                    "the personal calendar event was created",
                    {"Title": title, "Start": normalized_start},
                    "Canvas accepted the request but returned no event ID. Check your "
                    "Canvas calendar before retrying to avoid a duplicate.",
                )

            # Verify the created event is truly on the authenticated user's
            # personal calendar before claiming success.
            after = await make_canvas_request("get", f"/calendar_events/{event_id}")
            confirmed_time = (
                after.get("all_day") is True
                and after.get("all_day_date") == start_dt.date().isoformat()
                if isinstance(after, dict) and all_day
                else isinstance(after, dict)
                and _same_event_instant(after.get("start_at"), start_dt)
            )
            confirmed = (
                isinstance(after, dict)
                and "error" not in after
                and after.get("context_code") == context_code
                and after.get("title") == title
                and confirmed_time
            )
            if not confirmed:
                return unconfirmed_write_warning(
                    "the personal calendar event was created",
                    {"Event ID": event_id, "Title": title, "Start": normalized_start},
                    "Canvas returned an event ID, but the follow-up read did not "
                    "confirm the expected personal-calendar event. Check Canvas before retrying.",
                )

            return (
                "✅ Personal Canvas calendar event created.\n"
                f"Event ID: {event_id}\nTitle: {title}\nStart: {normalized_start}"
            )

    if "update_my_calendar_event" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=True))
        @validate_params
        async def update_my_calendar_event(
            event_id: str | int,
            title: str | None = None,
            start_at: str | None = None,
            end_at: str | None = None,
            description: str | None = None,
            all_day: bool | None = None,
            location_name: str | None = None,
            location_address: str | None = None,
            time_zone_edited: str | None = None,
            clear_end_at: bool = False,
        ) -> str:
            """Update one event on YOUR OWN personal Canvas calendar.

            The event is fetched first and must already belong to the authenticated
            user's personal calendar. The tool never sends context_code, so it
            cannot move an event to a course, group, account, or another user.
            Recurring-series operations are deliberately limited to this one event.
            """
            context_code, current, error = await _get_my_calendar_event_record(event_id)
            if error:
                return error
            assert context_code is not None and current is not None

            validated_event_id = coerce_canvas_id(event_id)
            if validated_event_id is None:
                return _INVALID_CALENDAR_EVENT_ID

            if not any(
                value is not None
                for value in (
                    title,
                    start_at,
                    end_at,
                    description,
                    all_day,
                    location_name,
                    location_address,
                    time_zone_edited,
                )
            ) and not clear_end_at:
                return "Error: provide at least one calendar event field to update"

            text_fields = (
                title,
                description,
                location_name,
                location_address,
                time_zone_edited,
            )
            if any(value and contains_fence_markers(value) for value in text_fields):
                return FENCE_LEAK_ERROR

            data: dict[str, Any] = {"which": "one"}
            if title is not None:
                clean_title = title.strip()
                if not clean_title:
                    return "Error: title cannot be empty"
                if len(clean_title) > 255:
                    return "Error: title cannot exceed 255 characters"
                data["calendar_event[title]"] = clean_title

            normalized_start: str | None = None
            start_dt: datetime | None = None
            if start_at is not None:
                normalized_start, start_dt, start_error = _parse_event_datetime(
                    start_at, "start_at"
                )
                if start_error:
                    return start_error
                assert normalized_start is not None and start_dt is not None
                data["calendar_event[start_at]"] = normalized_start

            effective_all_day = (
                all_day if all_day is not None else bool(current.get("all_day"))
            )
            if all_day is not None:
                data["calendar_event[all_day]"] = all_day

            normalized_end: str | None = None
            end_dt: datetime | None = None
            if effective_all_day and end_at is not None:
                return (
                    "Error: end_at is not supported when the resulting event is "
                    "all-day. Use clear_end_at=true if needed."
                )
            if end_at is not None:
                normalized_end, end_dt, end_error = _parse_event_datetime(
                    end_at, "end_at"
                )
                if end_error:
                    return end_error
                assert normalized_end is not None and end_dt is not None
                data["calendar_event[end_at]"] = normalized_end
            if clear_end_at or all_day is True:
                data["calendar_event[end_at]"] = ""

            if not effective_all_day:
                compare_start = start_dt
                if compare_start is None and isinstance(current.get("start_at"), str):
                    _, compare_start, _ = _parse_event_datetime(
                        str(current["start_at"]), "current start_at"
                    )
                compare_end = end_dt
                if (
                    compare_end is None
                    and not clear_end_at
                    and isinstance(current.get("end_at"), str)
                ):
                    _, compare_end, _ = _parse_event_datetime(
                        str(current["end_at"]), "current end_at"
                    )
                if (
                    compare_start is not None
                    and compare_end is not None
                    and compare_end.timestamp() < compare_start.timestamp()
                ):
                    return "Error: resulting end_at cannot be earlier than start_at"

            if description is not None:
                data["calendar_event[description]"] = description
            if location_name is not None:
                data["calendar_event[location_name]"] = location_name
            if location_address is not None:
                data["calendar_event[location_address]"] = location_address
            if time_zone_edited is not None:
                data["calendar_event[time_zone_edited]"] = time_zone_edited

            response = await make_canvas_request(
                "put",
                f"/calendar_events/{validated_event_id}",
                data=data,
                use_form_data=True,
            )
            uncertain = (
                isinstance(response, RequestFailure)
                and response.outcome is WriteOutcome.MAY_HAVE_WRITTEN
            )
            if isinstance(response, RequestFailure) and not uncertain:
                return f"❌ Calendar event update failed: {response['error']}"
            if (
                not isinstance(response, RequestFailure)
                and (not isinstance(response, dict) or "error" in response)
            ):
                detail = response.get("error") if isinstance(response, dict) else response
                return f"❌ Calendar event update failed: {detail}"

            after_context, after, after_error = await _get_my_calendar_event_record(
                validated_event_id
            )
            if after_error or after_context != context_code or after is None:
                return unconfirmed_write_warning(
                    "the personal calendar event was updated",
                    {"Event ID": validated_event_id},
                    "Canvas may have accepted the update, but the follow-up read "
                    "could not confirm the event. Check Canvas before retrying.",
                )

            checks: list[bool] = []
            if title is not None:
                checks.append(after.get("title") == title.strip())
            if all_day is not None:
                checks.append(bool(after.get("all_day")) is all_day)
            if start_dt is not None:
                if effective_all_day:
                    checks.append(
                        after.get("all_day_date") == start_dt.date().isoformat()
                    )
                else:
                    checks.append(_same_event_instant(after.get("start_at"), start_dt))
            if end_dt is not None:
                checks.append(_same_event_instant(after.get("end_at"), end_dt))
            if clear_end_at or all_day is True:
                checks.append(not after.get("end_at"))
            if description is not None:
                checks.append(str(after.get("description") or "") == description)
            if location_name is not None:
                checks.append(str(after.get("location_name") or "") == location_name)
            if location_address is not None:
                checks.append(
                    str(after.get("location_address") or "") == location_address
                )

            if checks and not all(checks):
                return unconfirmed_write_warning(
                    "the personal calendar event was updated",
                    {"Event ID": validated_event_id},
                    "Canvas returned the event after the update, but one or more "
                    "requested fields did not match. Check Canvas before retrying.",
                )

            suffix = " after an uncertain transport response" if uncertain else ""
            return (
                f"✅ Personal Canvas calendar event updated{suffix}.\n"
                + _format_personal_calendar_event(after, include_description=False)
            )

    if "delete_my_calendar_event" in enabled:

        @mcp.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=True))
        @validate_params
        async def delete_my_calendar_event(
            event_id: str | int,
            cancel_reason: str | None = None,
        ) -> str:
            """Delete one event from YOUR OWN personal Canvas calendar.

            The event is fetched first and must belong to the authenticated user's
            personal calendar. For recurring events, only the named event is
            deleted; the tool never deletes an entire series.
            """
            context_code, current, error = await _get_my_calendar_event_record(event_id)
            if error:
                return error
            assert context_code is not None and current is not None

            validated_event_id = coerce_canvas_id(event_id)
            if validated_event_id is None:
                return _INVALID_CALENDAR_EVENT_ID
            if cancel_reason and contains_fence_markers(cancel_reason):
                return FENCE_LEAK_ERROR

            params: dict[str, Any] = {"which": "one"}
            if cancel_reason is not None:
                params["cancel_reason"] = cancel_reason

            response = await make_canvas_request(
                "delete",
                f"/calendar_events/{validated_event_id}",
                params=params,
            )
            if isinstance(response, RequestFailure):
                if response.outcome is not WriteOutcome.MAY_HAVE_WRITTEN:
                    return f"❌ Calendar event deletion failed: {response['error']}"

                after = await make_canvas_request(
                    "get", f"/calendar_events/{validated_event_id}"
                )
                if (
                    isinstance(after, dict)
                    and "error" in after
                    and str(after["error"]).startswith("HTTP error: 404")
                ):
                    return (
                        "✅ Personal Canvas calendar event deleted after an uncertain "
                        f"transport response.\nEvent ID: {validated_event_id}"
                    )
                return unconfirmed_write_warning(
                    "the personal calendar event was deleted",
                    {"Event ID": validated_event_id},
                    "Canvas may have accepted the deletion, but it could not be "
                    "confirmed. Check Canvas before retrying.",
                )

            if not isinstance(response, dict) or "error" in response:
                detail = response.get("error") if isinstance(response, dict) else response
                return f"❌ Calendar event deletion failed: {detail}"
            if response.get("context_code") not in (None, context_code):
                return unconfirmed_write_warning(
                    "the personal calendar event was deleted",
                    {"Event ID": validated_event_id},
                    "Canvas returned an unexpected calendar context after deletion. "
                    "Check your personal calendar before retrying.",
                )

            title = fence_untrusted_inline(
                str(current.get("title") or "Untitled"), "calendar event title"
            )
            return (
                "✅ Personal Canvas calendar event deleted.\n"
                f"Event ID: {validated_event_id}\nTitle: {title}"
            )
