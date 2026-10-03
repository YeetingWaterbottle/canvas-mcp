"""Kaltura Media Gallery access through a Canvas LTI 1.3 launch.

The integration deliberately starts from Canvas's ``sessionless_launch`` API
instead of storing Kaltura credentials. Canvas authenticates the configured
user to the Kaltura LTI tool; the resulting KAF web session is then used to
open media pages and obtain their short-lived playback KS values.

Secrets (Canvas session tokens, LTI id_tokens, cookies, and player KS values)
are kept in memory only and are never included in returned errors.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit

import httpx

from ..core.client import make_canvas_request

_USER_AGENT = "Mozilla/5.0 Canvas-MCP-Kaltura/1.0"
_ENTRY_ID_RE = re.compile(r"^1_[A-Za-z0-9]+$")
_CHANNEL_PATH_RE = re.compile(r"^/channel/[^/]+/(?P<gallery_id>\d+)(?:/|$)")
_JS_LOCATION_RE = re.compile(
    r"(?:window\.)?location\.href\s*=\s*['\"](?P<url>[^'\"]+)['\"]"
)
_ENDLESS_SCROLLER_RE = re.compile(
    r"startEndlessScroller\(\s*['\"]channelGallery['\"]\s*,\s*"
    r"['\"](?P<url>[^'\"]+)['\"]"
)
_JSON_STRING_RE_TEMPLATE = r'"%s"\s*:\s*"((?:\\.|[^"\\])*)"'


class KalturaIntegrationError(RuntimeError):
    """Expected, user-facing failure in the Canvas -> Kaltura read flow."""


@dataclass(frozen=True)
class KalturaMediaEntry:
    entry_id: str
    gallery_id: str
    title: str
    duration: str | None = None


@dataclass(frozen=True)
class KalturaCaptionAsset:
    asset_id: str
    entry_id: str
    language: str | None
    language_code: str | None
    label: str | None
    is_default: bool
    status: int | None
    accuracy: int | None
    file_ext: str | None


@dataclass(frozen=True)
class KalturaTranscript:
    entry_id: str
    gallery_id: str
    title: str
    caption: KalturaCaptionAsset
    vtt: str
    used_segment_fallback: bool


@dataclass
class _ParsedForm:
    action: str
    method: str
    inputs: dict[str, str]


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[_ParsedForm] = []
        self._current: _ParsedForm | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "form":
            self._current = _ParsedForm(
                action=attr.get("action") or "",
                method=(attr.get("method") or "get").lower(),
                inputs={},
            )
            self.forms.append(self._current)
        elif tag == "input" and self._current is not None:
            name = attr.get("name")
            if name:
                self._current.inputs[name] = attr.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._current = None


class _PlayerScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_player_script = False
        self.player_script_parts: list[str] = []
        self._in_title = False
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "script" and attr.get("id") == "playerScript":
            self._in_player_script = True
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_player_script:
            self._in_player_script = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_player_script:
            self.player_script_parts.append(data)
        elif self._in_title:
            self.title_parts.append(data)


class _GalleryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[KalturaMediaEntry] = []
        self.total_count: int | None = None
        self._in_item = False
        self._entry_id: str | None = None
        self._gallery_id: str | None = None
        self._title = ""
        self._link_text: list[str] = []
        self._duration: list[str] = []
        self._item_link_depth = 0
        self._duration_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = set((attr.get("class") or "").split())

        if tag == "li" and "galleryItem" in classes:
            raw_total = attr.get("aria-setsize")
            if raw_total and raw_total.isdigit():
                self.total_count = int(raw_total)
            self._in_item = True
            self._entry_id = None
            self._gallery_id = None
            self._title = ""
            self._link_text = []
            self._duration = []
            return

        if not self._in_item:
            return

        candidate_title = attr.get("title") or ""
        if candidate_title and ("photo-group" in classes or "thumb_wrapper" in classes):
            self._title = candidate_title.strip()

        if tag == "a":
            href = attr.get("href") or ""
            match = re.search(r"/media/t/(1_[A-Za-z0-9]+)/(\d+)(?:[/?#]|$)", href)
            if match:
                self._entry_id = match.group(1)
                self._gallery_id = match.group(2)
                self._item_link_depth += 1

        if "duration" in classes:
            self._duration_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if not self._in_item:
            return
        if tag == "a" and self._item_link_depth:
            self._item_link_depth -= 1
        if self._duration_depth and tag in {"span", "div"}:
            self._duration_depth -= 1
        if tag == "li":
            if self._entry_id and self._gallery_id:
                title = self._title or " ".join(self._link_text).strip()
                duration = " ".join(self._duration).strip()
                self.entries.append(
                    KalturaMediaEntry(
                        entry_id=self._entry_id,
                        gallery_id=self._gallery_id,
                        title=title or self._entry_id,
                        duration=duration or None,
                    )
                )
            self._in_item = False
            self._entry_id = None
            self._gallery_id = None
            self._title = ""
            self._link_text = []
            self._duration = []
            self._item_link_depth = 0
            self._duration_depth = 0

    def handle_data(self, data: str) -> None:
        if not self._in_item:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._item_link_depth:
            self._link_text.append(text)
        if self._duration_depth:
            self._duration.append(text)


def _origin(url: str | httpx.URL) -> str:
    parsed = urlsplit(str(url))
    return f"{parsed.scheme}://{parsed.netloc}"


def _require_https(url: str, *, what: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise KalturaIntegrationError(f"{what} did not return a valid HTTPS URL.")


def _single_post_form(page: str, *, required: set[str], stage: str) -> _ParsedForm:
    parser = _FormParser()
    parser.feed(page)
    candidates = [
        form
        for form in parser.forms
        if form.method == "post" and required.issubset(form.inputs)
    ]
    if len(candidates) != 1:
        raise KalturaIntegrationError(
            f"The Kaltura LTI {stage} page did not contain the expected launch form."
        )
    return candidates[0]


def _json_string_value(script: str, key: str) -> str | None:
    match = re.search(_JSON_STRING_RE_TEMPLATE % re.escape(key), script)
    if not match:
        return None
    try:
        value = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def _parse_player_config(page: str) -> tuple[str, str, str]:
    parser = _PlayerScriptParser()
    parser.feed(page)
    script = "".join(parser.player_script_parts)
    if not script:
        raise KalturaIntegrationError(
            "The Kaltura media page did not contain a player configuration."
        )

    ks = _json_string_value(script, "ks")
    service_url = _json_string_value(script, "serviceUrl")
    if not ks:
        raise KalturaIntegrationError(
            "The Kaltura media page did not provide a playback session."
        )
    if not service_url:
        service_url = "https://www.kaltura.com/api_v3"
    _require_https(service_url, what="Kaltura player")
    host = urlsplit(service_url).hostname or ""
    if host != "kaltura.com" and not host.endswith(".kaltura.com"):
        raise KalturaIntegrationError(
            "The Kaltura player returned an unexpected API host."
        )

    title = " ".join("".join(parser.title_parts).split())
    suffix = " - University of California, San Diego"
    if title.endswith(suffix):
        title = title[: -len(suffix)]
    return ks, service_url.rstrip("/"), title


def _parse_gallery(page: str) -> tuple[list[KalturaMediaEntry], int | None]:
    parser = _GalleryParser()
    parser.feed(page)
    return parser.entries, parser.total_count


def _extract_gallery_id(url: str) -> str:
    match = _CHANNEL_PATH_RE.match(urlsplit(url).path)
    if not match:
        raise KalturaIntegrationError(
            "The Kaltura launch did not resolve to a Media Gallery channel."
        )
    return match.group("gallery_id")


def _extract_ajax_fragment(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, dict):
        return None
    fragment = first.get("content")
    return fragment if isinstance(fragment, str) else None


def _choose_caption(
    captions: list[KalturaCaptionAsset], language_code: str
) -> KalturaCaptionAsset:
    ready = [c for c in captions if c.status in (None, 2)]
    if not ready:
        raise KalturaIntegrationError(
            "No ready caption track is available for this media entry."
        )

    language_code = language_code.strip().lower()
    if language_code:
        matches = [
            c
            for c in ready
            if (c.language_code or "").lower() == language_code
            or (c.language or "").lower() == language_code
        ]
        if not matches and language_code == "en":
            matches = [
                c for c in ready if (c.language or "").lower().startswith("english")
            ]
        if matches:
            return next((c for c in matches if c.is_default), matches[0])

    return next((c for c in ready if c.is_default), ready[0])


def vtt_to_text(vtt: str) -> str:
    """Convert WebVTT cues to readable transcript text."""
    normalized = vtt.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[str] = []
    previous = None
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        cue = " ".join(lines[timing_index + 1 :])
        cue = re.sub(r"<[^>]+>", "", cue)
        cue = html.unescape(" ".join(cue.split()))
        if cue and cue != previous:
            cues.append(cue)
            previous = cue
    return "\n".join(cues)


class KalturaMediaGallerySession:
    """Authenticated KAF session established from the caller's Canvas token."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        course_id: int,
        external_tool_id: int,
        kaf_origin: str,
        channel_url: str,
        channel_html: str,
    ) -> None:
        self.client = client
        self.course_id = course_id
        self.external_tool_id = external_tool_id
        self.kaf_origin = kaf_origin
        self.channel_url = channel_url
        self.channel_html = channel_html
        self.gallery_id = _extract_gallery_id(channel_url)

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> KalturaMediaGallerySession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def list_entries(
        self, max_entries: int = 100
    ) -> tuple[list[KalturaMediaEntry], int | None]:
        if max_entries < 1 or max_entries > 500:
            raise KalturaIntegrationError("max_entries must be between 1 and 500.")

        entries, total = _parse_gallery(self.channel_html)
        unique = {entry.entry_id: entry for entry in entries}
        if len(unique) >= max_entries or (total is not None and len(unique) >= total):
            return list(unique.values())[:max_entries], total

        match = _ENDLESS_SCROLLER_RE.search(self.channel_html)
        if not match:
            return list(unique.values())[:max_entries], total
        ajax_url = urljoin(self.channel_url, html.unescape(match.group("url")))

        page = 2
        while len(unique) < max_entries:
            try:
                response = await self.client.get(
                    ajax_url,
                    params={"format": "ajax", "page": page},
                    headers={
                        "Referer": self.channel_url,
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )
            except httpx.HTTPError as exc:
                raise KalturaIntegrationError(
                    "Kaltura Media Gallery pagination failed."
                ) from exc
            if response.status_code != 200:
                raise KalturaIntegrationError(
                    "Kaltura Media Gallery pagination failed."
                )
            try:
                fragment = _extract_ajax_fragment(response.json())
            except json.JSONDecodeError as exc:
                raise KalturaIntegrationError(
                    "Kaltura Media Gallery returned an invalid pagination response."
                ) from exc
            if not fragment:
                break
            more, fragment_total = _parse_gallery(fragment)
            if fragment_total is not None:
                total = fragment_total
            before = len(unique)
            for entry in more:
                unique.setdefault(entry.entry_id, entry)
            if len(unique) == before:
                break
            if total is not None and len(unique) >= total:
                break
            page += 1

        return list(unique.values())[:max_entries], total

    async def _media_page(self, entry_id: str) -> str:
        if not _ENTRY_ID_RE.fullmatch(entry_id):
            raise KalturaIntegrationError("Invalid Kaltura entry ID.")
        url = f"{self.kaf_origin}/media/t/{entry_id}/{self.gallery_id}"
        try:
            response = await self.client.get(url, headers={"Referer": self.channel_url})
        except httpx.HTTPError as exc:
            raise KalturaIntegrationError(
                "The Kaltura media page could not be loaded."
            ) from exc
        if response.status_code != 200:
            raise KalturaIntegrationError("The Kaltura media page could not be loaded.")
        if entry_id not in response.text:
            raise KalturaIntegrationError(
                "The Kaltura session is not authorized for the requested media entry."
            )
        return response.text

    async def _caption_assets(
        self, entry_id: str, *, ks: str, service_url: str
    ) -> list[KalturaCaptionAsset]:
        endpoint = f"{service_url}/service/caption_captionasset/action/list"
        try:
            response = await self.client.post(
                endpoint,
                data={
                    "ks": ks,
                    "format": "1",
                    "filter:entryIdEqual": entry_id,
                    "pager:pageSize": "50",
                },
            )
        except httpx.HTTPError as exc:
            raise KalturaIntegrationError("Kaltura caption lookup failed.") from exc
        if response.status_code != 200:
            raise KalturaIntegrationError("Kaltura caption lookup failed.")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise KalturaIntegrationError(
                "Kaltura caption lookup returned invalid data."
            ) from exc
        if not isinstance(payload, dict):
            raise KalturaIntegrationError(
                "Kaltura caption lookup returned invalid data."
            )
        if payload.get("objectType") == "KalturaAPIException":
            raise KalturaIntegrationError(
                "The Kaltura playback session could not list captions."
            )

        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list):
            raise KalturaIntegrationError(
                "Kaltura caption lookup returned invalid data."
            )
        captions: list[KalturaCaptionAsset] = []
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            asset_id = obj.get("id")
            if not isinstance(asset_id, str) or not _ENTRY_ID_RE.fullmatch(asset_id):
                continue
            captions.append(
                KalturaCaptionAsset(
                    asset_id=asset_id,
                    entry_id=str(obj.get("entryId") or entry_id),
                    language=(
                        obj.get("language")
                        if isinstance(obj.get("language"), str)
                        else None
                    ),
                    language_code=(
                        obj.get("languageCode")
                        if isinstance(obj.get("languageCode"), str)
                        else None
                    ),
                    label=(
                        obj.get("label") if isinstance(obj.get("label"), str) else None
                    ),
                    is_default=bool(obj.get("isDefault")),
                    status=(
                        obj.get("status")
                        if isinstance(obj.get("status"), int)
                        else None
                    ),
                    accuracy=(
                        obj.get("accuracy")
                        if isinstance(obj.get("accuracy"), int)
                        else None
                    ),
                    file_ext=(
                        obj.get("fileExt")
                        if isinstance(obj.get("fileExt"), str)
                        else None
                    ),
                )
            )
        return captions

    async def _fetch_vtt(self, caption_asset_id: str, ks: str) -> tuple[str, bool]:
        encoded_ks = quote(ks, safe="")
        base = (
            "https://cfvod.kaltura.com/api_v3/index.php/"
            "service/caption_captionasset/action/serveWebVTT/"
            f"captionAssetId/{caption_asset_id}"
        )
        direct_url = f"{base}/segmentDuration/10000/ks/{encoded_ks}/segmentIndex/1.vtt"
        try:
            response = await self.client.get(direct_url)
        except httpx.HTTPError as exc:
            raise KalturaIntegrationError("Kaltura caption download failed.") from exc
        if response.status_code == 200 and response.text.lstrip().startswith("WEBVTT"):
            return response.text, False

        playlist_url = f"{base}/segmentDuration/300/ks/{encoded_ks}/a.m3u8"
        try:
            playlist = await self.client.get(playlist_url)
        except httpx.HTTPError as exc:
            raise KalturaIntegrationError("Kaltura caption download failed.") from exc
        if playlist.status_code != 200:
            raise KalturaIntegrationError("Kaltura caption download failed.")
        segment_urls = [
            urljoin(playlist_url, line.strip())
            for line in playlist.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not segment_urls or len(segment_urls) > 100:
            raise KalturaIntegrationError(
                "Kaltura returned an invalid caption playlist."
            )

        segments: list[str] = []
        for segment_url in segment_urls:
            try:
                segment = await self.client.get(segment_url)
            except httpx.HTTPError as exc:
                raise KalturaIntegrationError(
                    "Kaltura caption segment download failed."
                ) from exc
            if segment.status_code != 200 or not segment.text.lstrip().startswith(
                "WEBVTT"
            ):
                raise KalturaIntegrationError(
                    "Kaltura caption segment download failed."
                )
            segments.append(segment.text)
        return "\n\n".join(segments), True

    async def get_transcript(
        self, entry_id: str, *, language_code: str = "en"
    ) -> KalturaTranscript:
        page = await self._media_page(entry_id)
        ks, service_url, title = _parse_player_config(page)
        captions = await self._caption_assets(entry_id, ks=ks, service_url=service_url)
        caption = _choose_caption(captions, language_code)
        vtt, used_fallback = await self._fetch_vtt(caption.asset_id, ks)
        return KalturaTranscript(
            entry_id=entry_id,
            gallery_id=self.gallery_id,
            title=title or entry_id,
            caption=caption,
            vtt=vtt,
            used_segment_fallback=used_fallback,
        )


async def open_media_gallery(
    course_id: int, *, external_tool_id: int = 5826
) -> KalturaMediaGallerySession:
    """Launch Kaltura through Canvas and return an authenticated KAF session."""
    if course_id <= 0 or external_tool_id <= 0:
        raise KalturaIntegrationError(
            "course_id and external_tool_id must be positive integers."
        )

    launch = await make_canvas_request(
        "GET",
        f"/courses/{course_id}/external_tools/sessionless_launch",
        params={"id": str(external_tool_id), "launch_type": "course_navigation"},
    )
    if not isinstance(launch, dict) or "error" in launch:
        raise KalturaIntegrationError(
            "Canvas could not create a sessionless Kaltura launch."
        )
    launch_url = launch.get("url")
    if not isinstance(launch_url, str):
        raise KalturaIntegrationError("Canvas did not return a Kaltura launch URL.")
    _require_https(launch_url, what="Canvas sessionless launch")
    canvas_origin = _origin(launch_url)
    canvas_host = urlsplit(launch_url).hostname

    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        response = await client.get(launch_url)
        if response.status_code != 200:
            raise KalturaIntegrationError(
                "Canvas could not open the Kaltura launch page."
            )

        oidc_form = _single_post_form(
            response.text,
            required={
                "iss",
                "login_hint",
                "client_id",
                "lti_deployment_id",
                "target_link_uri",
                "lti_message_hint",
            },
            stage="OIDC initialization",
        )
        oidc_action = urljoin(str(response.url), oidc_form.action)
        _require_https(oidc_action, what="Kaltura OIDC initialization")
        kaf_origin = _origin(oidc_action)
        if kaf_origin == canvas_origin:
            raise KalturaIntegrationError(
                "The external-tool launch did not resolve to Kaltura."
            )

        response = await client.post(
            oidc_action,
            data=oidc_form.inputs,
            headers={"Origin": canvas_origin, "Referer": str(response.url)},
        )
        if (
            response.status_code != 200
            or urlsplit(str(response.url)).hostname != canvas_host
        ):
            raise KalturaIntegrationError(
                "Canvas did not complete Kaltura LTI authorization."
            )

        launch_form = _single_post_form(
            response.text,
            required={"id_token", "state"},
            stage="authorization",
        )
        oauth_action = urljoin(str(response.url), launch_form.action)
        _require_https(oauth_action, what="Kaltura OAuth launch")
        if _origin(oauth_action) != kaf_origin:
            raise KalturaIntegrationError(
                "Kaltura authorization returned an unexpected host."
            )

        response = await client.post(
            oauth_action,
            data=launch_form.inputs,
            headers={"Origin": canvas_origin, "Referer": str(response.url)},
        )
        if response.status_code != 200 or _origin(response.url) != kaf_origin:
            raise KalturaIntegrationError(
                "Kaltura did not establish an authenticated KAF session."
            )

        redirect = _JS_LOCATION_RE.search(response.text)
        if redirect:
            channel_url = urljoin(
                str(response.url), html.unescape(redirect.group("url"))
            )
        elif _CHANNEL_PATH_RE.match(urlsplit(str(response.url)).path):
            channel_url = str(response.url)
        else:
            raise KalturaIntegrationError(
                "Kaltura did not redirect to a Media Gallery channel."
            )
        _require_https(channel_url, what="Kaltura Media Gallery")
        if _origin(channel_url) != kaf_origin:
            raise KalturaIntegrationError(
                "Kaltura Media Gallery redirected to an unexpected host."
            )

        channel = await client.get(channel_url, headers={"Referer": str(response.url)})
        if channel.status_code != 200:
            raise KalturaIntegrationError("Kaltura Media Gallery could not be loaded.")
        _extract_gallery_id(str(channel.url))
        return KalturaMediaGallerySession(
            client,
            course_id=course_id,
            external_tool_id=external_tool_id,
            kaf_origin=kaf_origin,
            channel_url=str(channel.url),
            channel_html=channel.text,
        )
    except Exception:
        await client.aclose()
        raise
