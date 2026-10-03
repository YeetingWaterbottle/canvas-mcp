import httpx
import pytest

import canvas_mcp.integrations.kaltura as kaltura


def test_parse_gallery_extracts_media_entries_and_total():
    page = """
    <ul id="gallery">
      <li class="galleryItem" aria-setsize="2" aria-posinset="1">
        <div class="photo-group thumb_wrapper" title="Lecture 4">
          <a class="item_link" href="/media/t/1_abc123/420315023">Lecture 4</a>
        </div>
        <span class="duration">50:58</span>
      </li>
      <li class="galleryItem" aria-setsize="2" aria-posinset="2">
        <div class="photo-group thumb_wrapper" title="Lecture 3">
          <a class="item_link" href="/media/t/1_def456/420315023">Lecture 3</a>
        </div>
      </li>
    </ul>
    """

    entries, total = kaltura._parse_gallery(page)

    assert total == 2
    assert [(entry.entry_id, entry.gallery_id, entry.title) for entry in entries] == [
        ("1_abc123", "420315023", "Lecture 4"),
        ("1_def456", "420315023", "Lecture 3"),
    ]
    assert entries[0].duration == "50:58"


def test_parse_player_config_uses_player_script_not_other_ks_values():
    page = """
    <html><head>
      <script>var analytics = {"ks": "wrong-analytics-ks"};</script>
      <title>Lecture 4 - University of California, San Diego</title>
    </head><body>
      <script id="playerScript">
        var config = {
          "provider": {
            "partnerId": 2323111,
            "env": {"serviceUrl": "https:\\/\\/www.kaltura.com\\/api_v3"},
            "ks": "right-player-ks"
          }
        };
      </script>
    </body></html>
    """

    ks, service_url, title = kaltura._parse_player_config(page)

    assert ks == "right-player-ks"
    assert service_url == "https://www.kaltura.com/api_v3"
    assert title == "Lecture 4"


def test_vtt_to_text_removes_timing_and_deduplicates_adjacent_cues():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:02.000
Hello <c>class</c>.

00:00:02.000 --> 00:00:03.000
Hello <c>class</c>.

00:00:03.000 --> 00:00:04.000
Today we discuss DFS &amp; graphs.
"""

    assert kaltura.vtt_to_text(vtt) == "Hello class.\nToday we discuss DFS & graphs."


@pytest.mark.asyncio
async def test_open_media_gallery_completes_canvas_lti_flow(monkeypatch):
    async def fake_canvas_request(method, endpoint, params=None):
        assert method == "GET"
        assert endpoint == "/courses/78014/external_tools/sessionless_launch"
        assert params == {"id": "5826", "launch_type": "course_navigation"}
        return {
            "id": 5826,
            "url": "https://canvas.example.edu/courses/78014/external_tools/5826?session_token=secret",
        }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and url.startswith(
            "https://canvas.example.edu/courses/78014/external_tools/5826"
        ):
            return httpx.Response(
                200,
                text="""
                <form method="post" action="https://kaf.example.edu/hosted/index/oidc-init">
                  <input name="iss" value="https://canvas.instructure.com">
                  <input name="login_hint" value="secret-login-hint">
                  <input name="client_id" value="client">
                  <input name="lti_deployment_id" value="deployment">
                  <input name="target_link_uri" value="https://kaf.example.edu/course-gallery">
                  <input name="lti_message_hint" value="secret-message-hint">
                </form>
                """,
                request=request,
            )
        if (
            request.method == "POST"
            and url == "https://kaf.example.edu/hosted/index/oidc-init"
        ):
            return httpx.Response(
                302,
                headers={"Location": "https://canvas.example.edu/api/lti/authorize"},
                request=request,
            )
        if (
            request.method == "GET"
            and url == "https://canvas.example.edu/api/lti/authorize"
        ):
            return httpx.Response(
                200,
                text="""
                <form method="post" action="https://kaf.example.edu/hosted/index/oauth2-launch">
                  <input name="id_token" value="secret-id-token">
                  <input name="state" value="secret-state">
                </form>
                """,
                request=request,
            )
        if (
            request.method == "POST"
            and url == "https://kaf.example.edu/hosted/index/oauth2-launch"
        ):
            return httpx.Response(
                200,
                text="<script>window.location.href = '/channel/78014/420315023';</script>",
                request=request,
            )
        if (
            request.method == "GET"
            and url == "https://kaf.example.edu/channel/78014/420315023"
        ):
            return httpx.Response(
                200,
                text="""
                <li class="galleryItem" aria-setsize="1">
                  <div class="photo-group thumb_wrapper" title="Lecture 4">
                    <a class="item_link" href="/media/t/1_abc123/420315023">Lecture 4</a>
                  </div>
                </li>
                """,
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {url}")

    real_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    )
    monkeypatch.setattr(kaltura, "make_canvas_request", fake_canvas_request)
    monkeypatch.setattr(kaltura.httpx, "AsyncClient", lambda **_kwargs: real_client)

    session = await kaltura.open_media_gallery(78014, external_tool_id=5826)
    try:
        assert session.gallery_id == "420315023"
        assert session.kaf_origin == "https://kaf.example.edu"
        entries, total = await session.list_entries()
        assert total == 1
        assert [entry.entry_id for entry in entries] == ["1_abc123"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_caption_asset_list_and_large_segment_transcript():
    media_page = """
    <html><head><title>Lecture 4</title></head><body>
      <script id="playerScript">
        var config = {"provider": {
          "env": {"serviceUrl": "https:\\/\\/www.kaltura.com\\/api_v3"},
          "ks": "temporary-player-ks"
        }};
      </script>
      1_abc123
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://kaf.example.edu/media/t/1_abc123/420315023":
            return httpx.Response(200, text=media_page, request=request)
        if (
            url
            == "https://www.kaltura.com/api_v3/service/caption_captionasset/action/list"
        ):
            form = request.content.decode()
            assert "temporary-player-ks" in form
            return httpx.Response(
                200,
                json={
                    "totalCount": 1,
                    "objects": [
                        {
                            "id": "1_caption1",
                            "entryId": "1_abc123",
                            "language": "English",
                            "languageCode": "en",
                            "label": "English (auto-generated)",
                            "isDefault": True,
                            "status": 2,
                            "accuracy": 96,
                            "fileExt": "srt",
                        }
                    ],
                    "objectType": "KalturaCaptionAssetListResponse",
                },
                request=request,
            )
        if "/serveWebVTT/captionAssetId/1_caption1/segmentDuration/10000/" in url:
            assert "temporary-player-ks" in url
            return httpx.Response(
                200,
                text="WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello class.\n",
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = kaltura.KalturaMediaGallerySession(
        client,
        course_id=78014,
        external_tool_id=5826,
        kaf_origin="https://kaf.example.edu",
        channel_url="https://kaf.example.edu/channel/78014/420315023",
        channel_html="",
    )
    try:
        transcript = await session.get_transcript("1_abc123")
    finally:
        await session.close()

    assert transcript.caption.asset_id == "1_caption1"
    assert transcript.used_segment_fallback is False
    assert kaltura.vtt_to_text(transcript.vtt) == "Hello class."


@pytest.mark.asyncio
async def test_caption_api_exception_does_not_echo_player_ks():
    secret = "do-not-echo-this-player-ks"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "INVALID_KS",
                "message": "Invalid KS",
                "args": [{"name": "ks", "value": secret}],
                "objectType": "KalturaAPIException",
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = kaltura.KalturaMediaGallerySession(
        client,
        course_id=78014,
        external_tool_id=5826,
        kaf_origin="https://kaf.example.edu",
        channel_url="https://kaf.example.edu/channel/78014/420315023",
        channel_html="",
    )
    try:
        with pytest.raises(kaltura.KalturaIntegrationError) as exc_info:
            await session._caption_assets(
                "1_abc123",
                ks=secret,
                service_url="https://www.kaltura.com/api_v3",
            )
    finally:
        await session.close()

    assert secret not in str(exc_info.value)
