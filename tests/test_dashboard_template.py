from dashboard.app import create_app


def test_voice_only_index():
    """Voice-only page: arc UI, no telemetry grid, voice_console script."""
    app = create_app(None)
    r = app.test_client().get("/")
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "/static/css/main.css" in html
    assert "/static/js/voice_console.js" in html
    assert 'id="main-content"' in html
    assert "voice-only" in html
    assert "arc-reactor-panel" in html
    assert "systems-dashboard" not in html
    assert "noindex" in html
    assert 'id="langSelect"' in html
