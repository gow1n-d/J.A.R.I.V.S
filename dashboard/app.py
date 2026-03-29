"""
J.A.R.V.I.S — Web Dashboard
Real-time monitoring, control, and visualization of J.A.R.V.I.S state.
"""

import asyncio
import concurrent.futures
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
import json


def _run_jarvis_async(coro):
    """
    Run async J.A.R.V.I.S methods from Flask (always a sync context here).
    Uses a worker thread so this stays reliable even if the caller thread
    already has a running asyncio loop (tests, nested contexts).
    """

    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result(timeout=600)


def create_app(jarvis=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "jarvis-dashboard-secret"
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        if not jarvis:
            return jsonify({"error": "J.A.R.V.I.S not connected"}), 503

        return jsonify({
            "status": "online",
            "uptime": time.time() - jarvis.metrics.get("start_time", time.time()),
            "requests": jarvis.metrics.get("total_requests", 0),
            "success_rate": round(
                jarvis.metrics.get("successful_tasks", 0) /
                max(1, jarvis.metrics.get("total_requests", 1)) * 100, 1
            ),
            "avg_response": round(jarvis.metrics.get("avg_response_time", 0.0), 3),
            "mode": jarvis.personality.current_mode if jarvis else "unknown",
            "rl_stats": jarvis.rl_engine.get_stats() if jarvis else {},
            "world": jarvis.world_model.get_world_summary() if jarvis else {},
            "safety": jarvis.safety.get_stats() if jarvis else {}
        })

    @app.route("/api/memory")
    def api_memory():
        if not jarvis:
            return jsonify({}), 503
        return jsonify({
            "short_term": len(jarvis.memory.short_term.get_all()),
            "recent": [
                item.get("input", "")[:80]
                for item in jarvis.memory.short_term.get_recent(5)
            ]
        })

    @app.route("/api/command", methods=["POST"])
    def api_command():
        if not jarvis:
            return jsonify({"error": "J.A.R.V.I.S offline"}), 503
        data = request.get_json(silent=True) or {}
        cmd = (data.get("command") or "").strip()
        if not cmd:
            return jsonify({"error": "No command provided"}), 400

        want_voice = data.get("voice", True)
        lang = (data.get("lang") or data.get("locale") or "").strip() or None

        if want_voice:
            result = _run_jarvis_async(
                jarvis.process_and_speak(cmd, response_locale=lang)
            )
            return jsonify({
                "response": result.get("text", ""),
                "audio": result.get("audio"),
                "llm_active": result.get("llm_active", False),
                "tts_active": result.get("tts_active", False),
                "lang": lang,
                "tts_voice": result.get("tts_voice"),
                "timestamp": time.time(),
            })
        text = _run_jarvis_async(jarvis.process(cmd, response_locale=lang))
        tts = getattr(jarvis, "tts", None)
        return jsonify({
            "response": text,
            "audio": None,
            "llm_active": getattr(jarvis.llm, "is_llm_active", False),
            "tts_active": bool(tts and getattr(tts, "is_available", False)),
            "timestamp": time.time(),
        })

    @socketio.on("connect")
    def on_connect():
        emit("status", {"message": "Connected to J.A.R.V.I.S Dashboard"})

    return app
