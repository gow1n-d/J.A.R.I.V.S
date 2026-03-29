"""
Boot J.A.R.V.I.S and serve the voice-only web UI (arc reactor).
Run from project root: python run_voice_web.py
Then open http://127.0.0.1:5000 — use Chrome or Edge for speech input.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.jarvis import JARVIS


async def main():
    jarvis = JARVIS()
    await jarvis.initialize()
    await jarvis.launch_dashboard()
    print("Voice UI: open http://127.0.0.1:5000  (Ctrl+C to stop)\n")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await jarvis.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
