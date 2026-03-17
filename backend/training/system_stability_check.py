"""End-to-end 5-minute stability checker for backend + frontend + websocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", type=str, default="http://localhost:5173")
    return parser.parse_args()


def check_http(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(response.status)
            return status == 200, f"HTTP {status}"
    except urllib.error.URLError as exc:
        return False, str(exc)


async def run_stability(minutes: float, backend: str, frontend: str) -> dict:
    end_time = time.time() + minutes * 60.0
    ws_url = backend.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live-metrics"

    backend_ok, backend_msg = check_http(f"{backend}/api/health")
    frontend_ok, frontend_msg = check_http(frontend)

    report = {
        "duration_minutes": minutes,
        "backend_health_initial": backend_ok,
        "frontend_health_initial": frontend_ok,
        "backend_health_initial_msg": backend_msg,
        "frontend_health_initial_msg": frontend_msg,
        "messages_received": 0,
        "ws_disconnects": 0,
        "errors": [],
        "start_epoch": time.time(),
        "end_epoch": None,
        "pass": False,
    }

    while time.time() < end_time:
        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                while time.time() < end_time:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    json.loads(msg)
                    report["messages_received"] += 1
        except Exception as exc:
            report["ws_disconnects"] += 1
            report["errors"].append(str(exc))
            await asyncio.sleep(1.0)

    backend_ok_final, backend_msg_final = check_http(f"{backend}/api/health")
    frontend_ok_final, frontend_msg_final = check_http(frontend)

    report["backend_health_final"] = backend_ok_final
    report["frontend_health_final"] = frontend_ok_final
    report["backend_health_final_msg"] = backend_msg_final
    report["frontend_health_final_msg"] = frontend_msg_final
    report["end_epoch"] = time.time()

    report["pass"] = (
        report["backend_health_initial"]
        and report["frontend_health_initial"]
        and report["backend_health_final"]
        and report["frontend_health_final"]
        and report["messages_received"] > int(minutes * 60)
        and report["ws_disconnects"] <= 5
    )

    return report


async def async_main() -> None:
    args = parse_args()
    report = await run_stability(args.minutes, args.backend, args.frontend)

    out_path = RESULTS_DIR / "system_stability_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Stability check complete")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
