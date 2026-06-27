from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texas_holdem.web_session import PokerWebSession


WEB_ROOT = ROOT / "web"


class PokerRequestHandler(SimpleHTTPRequestHandler):
    session: PokerWebSession

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._send_json(self.session.state())
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path == "/api/new-hand":
                state = self.session.new_hand(
                    ai_count=payload.get("ai_count"),
                    seed=payload.get("seed"),
                )
                self._send_json(state)
                return
            if self.path == "/api/action":
                state = self.session.act(payload["action"])
                self._send_json(state)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route")
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK):
        encoded = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Serve the local Texas Hold'em Web table.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ai-count", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "dqn.pt")
    args = parser.parse_args()

    PokerRequestHandler.session = PokerWebSession(
        seed=args.seed,
        ai_count=args.ai_count,
        checkpoint_path=args.checkpoint,
    )
    server = ThreadingHTTPServer((args.host, args.port), PokerRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Texas Hold'em Web table running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
