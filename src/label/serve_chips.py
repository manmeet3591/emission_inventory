"""Static file server for chips with CORS headers (Label Studio fetches images cross-origin).

    python -m src.label.serve_chips --dir data/tiles --port 8081
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "max-age=3600")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    ThreadingHTTPServer((a.bind, a.port), functools.partial(CORSHandler, directory=a.dir)).serve_forever()


if __name__ == "__main__":
    main()
