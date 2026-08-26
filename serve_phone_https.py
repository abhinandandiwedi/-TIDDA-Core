from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl

ROOT = Path(__file__).resolve().parent / "html"
CERT_FILE = Path(__file__).resolve().parent / "local-cert.pem"
KEY_FILE = Path(__file__).resolve().parent / "local-key.pem"
HOST = "0.0.0.0"
PORT = 8444

if not CERT_FILE.is_file() or not KEY_FILE.is_file():
    raise SystemExit("local-cert.pem and local-key.pem are required")

class PhoneHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

server = ThreadingHTTPServer((HOST, PORT), PhoneHandler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
server.socket = context.wrap_socket(server.socket, server_side=True)
print(f"Serving phone client at https://192.168.1.4:{PORT}/phone_client.html")
server.serve_forever()
