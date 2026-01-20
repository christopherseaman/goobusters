#!/usr/bin/env python3
"""
Simplified iOS app entry point - starts a minimal HTTP server for testing
This version doesn't require Flask or other dependencies
"""

import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

# Write debug log to temp file
debug_log = "/tmp/goobusters_python_debug.log"
try:
    with open(debug_log, "w") as f:
        f.write(f"[iOS Python] STARTING - Simple HTTP Server\n")
        f.write(f"[iOS Python] Python version: {sys.version}\n")
        f.write(f"[iOS Python] Current directory: {os.getcwd()}\n")
        f.write(f"[iOS Python] sys.path: {sys.path}\n")
        f.write(f"[iOS Python] __name__: {__name__}\n")
        f.flush()
except Exception as e:
    pass  # Can't write, continue anyway

print("=" * 80)
print("[iOS Python] STARTING - Simple HTTP Server")
print(f"[iOS Python] Python version: {sys.version}")
print(f"[iOS Python] Current directory: {os.getcwd()}")
print("=" * 80)


class SimpleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to use print instead of stderr"""
        print(f"[HTTP] {format % args}")

    def do_GET(self):
        """Handle GET requests"""
        print(f"[HTTP] GET {self.path}")

        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK - Python is running!\n")
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""
            <html>
            <head><title>Goobusters iOS Python</title></head>
            <body style="font-family: sans-serif; padding: 20px;">
                <h1>🎉 Python is working on iOS!</h1>
                <p><strong>Python Version:</strong> {sys.version}</p>
                <p><strong>Working Directory:</strong> {os.getcwd()}</p>
                <p><strong>Executable:</strong> {sys.executable}</p>
                <hr>
                <p><a href="/healthz">Health Check</a></p>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found\n")


def main(port=8080):
    """Start the simple HTTP server"""
    print(f"[iOS Python] Starting HTTP server on port {port}...")
    
    # Write to debug log
    try:
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write(f"[iOS Python] main() called with port {port}\n")
            f.flush()
    except:
        pass

    import socket
    
    try:
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write("[iOS Python] Creating socket...\n")
            f.flush()
        
        # Try creating a basic socket first to test
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write("[iOS Python] Socket created, trying to bind...\n")
            f.flush()
        
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_socket.bind(("127.0.0.1", port))
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write(f"[iOS Python] Bound to 127.0.0.1:{port}, trying to listen...\n")
            f.flush()
        
        test_socket.listen(1)
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write("[iOS Python] OK Socket listening! Closing and using HTTPServer...\n")
            f.flush()
        test_socket.close()
        
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write("[iOS Python] Creating HTTPServer...\n")
            f.flush()
        server = HTTPServer(("127.0.0.1", port), SimpleHandler)
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write(f"[iOS Python] OK Server created, starting serve_forever\n")
            f.flush()
        print(f"[iOS Python] OK Server started successfully on http://127.0.0.1:{port}")
        print(f"[iOS Python] Serving requests... (press Ctrl+C to stop)")
        server.serve_forever()
    except Exception as e:
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write(f"[iOS Python] ERROR: {type(e).__name__}: {e}\n")
            import traceback
            f.write(traceback.format_exc())
            f.flush()
        print(f"[iOS Python] ERROR starting server: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        with open("/tmp/goobusters_python_debug.log", "a") as f:
            f.write("[iOS Python] Inside __main__ block, calling main() directly\n")
            f.flush()
    except:
        pass
    
    # Call main directly without argparse (it was hanging on parse_args)
    try:
        main(port=8080)
    except BaseException as e:  # Catch ALL exceptions including SystemExit
        try:
            with open("/tmp/goobusters_python_debug.log", "a") as f:
                f.write(f"[iOS Python] TOP-LEVEL EXCEPTION: {type(e).__name__}: {e}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
        except:
            pass
        raise
