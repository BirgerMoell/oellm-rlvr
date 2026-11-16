#!/usr/bin/env python3
"""Minimal token-protected HTTP/CONNECT forward proxy (stdlib only) so offline Leonardo compute nodes can reach package
mirrors and container registries during image builds. Run it on an internet-enabled node (lrd_all_serial) for the
duration of a build job; clients use http://build:<token>@<host>:<port> as http_proxy/https_proxy.

  laion_proxy.py PORT HOSTFILE TOKENFILE [MAX_SECONDS]

PORT 0 picks a free port; HOSTFILE receives host:port; TOKENFILE (mode 600, created if missing) holds the shared
secret. Requests without the matching Proxy-Authorization get 407. The proxy exits after MAX_SECONDS (default 1500)."""
import asyncio, base64, os, secrets, socket, sys

TOKEN = ""

async def pipe(r, w):
    try:
        while (d := await r.read(65536)):
            w.write(d); await w.drain()
    except Exception:
        pass
    finally:
        try: w.close()
        except Exception: pass

def authorized(lines):
    want = "Basic " + base64.b64encode(f"build:{TOKEN}".encode()).decode()
    for l in lines[1:]:
        if l.lower().startswith(b"proxy-authorization:"):
            return secrets.compare_digest(l.split(b":", 1)[1].strip().decode("latin1"), want)
    return False

async def handle(cr, cw):
    try:
        head = await cr.readuntil(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        if not authorized(lines):
            cw.write(b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=build\r\nContent-Length: 0\r\n\r\n")
            await cw.drain(); cw.close(); return
        method, target, ver = lines[0].decode("latin1").split(" ", 2)
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
            ur, uw = await asyncio.open_connection(host, int(port))
            cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n"); await cw.drain()
            await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
        else:  # absolute-URI request (plain http, e.g. apt mirrors)
            assert target.startswith("http://"), target
            hostport, _, path = target[7:].partition("/")
            host, _, port = hostport.partition(":")
            ur, uw = await asyncio.open_connection(host, int(port or 80))
            lines[0] = f"{method} /{path} {ver}".encode()
            lines = [l for l in lines if not l.lower().startswith((b"proxy-connection", b"proxy-authorization"))]
            uw.write(b"\r\n".join(lines)); await uw.drain()
            await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    except Exception:
        try: cw.close()
        except Exception: pass

async def main(port, hostfile, tokenfile, max_s):
    global TOKEN
    if not os.path.exists(tokenfile):
        fd = os.open(tokenfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, secrets.token_urlsafe(24).encode()); os.close(fd)
    TOKEN = open(tokenfile).read().strip()
    srv = await asyncio.start_server(handle, "0.0.0.0", port)
    p = srv.sockets[0].getsockname()[1]
    open(hostfile, "w").write(f"{socket.gethostname()}:{p}\n")
    print(f"proxy on {socket.gethostname()}:{p}, exits in {max_s}s", flush=True)
    async with srv:
        try: await asyncio.wait_for(srv.serve_forever(), max_s)
        except asyncio.TimeoutError: pass

if __name__ == "__main__":
    a = sys.argv
    asyncio.run(main(int(a[1]), a[2], a[3], int(a[4]) if len(a) > 4 else 1500))
