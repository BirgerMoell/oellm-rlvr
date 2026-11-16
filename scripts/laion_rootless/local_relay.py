#!/usr/bin/env python3
"""Loopback relay for rootless builds: clients (apt, pip, git, JVMs, ...) use an UNauthenticated proxy on 127.0.0.1; this
adds the build proxy's credentials and forwards. Java and others mishandle proxy credentials in the URL, so the token
stays here instead of in every tool's environment.

  local_relay.py UPSTREAM_HOSTFILE TOKENFILE PORTFILE [MAX_SECONDS]

Binds 127.0.0.1 on a free port (written to PORTFILE). Only connections from the same uid are served: the peer's uid is read
from /proc/net/tcp(6), because other users share the node's loopback."""
import asyncio, base64, os, sys

UID = os.getuid()

def peer_uid(port):
    want = "0100007F:%04X" % port
    for path in ("/proc/net/tcp",):
        try:
            for line in open(path).read().split("\n")[1:]:
                f = line.split()
                if len(f) > 7 and f[1].upper() == want:
                    return int(f[7])
        except OSError:
            pass
    return None

async def pipe(r, w):
    try:
        while True:
            d = await r.read(65536)
            if not d:
                break
            w.write(d); await w.drain()
    except Exception:
        pass
    finally:
        try: w.close()
        except Exception: pass

def make_handler(upstream, auth):
    async def handle(cr, cw):
        try:
            if peer_uid(cw.get_extra_info("peername")[1]) != UID:
                cw.close(); return
            head = await cr.readuntil(b"\r\n\r\n")
            lines = head.split(b"\r\n")
            lines = [l for l in lines if not l.lower().startswith(b"proxy-authorization")]
            lines.insert(1, b"Proxy-Authorization: " + auth)
            host, port = upstream
            ur, uw = await asyncio.open_connection(host, port)
            uw.write(b"\r\n".join(lines)); await uw.drain()
            await asyncio.gather(pipe(cr, uw), pipe(ur, cw))   # CONNECT 200 and plain responses flow back unchanged
        except Exception:
            try: cw.close()
            except Exception: pass
    return handle

async def main(hostfile, tokenfile, portfile, max_s):
    host, _, port = open(hostfile).read().strip().rpartition(":")
    token = open(tokenfile).read().strip()
    auth = b"Basic " + base64.b64encode(("build:%s" % token).encode())
    srv = await asyncio.start_server(make_handler((host, int(port)), auth), "127.0.0.1", 0)
    open(portfile, "w").write(str(srv.sockets[0].getsockname()[1]))
    async with srv:
        try: await asyncio.wait_for(srv.serve_forever(), max_s)
        except asyncio.TimeoutError: pass

if __name__ == "__main__":
    a = sys.argv
    asyncio.run(main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 43200))
