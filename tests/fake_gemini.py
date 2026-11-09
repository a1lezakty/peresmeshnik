"""A fake Gemini Live server on 127.0.0.1 for the real google-genai client (TLS with the test-only certificate
tests/data/fake-gemini.pem: the client always speaks wss). The translator answers each block of input `lag` s later
with as long a "translation" at 24 kHz: a tone for speech, zeros for silence (the real one streams zeros through
silence), and a resumption handle now and then. Faults, set from the test's thread:

    drop()                 the open connections end without a close frame (the network is gone)
    refuse = (1011, why)   new connections are closed with that code after their setup (quota, a bad key)
    down = True            new connections get HTTP 503 at the handshake (the server is not reachable)
    lag = 10.0             the translation comes this late (too slow)

`client()` is a genai.Client for it; `inputs` lists (connection number, first sample of each block) as received.
"""
from __future__ import annotations

import asyncio
import base64
import json
import ssl
import threading
import time
from pathlib import Path

import numpy as np

PEM = Path(__file__).parent / 'data' / 'fake-gemini.pem'
OUT_RATE = 24_000


class FakeGemini:
    def __init__(self, lag: float = 0.3):
        self.lag, self.refuse, self.down = lag, None, False
        self.inputs: list[tuple[int, int]] = []
        self.connections = 0
        self._open: set = set()
        self._loop = asyncio.new_event_loop()
        started = threading.Event()
        threading.Thread(target=self._run, args=(started,), name='fake-gemini', daemon=True).start()
        started.wait(10)

    def _run(self, started: threading.Event) -> None:
        from websockets.asyncio.server import serve
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(PEM)

        async def main():
            async with serve(self._serve, '127.0.0.1', 0, ssl=ctx, process_request=self._handshake) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._stop = asyncio.Event()
                started.set()
                await self._stop.wait()
        self._loop.run_until_complete(main())

    def client(self):
        from google import genai
        from google.genai import types
        ctx = ssl.create_default_context(cafile=str(PEM))
        return genai.Client(api_key='fake-key', http_options=types.HttpOptions(
            api_version='v1beta', base_url=f'https://127.0.0.1:{self.port}', async_client_args={'ssl': ctx}))

    def drop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: [ws.transport.abort() for ws in list(self._open)])

    def close(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop.set())

    # server side ---------------------------------------------------------------------------------
    def _handshake(self, connection, request):
        if self.down:
            return connection.respond(503, 'Service Unavailable\n')
        return None

    async def _serve(self, ws) -> None:
        json.loads(await ws.recv())  # the setup
        if self.refuse:
            code, why = self.refuse
            await ws.close(code, why)
            return
        self.connections += 1
        n = self.connections
        await ws.send(json.dumps({'setupComplete': {}}))
        out: asyncio.Queue = asyncio.Queue()
        sender = asyncio.create_task(self._send(ws, out))
        self._open.add(ws)
        k = 0
        try:
            async for raw in ws:
                msg = json.loads(raw)
                m = msg.get('realtime_input') or msg.get('realtimeInput') or {}
                audio = m.get('audio') or next(iter(m.get('mediaChunks') or ()), None)
                if not audio:
                    continue
                data = audio['data']  # the client sends URL-safe base64 without padding
                blk = np.frombuffer(base64.urlsafe_b64decode(data + '=' * (-len(data) % 4)), '<i2')
                self.inputs.append((n, int(blk[0])))
                ans = np.zeros(len(blk) * OUT_RATE // 16_000, '<i2')
                if np.sqrt(np.mean(blk.astype(np.float32) ** 2)) > 500:  # speech: a tone of the same length
                    ans = (6000 * np.sin(np.arange(len(ans)) * 2 * np.pi * 440 / OUT_RATE)).astype('<i2')
                k += 1
                msgs = [{'serverContent': {'modelTurn': {'parts': [{'inlineData': {
                    'mimeType': f'audio/pcm;rate={OUT_RATE}', 'data': base64.b64encode(ans.tobytes()).decode()}}]}}}]
                if k % 10 == 0:
                    msgs.append({'sessionResumptionUpdate': {'newHandle': f'h{n}.{k}', 'resumable': True}})
                out.put_nowait((time.monotonic() + self.lag, msgs))
        except Exception:  # dropped (ConnectionClosed)
            pass
        finally:
            self._open.discard(ws)
            sender.cancel()

    @staticmethod
    async def _send(ws, out: asyncio.Queue) -> None:
        while True:
            due, msgs = await out.get()
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            for m in msgs:
                await ws.send(json.dumps(m))
