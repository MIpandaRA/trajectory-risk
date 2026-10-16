"""TCP transport between a producer (simulator, replayer, or a real tracker)
and consumers (dashboard, recorder). Framing and message schema are in
protocol.py.

Why a localhost socket rather than multiprocessing.Queue: the producer and
consumer are independent processes that start, stop and restart on their own
(and pygame wants one window per process). A Queue only connects processes
forked from one parent; a socket also means a real sensor feed on another
machine is just a different host:port.

Both ends refuse to let the other slow them down:
* the server drops a consumer whose socket errors instead of blocking the sim;
* the client reads on a background thread into a small drop-oldest queue, so
  a slow render frame never backs up the socket.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
from typing import Callable, Optional

from .protocol import encode

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class LiveStateServer:
    """Runs in Window 1. Accepts any number of consumer connections and
    broadcasts every tick to all of them. A slow/dead consumer is dropped
    rather than allowed to back-pressure the simulation -- Window 1's own
    frame rate must never depend on whether anyone is listening.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 hello: Optional[dict] = None):
        """hello: if given, sent to each consumer individually right when
        it connects (NOT a one-time broadcast() -- a consumer connecting
        after that single broadcast would otherwise miss it entirely,
        since broadcast() only reaches sockets already in self._clients)."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(8)
        self._sock.setblocking(False)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        self._hello = hello
        self.port = self._sock.getsockname()[1]
        print(f"[LiveStateServer] Listening on {host}:{self.port}")

    def accept_pending(self) -> None:
        """Non-blocking: call once per tick from the main loop to pick up
        any consumers that have connected since the last call."""
        while True:
            try:
                conn, addr = self._sock.accept()
            except BlockingIOError:
                return
            conn.setblocking(True)
            conn.settimeout(1.0)
            if self._hello is not None:
                try:
                    conn.sendall(encode(self._hello))
                except OSError:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    continue
            with self._lock:
                self._clients.append(conn)
            print(f"[LiveStateServer] Consumer connected from {addr}")

    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, message: dict) -> None:
        line = encode(message)
        with self._lock:
            alive = []
            for conn in self._clients:
                try:
                    conn.sendall(line)
                    alive.append(conn)
                except OSError:
                    try:
                        conn.close()
                    except OSError:
                        pass
            self._clients = alive

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for conn in self._clients:
                try:
                    conn.close()
                except OSError:
                    pass
            self._clients = []
        self._sock.close()


class LiveStateClient:
    """Runs in Window 2. Connects to Window 1's server and reads messages
    on a background thread into a bounded queue, so a rendering hiccup on
    the main thread never stalls the socket's own recv buffer.

    The queue is small and drop-oldest: Window 2 cares about the LATEST
    ground truth for smooth rendering, not about replaying every single
    historical tick if it ever falls behind.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 maxsize: int = 256, on_connect: Optional[Callable[[], None]] = None):
        self._host, self._port = host, port
        self.queue: "queue.Queue[dict]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._on_connect = on_connect
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self._host, self._port), timeout=5.0)
            except OSError:
                self._stop.wait(1.0)
                continue
            print(f"[LiveStateClient] Connected to {self._host}:{self._port}")
            if self._on_connect:
                self._on_connect()
            fh = sock.makefile("r", encoding="utf-8")
            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._push(msg)
                    if self._stop.is_set():
                        break
            except OSError:
                pass
            finally:
                try:
                    fh.close()
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            if not self._stop.is_set():
                print("[LiveStateClient] Disconnected, retrying...")
                self._stop.wait(1.0)

    def _push(self, msg: dict) -> None:
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
