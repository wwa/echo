# ws_logging.py
import asyncio
import logging
import threading

_ws_loop = None
_ws_streams = {}


class WebSocketLogHandler(logging.Handler):
    def __init__(self, stream_name, formatter=None):
        super().__init__()
        self.stream_name = stream_name
        if formatter is not None:
            self.setFormatter(formatter)

    def emit(self, record):
        global _ws_loop, _ws_streams
        if _ws_loop is None:
            return

        stream = _ws_streams.get(self.stream_name)
        if not stream:
            return

        loop = _ws_loop
        queue = stream.get("queue")
        if not queue:
            return

        try:
            msg = self.format(record)
            asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
        except Exception:
            pass


async def _ws_broadcaster(stream_name: str):
    stream = _ws_streams[stream_name]
    queue = stream["queue"]
    clients = stream["clients"]

    while True:
        msg = await queue.get()
        if not clients:
            continue

        dead = []
        for ws in list(clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)

        for ws in dead:
            try:
                clients.remove(ws)
            except KeyError:
                pass


async def _ws_handler(connection):
    """
    Single server handler (websockets >= 14).

      /app   -> 'app'
      /llm   -> 'llm'
      /trace -> 'trace'
      /tools -> 'tools'

    Anything else falls back to 'app'.
    """
    request = getattr(connection, "request", None)
    raw_path = getattr(request, "path", "/") if request is not None else "/"

    path = raw_path.split("?", 1)[0]
    p = (path or "/").strip("/")

    if p in _ws_streams:
        stream_name = p
    elif p == "" or p == "/":
        stream_name = "app"
    else:
        stream_name = "app"

    stream = _ws_streams.get(stream_name)
    clients = stream["clients"]
    clients.add(connection)

    logging.getLogger("echo.trace").debug(
        "WS client connected: path=%s -> stream=%s", raw_path, stream_name
    )

    try:
        async for _ in connection:
            # Ignore incoming messages
            pass
    finally:
        clients.discard(connection)
        logging.getLogger("echo.trace").debug(
            "WS client disconnected: path=%s -> stream=%s", raw_path, stream_name
        )


def _start_ws_server(host: str, port: int, stream_names):
    global _ws_loop, _ws_streams
    import websockets

    loop = asyncio.new_event_loop()
    _ws_loop = loop
    asyncio.set_event_loop(loop)

    async def server_main():
        for name in stream_names:
            _ws_streams[name] = {
                "queue": asyncio.Queue(),
                "clients": set(),
            }

        for name in stream_names:
            asyncio.create_task(_ws_broadcaster(name))

        # Start websocket server and keep it alive forever
        async with websockets.serve(_ws_handler, host, port):
            print(f"[WS-LOG] WebSocket log server on ws://{host}:{port}")
            print("[WS-LOG] Streams: " + ", ".join(f"/{n}" for n in stream_names))
            await asyncio.Future()  # run forever

    try:
        loop.run_until_complete(server_main())
    finally:
        loop.close()


def setup_ws_log_streaming(host: str,
                           port: int,
                           stream_to_logger: dict,
                           formatter: logging.Formatter):
    t = threading.Thread(
        target=_start_ws_server,
        args=(host, port, list(stream_to_logger.keys())),
        daemon=True,
    )
    t.start()

    # Attach WS handlers to configured loggers
    for stream_name, logger_name in stream_to_logger.items():
        log = logging.getLogger(logger_name)
        ws_handler = WebSocketLogHandler(stream_name, formatter)
        ws_handler.setLevel(logging.DEBUG)
        log.addHandler(ws_handler)

    print(
        f"WebSocket log streaming enabled at ws://{host}:{port} "
        f"with paths " + ", ".join(f"/{n}" for n in stream_to_logger.keys())
    )

    logging.getLogger("echo").info("WS streaming: test log from 'echo'")
