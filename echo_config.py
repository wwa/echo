# echo_config.py
import os
import logging

from ws_logging import setup_ws_log_streaming  # assumes you already have ws_logging.py

LOG_DIR = os.getenv("LOG_DIR", "logs")
DEFAULT_CONTEXT_LIMIT = 128000
MODEL_CONTEXT_LIMITS = {
    "gpt-4-turbo-preview": 128000,
    "gpt-4.1": 128000,
    "gpt-4.1-mini": 128000,
    "gpt-4.1-small": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-transcribe": 16000,
    "gpt-5-mini": 400000,
    "gpt-5": 400000,
    "gpt-5.1": 400000,
}


def init_logging_and_ws():
    """
    Set up:
      - root logger
      - file handlers (important / other / llm / trace / tools)
      - named loggers: echo, echo.llm, echo.trace, echo.toolkit
      - optional WebSocket log streaming

    Returns:
      CONTEXT_WARN_THRESHOLD (float) parsed from env.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # ----- root logger -----
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    # ----- file handlers -----
    important_handler = logging.FileHandler(os.path.join(LOG_DIR, "important.log"))
    important_handler.setLevel(logging.INFO)
    important_handler.setFormatter(formatter)

    other_handler = logging.FileHandler(os.path.join(LOG_DIR, "other.log"))
    other_handler.setLevel(logging.DEBUG)
    other_handler.setFormatter(formatter)

    llm_handler = logging.FileHandler(os.path.join(LOG_DIR, "llm.log"))
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.setFormatter(formatter)

    trace_handler = logging.FileHandler(os.path.join(LOG_DIR, "trace.log"))
    trace_handler.setLevel(logging.DEBUG)
    trace_handler.setFormatter(formatter)

    tools_handler = logging.FileHandler(os.path.join(LOG_DIR, "tools.log"))
    tools_handler.setLevel(logging.DEBUG)
    tools_handler.setFormatter(formatter)

    # ----- named loggers -----
    echo_logger = logging.getLogger("echo")
    echo_logger.setLevel(logging.DEBUG)
    echo_logger.handlers.clear()
    echo_logger.addHandler(important_handler)
    echo_logger.addHandler(other_handler)
    echo_logger.propagate = False

    llm_logger = logging.getLogger("echo.llm")
    llm_logger.setLevel(logging.DEBUG)
    llm_logger.handlers.clear()
    llm_logger.addHandler(llm_handler)
    llm_logger.propagate = False

    trace_logger = logging.getLogger("echo.trace")
    trace_logger.setLevel(logging.DEBUG)
    trace_logger.handlers.clear()
    trace_logger.addHandler(trace_handler)
    trace_logger.propagate = False

    toolkit_logger = logging.getLogger("echo.toolkit")
    toolkit_logger.setLevel(logging.DEBUG)
    toolkit_logger.handlers.clear()
    toolkit_logger.addHandler(tools_handler)  # tools.log (DEBUG+)
    toolkit_logger.propagate = False

    # ----- optional WebSocket log streaming -----
    ws_enabled = os.getenv("WEBSOCKET_LOG_ENABLED", "false").lower() == "true"
    if ws_enabled:
        ws_host = os.getenv("WEBSOCKET_LOG_HOST", "127.0.0.1")
        ws_port = int(os.getenv("WEBSOCKET_LOG_PORT", "9876"))

        ws_streams_cfg = {
            "app":   "echo",
            "llm":   "echo.llm",
            "trace": "echo.trace",
            "tools": "echo.toolkit",
        }

        try:
            setup_ws_log_streaming(ws_host, ws_port, ws_streams_cfg, formatter)
        except ImportError:
            print(
                "⚠️  WEBSOCKET_LOG_ENABLED=true but 'websockets' package "
                "is not installed. Disable it or run: pip install websockets"
            )
        except Exception as e:
            print(f"⚠️  Failed to start WebSocket log server: {e}")

    logger = logging.getLogger("echo")
    logger.info("Starting ECHO...")

    # ----- CONTEXT_WARN_THRESHOLD -----
    try:
        ctx_thr = float(os.getenv("CONTEXT_WARN_THRESHOLD", "0.90"))
        if not (0 < ctx_thr < 1):
            print("⚠️  Invalid CONTEXT_WARN_THRESHOLD in .env, using default 0.90")
            ctx_thr = 0.90
    except Exception:
        print("⚠️  Failed to parse CONTEXT_WARN_THRESHOLD, using default 0.90")
        ctx_thr = 0.90

    return ctx_thr
