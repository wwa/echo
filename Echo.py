import os
import json
import traceback
from timeit import default_timer as timer
import logging
from dotenv import load_dotenv

#Own
from Toolkit import BaseToolkit

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

def estimate_tokens_from_messages(messages):
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            # e.g. list for vision, or other structures
            try:
                total_chars += len(str(content))
            except Exception:
                pass
    return int(total_chars / 3.5)

def modelOne(toolkit, messages):
    logger = logging.getLogger("echo.llm")
    ts_s = timer()

    # Log request
    try:
        logger.info("LLM Request: model=%s", toolkit.openai_chat_model)
        logger.debug("LLM Request Messages: %s", json.dumps(messages, indent=2, ensure_ascii=False))
        logger.debug("LLM Tools Spec: %s", json.dumps(toolkit.toolMessage(), indent=2, ensure_ascii=False))
    except Exception:
        logger.exception("Failed logging LLM request")
        traceback.print_exc()
    print("Prompting...")

    res = toolkit.openai.chat.completions.create(
        model    = toolkit.openai_chat_model,
        messages = messages,
        tools    = toolkit.toolMessage(),
        tool_choice = "auto"
    )

    ts_e = timer()
    print(f"... took {ts_e-ts_s}s")

    # Log response
    try:
        logger.info("LLM Response received.")
        logger.debug("LLM Raw Response JSON: %s", res.model_dump_json(indent=2))
    except Exception:
        logger.exception("Failed logging LLM response")
        traceback.print_exc()
    reason  = res.choices[0].finish_reason
    message = res.choices[0].message

    if reason == "stop":
        messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'tool_calls'})))
        return reason, message.content, messages

    if reason == "tool_calls":
        messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'content'})))
        for tc in message.tool_calls:
            if tc.type == "function":
                messages.append(toolkit.call(tc.id, tc.function))

    return reason, None, messages

def modelLoop(toolkit, history=[]):
  # Determine which history messages will be used
  if getattr(toolkit, "chain_enabled", True):
    history_messages = sum(history, [])
  else:
    history_messages = []

  # ----------------------------------------------------------
  # Context window WARNING based on HISTORY ONLY
  # ----------------------------------------------------------
  if getattr(toolkit, "chain_enabled", True) and history_messages:
    try:
      used_tokens = estimate_tokens_from_messages(history_messages)
      max_tokens = MODEL_CONTEXT_LIMITS.get(toolkit.openai_chat_model, 128000)
      threshold = int(max_tokens * CONTEXT_WARN_THRESHOLD)

      if used_tokens >= threshold:
        percent = (used_tokens / max_tokens) * 100
        print(
          f"⚠️  WARNING: Conversation history uses ~{used_tokens}/{max_tokens} tokens "
          f"({percent:.1f}% of context window)."
        )
        print("⚠️  Consider 'clear', 'reset', or 'chain off' to avoid running out of context.\n")

        logging.getLogger("echo.context").warning(
          "History token usage: %s/%s (%.1f%%)",
          used_tokens, max_tokens, percent
        )
    except Exception:
      logging.getLogger("echo.context").exception("Failed to estimate history token usage")

  # Now build the full messages for this turn
  messages = [{
    "role": "system",
    "content": f"""
      You are a helpful assistant called ECHO.
      Based on user request and available functions devise a plan of action and execute it.
      Keep in mind multiple data sources are available. If you are unable to fulfil the request with one data source, try again with another.
      A demonstrative pronoun such as this/that/these/it likely refers to something in conversation history, or data copied to cliboard or something that user sees on his screen.
      Regardless of action taken, respond in JSON with {{plan:<plan>,response:<text response>}}
    """ + toolkit.toolPrompt()
  }] + history_messages + [
    {"role": "user", "content": toolkit.userPrompt()}
  ] + toolkit.fake('listTools') + toolkit.fake('clipboardRead')

  content = None
  while True:
    reason, content, messages = modelOne(toolkit, messages)
    if reason == "stop":
      break

  history.append(messages)
  return content, history

def promptOption(prompt, history, helpText, toolkit):
  loopBehav = ""

  if (prompt == "exit" or prompt == "Exit" or prompt == "e"):
    print("Goodbye!")
    loopBehav = "break"
  elif (prompt == "history" or prompt == "History" or prompt == "hh"):
    print(history)
    loopBehav = "continue"
  elif (prompt == "clear" or prompt == "Clear" or prompt == "c"):
    history = []
    loopBehav = "continue"
  elif (prompt == "reset" or prompt == "Reset" or prompt == "r"):
    toolkit.reset()
    print("All tools reset.")
    loopBehav = "continue"
  elif (prompt == "help" or prompt == "?" or prompt == "Help" or prompt == "h"):
    print(helpText)
    loopBehav = "continue"
  elif prompt.lower().startswith("log "):
    parts = prompt.split()
    if len(parts) >= 2:
      level_name = parts[1].upper()
      if level_name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger = logging.getLogger("echo")
        logger.setLevel(getattr(logging, level_name))
        logging.getLogger().setLevel(getattr(logging, level_name))

        print(f"Log level changed to {level_name}")
      else:
        print("Unknown log level. Use one of: debug, info, warning, error, critical.")
    else:
      print("Usage: log <level>, e.g. 'log debug' or 'log info'")
    loopBehav = "continue"
  elif prompt.lower().startswith("chain "):
    parts = prompt.split()
    if len(parts) >= 2:
      mode = parts[1].lower()
      if mode in ("on", "off"):
        toolkit.chain_enabled = (mode == "on")
        print(f"Conversation history chaining is now {'ENABLED' if toolkit.chain_enabled else 'DISABLED'}.")
      else:
        print("Usage: chain on|off")
    else:
      print("Usage: chain on|off")
    loopBehav = "continue"
  elif prompt.lower().startswith("profile "):
    parts = prompt.split()
    if len(parts) >= 2:
      profile = parts[1].lower()
      if hasattr(toolkit, "_apply_model_profile"):
        ok = toolkit._apply_model_profile(profile)
        if ok:
          print(f"Model profile switched to '{profile}'.")
          print("Active models:")
          print(f"  chat    : {toolkit.openai_chat_model}")
          print(f"  vision  : {toolkit.openai_vision_model}")
          print(f"  research: {toolkit.openai_research_model}")
          print(f"  stt     : {toolkit.openai_stt_model}")
        else:
          print(f"Unknown profile '{profile}'. Available: {', '.join(toolkit.model_profiles.keys())}")
      else:
        print("Toolkit does not support model profiles.")
    else:
      print("Usage: profile <name>, e.g. 'profile legacy' or 'profile current'")
    loopBehav = "continue"

  return loopBehav

def mainLoop(toolkit, limit=10):
  history = []

  helpText = (
    "Type 'history' to see conversation history. \n"
    "Type 'clear' to clear history. \n"
    "Type 'reset' to reset all tools. \n\n"
    "Type 'chain on/off' to enable or disable conversation history chaining. \n"
    "Type 'log LEVEL' to change log verbose lvl. \n"
    "Type 'profile NAME' to switch LLM model profile, e.g. 'profile legacy' or 'profile current'. \n\n"
    "Type 'exit' to quit if you need rest. \n\n")


  print("Welcome to ECHO! Deep dive into my power. \n "
        " ------------------ \n"
        f"{helpText}")

  while True:
    try:
      print(">> ", end="")
      prompt = toolkit.input()

      lOps = promptOption(prompt, history, helpText, toolkit)
      if lOps == "break":
        break
      elif lOps == "continue":
        continue

      print(f"User input: {prompt}")
      content, history = modelLoop(toolkit, history)
      history = history[:limit]
      print(content)
    except Exception as e:
      traceback.print_exc()
      pass

if __name__ == "__main__":
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # root logger
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    # handlers
    important_handler = logging.FileHandler(os.path.join(LOG_DIR, "important.log"))
    important_handler.setLevel(logging.INFO)
    important_handler.setFormatter(formatter)

    other_handler = logging.FileHandler(os.path.join(LOG_DIR, "other.log"))
    other_handler.setLevel(logging.DEBUG)
    other_handler.setFormatter(formatter)

    llm_handler = logging.FileHandler(os.path.join(LOG_DIR, "llm.log"))
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.setFormatter(formatter)

    # echo.* logger (app + toolkit)
    echo_logger = logging.getLogger("echo")
    echo_logger.setLevel(logging.DEBUG)
    echo_logger.handlers.clear()
    echo_logger.addHandler(important_handler)   # important.log  (INFO+)
    echo_logger.addHandler(other_handler)       # other.log      (DEBUG+)
    echo_logger.propagate = False

    # echo.llm logger (LLM traffic)
    llm_logger = logging.getLogger("echo.llm")
    llm_logger.setLevel(logging.DEBUG)
    llm_logger.handlers.clear()
    llm_logger.addHandler(llm_handler)          # llm.log (DEBUG+)
    llm_logger.propagate = False

    logger = logging.getLogger("echo")
    logger.info("Starting ECHO...")

    try:
        CONTEXT_WARN_THRESHOLD = float(os.getenv("CONTEXT_WARN_THRESHOLD", "0.90"))
        if not (0 < CONTEXT_WARN_THRESHOLD < 1):
            print("⚠️  Invalid CONTEXT_WARN_THRESHOLD in .env, using default 0.90")
            CONTEXT_WARN_THRESHOLD = 0.90
    except:
        print("⚠️  Failed to parse CONTEXT_WARN_THRESHOLD, using default 0.90")
        CONTEXT_WARN_THRESHOLD = 0.90

    # ---------------------------------------
    # Start toolkit
    # ---------------------------------------
    toolkit = BaseToolkit()
    if not toolkit.openai:
      raise Exception('OpenAI API not initialized')
    # Turn audio off for console I/O:
    # toolkit.toggleTool('listen','disabled')
    # toolkit.toggleTool('speak','disabled')
    mainLoop(toolkit)
