import os
import json
import traceback
from timeit import default_timer as timer
import logging
from dotenv import load_dotenv

#Own
from Toolkit import BaseToolkit
from echo_config import init_logging_and_ws, MODEL_CONTEXT_LIMITS, DEFAULT_CONTEXT_LIMIT
from echo_cli import promptOption, helpText

def estimate_tokens_from_messages(messages):
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            try:
                total_chars += len(str(content))
            except Exception:
                pass
    return int(total_chars / 3.5)

def modelOne(toolkit, messages):
    logger = logging.getLogger("echo.llm")
    trace = logging.getLogger("echo.trace")
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
    #trace.info("ACTION: Sending prompt to LLM (model=%s).", toolkit.openai_chat_model)
    trace.info("ACTION: Sending prompt to LLM.")

    res = toolkit.openai.chat.completions.create(
        model    = toolkit.openai_chat_model,
        messages = messages,
        tools    = toolkit.toolMessage(),
        tool_choice = "auto"
    )

    ts_e = timer()
    #trace.info("ACTION: LLM responded with finish_reason='%s'.", reason)
    trace.info("ACTION: LLM responded with finish_reason.")
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
  trace = logging.getLogger("echo.trace")

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
      max_tokens = MODEL_CONTEXT_LIMITS.get(
          toolkit.openai_chat_model,
          DEFAULT_CONTEXT_LIMIT,
      )
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
    #trace.info("ACTION: Starting new LLM turn with %d history messages.", len(history_messages))
    trace.info("ACTION: Starting new LLM turn.")
    reason, content, messages = modelOne(toolkit, messages)
    if reason == "stop":
      break

  history.append(messages)
  return content, history

def mainLoop(toolkit, limit=10):
  history = []

  print("Welcome to ECHO! Deep dive into my power. \n "
        " ------------------ \n"
        f"{helpText}")

  while True:
    try:
      print(">> ", end="")
      prompt = toolkit.input()

      lOps = promptOption(prompt, history, toolkit)
      if lOps == "break":
        break
      elif lOps == "continue":
        continue
      elif lOps == "test_vuln":
        # For TestCmd we already set toolkit.data.prompt to the fixed test string
        print(f"User input (TestCmd): {toolkit.userPrompt()}")
      else:
        # Normal behavior: use what user actually typed
        print(f"User input: {prompt}")

      content, history = modelLoop(toolkit, history)
      history = history[:limit]
      print(content)
    except Exception as e:
      traceback.print_exc()
      pass

if __name__ == "__main__":
    # All logging + WS setup + CONTEXT_WARN_THRESHOLD comes from config helper
    CONTEXT_WARN_THRESHOLD = init_logging_and_ws()

    # ---------------------------------------
    # Start toolkit
    # ---------------------------------------
    toolkit = BaseToolkit()
    if not toolkit.openai:
        raise Exception('OpenAI API not initialized')

    # Turn audio off for console I/O if you want:
    if os.getenv("ENABLE_LISTEN", "false").lower() == "false":
        toolkit.toggleTool('listen', 'disabled')
    if os.getenv("ENABLE_SPEAK", "false").lower() == "false":
        toolkit.toggleTool('speak', 'disabled')


    mainLoop(toolkit)
