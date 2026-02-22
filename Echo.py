import os
import json
import traceback
from timeit import default_timer as timer
import logging
from dotenv import load_dotenv

#Own
from Toolkit import BaseToolkit, FullToolkit
from echo_config import (
    init_logging_and_ws,
    MODEL_CONTEXT_LIMITS,
    DEFAULT_CONTEXT_LIMIT,
)
from echo_cli import promptOption, shortHelpText, normalize_llm_output, execute_sequence

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
        logger.info("LLM Request: model=%s", toolkit.chat_model)
        logger.debug("LLM Request Messages: %s", json.dumps(messages, indent=2, ensure_ascii=False))
        logger.debug("LLM Tools Spec: %s", json.dumps(toolkit.toolMessage(), indent=2, ensure_ascii=False))
    except Exception:
        logger.exception("Failed logging LLM request")
        traceback.print_exc()
    print("Prompting...")
    #trace.info("ACTION: Sending prompt to LLM (model=%s).", toolkit.chat_model)
    trace.info("ACTION: Sending prompt to LLM.")

    llm_res = toolkit.llm_call(
        messages=messages,
        tools=toolkit.toolMessage(),
        tool_choice="auto"
    )

    ts_e = timer()
    #trace.info("ACTION: LLM responded with finish_reason='%s'.", reason)
    trace.info("ACTION: LLM responded with finish_reason.")
    print(f"... took {ts_e-ts_s}s")

    res = llm_res["raw"]

    # Log response
    try:
        logger.info("LLM Response received.")
        logger.debug("LLM Raw Response JSON: %s", res.model_dump_json(indent=2))
    except Exception:
        logger.exception("Failed logging LLM response")
        traceback.print_exc()

    reason = llm_res["finish_reason"]
    message = llm_res["message"]
    backend = llm_res["backend"]

    if reason == "stop":
        if backend == "completions":
            messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'tool_calls'})))
            content = message.content
        else:  # responses backend
            messages.append({
                "role": message.role,
                "content": message.content,
            })
            content = message.content

        return reason, content, messages

    if reason == "tool_calls" and backend == "completions":
        messages.append(json.loads(message.model_dump_json(exclude={'function_call', 'content'})))
        for tc in message.tool_calls:
            if tc.type == "function":
                messages.append(toolkit.call(tc.id, tc.function))

        return reason, None, messages

    # Responses currently doesn't do client-side tool chaining in your code,
    # so we just treat any non-stop as stop to avoid loops:
    return "stop", getattr(message, "content", None), messages

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
          toolkit.chat_model,
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

  if isinstance(content, str):
      content = normalize_llm_output(content)

      if getattr(toolkit, "redact_mode", False):
        try:
          content = toolkit._redact_text(content)
        except Exception:
          logging.getLogger("echo").exception("Failed to redact output text")

  history.append(messages)
  return content, history

def mainLoop(toolkit, limit=10):
  history = []

  print("Welcome to ECHO! Deep dive into my power. \n "
        " ------------------ \n"
        f"{shortHelpText}")

  prof_key = getattr(toolkit, "current_model_profile", "current")
  prof_label = prof_key.capitalize()  # "current" -> "Current", "legacy" -> "Legacy"

  # Helper function to get provider display name
  def get_provider_display(model_name):
    if model_name in toolkit.model_provider_map:
      provider_name = toolkit.model_provider_map[model_name].get("providerName", "")
      if provider_name in toolkit.llm_providers:
        return toolkit.llm_providers[provider_name].get("name", provider_name)
      return provider_name
    return "default"

  print(f"Active model profile: {prof_label}")
  print(f"  chat    : {toolkit.chat_model} [{get_provider_display(toolkit.chat_model)}]")
  print(f"  vision  : {toolkit.vision_model} [{get_provider_display(toolkit.vision_model)}]")
  print(f"  research: {toolkit.research_model} [{get_provider_display(toolkit.research_model)}]")
  print(f"  stt     : {toolkit.stt_model} [{get_provider_display(toolkit.stt_model)}]")
  print(f"Backend: {toolkit.llm_backend} | Providers loaded: {len(toolkit.llm_providers)}")


  # Sequence state tracking
  sequence_state = None

  while True:
    try:
      # If we're in sequence mode and waiting after LLM response
      if sequence_state:
        seq_name, step_num, total_steps, auto_exec = sequence_state
        is_last_step = (step_num == total_steps)

        if is_last_step:
          print(f"\n{'='*60}")
          print(f"⚠️  Step {step_num}/{total_steps} completed.")
          print(f"{'='*60}")
          print("\n✅ Sequence completed successfully!\n")
          sequence_state = None
          continue

        # Continue with next step (showing previous completion)
        sequence_state = None
        lOps = execute_sequence(seq_name, toolkit, history, start_from=step_num + 1, show_prev_completed=True)

        if lOps == "break":
          break
        elif lOps == "continue":
          continue
        elif isinstance(lOps, tuple) and lOps[0] == "sequence_exec":
          # Another LLM step
          _, seq_name, seq_cmd, step_num, total_steps, auto_exec = lOps

          # Set user prompt using toolkit method (handles redaction expansion)
          toolkit.set_prompt(seq_cmd)
          print(f"User input: {seq_cmd}")

          sequence_state = (seq_name, step_num, total_steps, auto_exec)
          content, history = modelLoop(toolkit, history)
          history = history[:limit]
          print(content)
        continue

      prompt = toolkit.input(">> ")

      lOps = promptOption(prompt, history, toolkit)
      if lOps == "break":
        break
      elif lOps == "continue":
        continue
      elif lOps == "test_vuln":
        print(f"User input (TestCmd): {toolkit.userPrompt()}")
      elif isinstance(lOps, tuple) and lOps[0] == "sequence_exec":
        # Sequence is executing an LLM command
        _, seq_name, seq_cmd, step_num, total_steps, auto_exec = lOps

        # Set user prompt using toolkit method (handles redaction expansion)
        toolkit.set_prompt(seq_cmd)
        print(f"User input: {seq_cmd}")

        # Store sequence state for after LLM response
        sequence_state = (seq_name, step_num, total_steps, auto_exec)

        # Execute LLM
        content, history = modelLoop(toolkit, history)
        history = history[:limit]
        print(content)
        continue
      else:
        print(f"User input: {prompt}")

      content, history = modelLoop(toolkit, history)
      history = history[:limit]
      print(content)

    except KeyboardInterrupt:
      print("\n^C – interrupted. Goodbye!")
      break

    except Exception:
      traceback.print_exc()
      pass

if __name__ == "__main__":
    CONTEXT_WARN_THRESHOLD = init_logging_and_ws()

    # ---------------------------------------
    # Start toolkit
    # ---------------------------------------
    toolkit = FullToolkit()

    # Check if providers for current profile models are initialized
    profile_models = [
        toolkit.chat_model,
        toolkit.vision_model,
        toolkit.research_model,
        toolkit.stt_model,
    ]

    missing_providers = []
    for model in profile_models:
        client, provider_info, _ = toolkit._get_client_for_model(model)
        if not client:
            # Try to resolve model identifier to get provider name
            _, provider_name = toolkit._resolve_model_identifier(model)
            if not provider_name:
                provider_name = "unknown"
            missing_providers.append(f"{model} (provider: {provider_name})")

    if missing_providers:
        print("⚠️  WARNING: Some models in the current profile are not properly configured:")
        for m in missing_providers:
            print(f"   - {m}")
        print("\nPlease check your LLM_PROVIDERS configuration in .env and ensure API keys are set.")
        raise Exception('LLM providers not properly initialized for current profile')

    if os.getenv("ENABLE_SPEAK", "false").lower() == "false":
        toolkit.toggleTool('speak', 'disabled')
    if os.getenv("ENABLE_CLIPBOARD", "false").lower() == "false":
        toolkit.toggleTool('clipboardRead', 'disabled')
        toolkit.toggleTool('clipboardWrite', 'disabled')

    mainLoop(toolkit)
