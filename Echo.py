import os
import json
import traceback
from timeit import default_timer as timer
import logging
from dotenv import load_dotenv

#Own
from Toolkit import BaseToolkit

def modelOne(toolkit, messages):
  ts_s = timer()
  logger = logging.getLogger("echo.modelOne.LLM")

  try:
    logger.info("LLM Request: model=%s", toolkit.openai_chat_model)
    logger.debug("LLM Request Messages: %s", json.dumps(messages, indent=2, ensure_ascii=False))
    logger.debug("LLM Tools Spec: %s", json.dumps(toolkit.toolMessage(), indent=2, ensure_ascii=False))
  except Exception:
    logger.exception("Failed to log LLM request.")
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
  try:
    logger.info("LLM Response received.")
    logger.debug("LLM Raw Response JSON: %s", res.model_dump_json(indent=2))
  except Exception:
    logger.exception("Failed to log LLM response.")
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
  messages = [{
    "role": "system",
    "content": f"""
      You are a helpful assistant called ECHO.
      Based on user request and available functions devise a plan of action and execute it.
      Keep in mind multiple data sources are available. If you are unable to fulfil the request with one data source, try again with another.
      A demonstrative pronoun such as this/that/these/it likely refers to something in conversation history, or data copied to cliboard or something that user sees on his screen.
      Regardless of action taken, respond in JSON with {{plan:<plan>,response:<text response>}}
    """ + toolkit.toolPrompt()
    }] + sum(history, []) + [{"role":"user", "content":toolkit.userPrompt()}] + toolkit.fake('listTools') + toolkit.fake('clipboardRead')
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

  return loopBehav

def mainLoop(toolkit, limit=10):
  history = []

  helpText = (
    "Type 'history' to see conversation history. \n"
    "Type 'clear' to clear history. \n"
    "Type 'reset' to reset all tools. \n\n"
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

  load_dotenv()

  log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
  print(log_level_name)
  log_level = getattr(logging, log_level_name, logging.INFO)

  logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
      logging.FileHandler(os.path.join(LOG_DIR, "echo.log")),
      logging.StreamHandler()
    ]
  )

  logger = logging.getLogger("echo")
  logger.info("Starting ECHO...")

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
