import inspect
import json
import secrets
import traceback
from types import ModuleType
from openai import OpenAI
from dotenv import load_dotenv
from timeit import default_timer as timer
import logging

# Tool imports
import shodan
import time
import os
import webbrowser
import threading
import pytesseract
import pyperclip
import pyttsx3
import base64
import pywinctl as pwc
import pyautogui
import serpapi
import arxiv
import urllib
import urllib.parse
from playsound3 import playsound
import speech_recognition as sr
import mss
import mss.tools
from PIL import ImageGrab, Image
from io import BytesIO
import subprocess
import shlex
import requests
import pyxploitdb

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def genToolspec(name, desc, args={}, reqs=[], **kwargs):
  # openAI tool_calls specification json
  # TODO: validate vs schema
  return {
    'type': 'function',
    'function': {
      'name': name,
      'description': desc,
      "parameters": {
        "type": "object",
        "properties": args,
        "required": reqs
      }
    }
  }
def toolspec(**kwargs):
  def decorator(func):
    if not hasattr(func, '_toolspec'):
      func._toolspec = AttrDict()
    source = kwargs.get('source')
    if source is None:
      try:
        source = inspect.getsource(func)
      except:
        pass
    func._toolspec = AttrDict({
      'state'    : kwargs.get('state',"enabled"),
      'function' : func, 
      'spec'     : genToolspec(name = func.__name__, **kwargs),
      'source'   : source,
      'prompt'   : kwargs.get('prompt',"")
    })
    return func
  return decorator
def b64(img):
  if isinstance(img, Image.Image):
    with BytesIO() as buf:
      img.save(buf, format="PNG")
      return base64.b64encode(buf.getvalue()).decode('utf-8')
  with open(img, "rb") as f:
    return base64.b64encode(f.read()).decode('utf-8')

class BaseCoreToolkit:
  """
  Internal core:
  - env / OpenAI / profiles
  - tool registry + decorator integration
  - generic tool management methods (call, fake, addTool, etc.)
  """

  def __init__(self):
    # Core state
    self.data = AttrDict()
    self.module = ModuleType("DynaToolKit")
    self._toolspec = AttrDict()
    self.logger = logging.getLogger(f"echo.toolkit.{self.__class__.__name__}")
    self.trace = logging.getLogger("echo.trace")
    self.echo_toolkit = logging.getLogger("echo.toolkit")

    # Load .env
    load_dotenv()

    # --- API keys ---
    self.shodan_api_key = os.getenv("SHODAN_API_KEY", "Missing Key")
    self.nvd_api_key = os.getenv("NVD_API_KEY")

    # --- OpenAI config from .env ---
    self.openai_api_key = os.getenv("OPENAI_API_KEY")
    self.openai_base_url = os.getenv("OPENAI_BASE_URL")
    self.llm_backend = os.getenv("OPENAI_LLM_BACKEND", "completions").lower()

    # --- Base model values (used for 'current' profile by default) ---
    default_chat = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    default_vision = os.getenv("OPENAI_VISION_MODEL", "gpt-5.0")
    default_research = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.1")
    default_stt = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

    # Store active values (will be overridden by profile)
    self.openai_chat_model = default_chat
    self.openai_vision_model = default_vision
    self.openai_research_model = default_research
    self.openai_stt_model = default_stt

    # --- Named model profiles (template sets) ---
    self.model_profiles = {
      "legacy": {
        "chat": os.getenv("OPENAI_CHAT_MODEL_LEGACY", "gpt-4-turbo-preview"),
        "vision": os.getenv("OPENAI_VISION_MODEL_LEGACY", "gpt-4-vision-preview"),
        "research": os.getenv("OPENAI_RESEARCH_MODEL_LEGACY", "gpt-4-turbo-preview"),
        "stt": os.getenv("OPENAI_STT_MODEL_LEGACY", "whisper-1"),
      },
      "current": {
        "chat": os.getenv("OPENAI_CHAT_MODEL_CURRENT", default_chat),
        "vision": os.getenv("OPENAI_VISION_MODEL_CURRENT", default_vision),
        "research": os.getenv("OPENAI_RESEARCH_MODEL_CURRENT", default_research),
        "stt": os.getenv("OPENAI_STT_MODEL_CURRENT", default_stt),
      },
    }

    # Select starting profile
    self.current_model_profile = os.getenv("OPENAI_MODEL_PROFILE", "current").lower()

    # OpenAI client
    client_kwargs = {}
    if self.openai_api_key:
      client_kwargs["api_key"] = self.openai_api_key
    if self.openai_base_url:
      client_kwargs["base_url"] = self.openai_base_url

    if client_kwargs:
      self.openai = OpenAI(**client_kwargs)
    else:
      self.openai = None

    # Apply selected profile at startup
    try:
      self._apply_model_profile(self.current_model_profile)
    except Exception:
      self.logger.exception("Failed to apply model profile '%s'", self.current_model_profile)

    # Discover all @toolspec-decorated methods on *this instance*
    for name in dir(self):
      func = getattr(self, name)
      if not callable(func):
        continue
      if not hasattr(func, "_toolspec"):
        continue
      func._toolspec.function = func
      self._toolspec[name] = func._toolspec

  #
  # --- Tool management / decorator integration ---
  #
  def toolspecBySrc(self, src, context=""):
    # Generates OpenAI tool specs from source code using the current backend
    if not self.openai:
      raise Exception("Model-assisted functions unavailable")

    system_prompt = f"""
    A Function description is an object describing a function and its arguments.
    It consists of 3 elements:
      1. name: function name
      2. description: a short (2 sentences max) description of what the function does.
      3. arguments: an argument description.
    An argument description is: {{name:<name>, type:<type>, description:<description>}}
    <type> must be one of: number/integer/string
    If function requires ApiKey, ApiKey should be compatible with setApiKey tool.

    Generate function descriptions for each function in the source code shown below.
    Answer as JSON: {{"functions":[{{"name":<name>, "description":<description>, "args":[{{"name":..., "type":..., "description":...}}, ...]}}, ...]}}
    

    <code>
    {src}
    </code>
    <context>
    {context}
    </context>
    """

    llm_res = self.llm_call(
      messages=[{"role": "system", "content": system_prompt}],
      response_format={"type": "json_object"},
      tool_choice="none"  # ignored for responses, fine for chat
    )
    raw = llm_res["raw"]

    # Normalize content extraction for both backends
    if self.llm_backend == "chat":
      content = raw.choices[0].message.content

    elif self.llm_backend == "responses":
      first_step = raw.output[0]
      text_block = next(
        c for c in first_step.content
        if getattr(c, "type", None) in ("output_text", "message", None)
      )
      content = getattr(text_block, "text", str(text_block))

    else:
      raise ValueError(f"Unknown llm_backend: {self.llm_backend}")

    descs = json.loads(content)["functions"]

    tools = []
    for desc in descs:
      args = {}
      reqs = []
      for a in desc["args"]:
        args[a["name"]] = {
          "type": "string",
          "description": a["description"],
        }
        reqs.append(a["name"])
      tools.append(genToolspec(desc["name"], desc["description"], args, reqs))

    return tools

  def addTool(self, func, spec, source=None, prompt=""):
    dec = toolspec(
      desc=spec['function']['description'],
      args=spec['function']['parameters']['properties'],
      reqs=spec['function']['parameters']['required'],
      source=source,
      prompt=prompt
    )
    dec(func)
    self._toolspec[func.__name__] = func._toolspec
    return "{status: success}"

  def addToolByRef(self, func):
    # Registers a function by reference
    src = inspect.getsource(func)
    spec = self.toolspecBySrc(src)[0]
    return self.addTool(func, spec, src)

  def toolPrompt(self):
    prompt = ""
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        prompt += tool.prompt
    return prompt

  def toolMessage(self):
    # Generates tool_calls table
    msgs = []
    for k in self._toolspec:
      tool = self._toolspec[k]
      if tool.state == "enabled":
        msgs.append(tool.spec)
    return msgs

  def call(self, cid, func):
    ts_s = timer()
    self.logger.info("Tool call requested: %s", func.name)
    self.trace.info(
      f"ACTION: LLM selected tool '{func.name}' (tool_call_id={cid}) (args={getattr(func, 'arguments', None)})"
    )

    res = "Error: Unknown error."

    if func.name not in self._toolspec:
      res = "Error: Function not found."
      self.logger.error("Tool %s not found", func.name)
      self.trace.warning("ACTION: Tool '%s' not found", func.name)
    elif self._toolspec[func.name].state == "disabled":
      res = "Error: Function is disabled."
      self.logger.warning("Tool %s is disabled", func.name)
      self.trace.info("ACTION: Tool '%s' is disabled, skipping call", func.name)
    else:
      try:
        args = json.loads(func.arguments)
        self.logger.info("Calling tool %s with args=%s", func.name, args)
        self.trace.info("ACTION: Calling tool '%s' with args=%s", func.name, args)
        self.echo_toolkit.info("Tool %s Input Args:\n %s", func.name, args)
        res = self._toolspec[func.name].function(**args)
        self.logger.info("Tool %s completed successfully.", func.name)
        self.trace.info("ACTION: Tool '%s' completed.", func.name)
      except Exception as e:
        res = f"Error: <backtrace>\n{traceback.format_exc()}\n</backtrace>"
        self.logger.error("Tool %s raised exception: %s", func.name, e)
        self.trace.error("ACTION: Tool '%s' raised exception: %s", func.name, e)
        print(res)

    ts_e = timer()
    self.logger.info("Tool %s finished in %.3fs", func.name, ts_e - ts_s)
    self.trace.info("ACTION: Tool '%s' finished in %.3fs", func.name, ts_e - ts_s)
    print(f"... took {ts_e - ts_s}s")

    output = {
      "role": "tool",
      "tool_call_id": cid,
      "name": func.name,
      "content": json.dumps({"result": res})
    }

    self.echo_toolkit.info("Tool %s Output:\n %s", func.name, output)
    return output

  def fake(self, name, args='{}'):
    # Fake a tool call. Saves a model call while preserving context flow.
    func = AttrDict({'name': name, 'arguments': args})
    cid = f"call_{secrets.token_urlsafe(24)}"
    res = self.call(cid, func)
    return [{
      'role': 'assistant',
      'tool_calls': [{
        'id': cid,
        'function': {
          'arguments': args,
          'name': name
        },
        'type': 'function'
      }],
    }, res]

  @toolspec(
    desc="List toolkit functions and their current state. "
         "Mode can be 'disabled' (default), 'enabled', or 'all'.",
    args={
      "mode": {
        "type": "string",
        "description": "Filter mode: 'disabled' (default), 'enabled', or 'all'."
      }
    },
    reqs=[]
  )
  def listTools(self, mode="disabled"):
    tools = []
    mode = (mode or "disabled").lower()

    for name, tool in self._toolspec.items():
      state = tool.state

      if mode == "disabled" and state != "disabled":
        continue
      if mode == "enabled" and state != "enabled":
        continue
      # mode == "all" → no filter

      tools.append({
        "name": name,
        "description": tool.spec["function"]["description"],
        "state": state,
      })

    return tools

  @toolspec(
    desc="Toggles tool state: enabled/disabled. Disabled tools are not added to tool_calls, saving tokens",
    args={
      "name": {"type": "string", "description": "Tool name to toggle"},
      "state": {"type": "string", "description": "One of: enabled/disabled"}
    },
    reqs=["name", "state"]
  )
  def toggleTool(self, name, state):
    if name not in self._toolspec:
      return f"{{status: error, error:{name} not found}}"
    self._toolspec[name].state = state
    return "{status: success}"

  @toolspec(
    desc="Adds functions defined by Python source code to the toolkit. This should only be used if user explicitly asked to add a function to toolkit.",
    args={"src": {"type": "string", "description": "Python source code of functions to be added to toolkit"}},
    reqs=["src"]
  )
  def addToolBySrc(self, src):
    # Registers a function by source code
    logs = ""
    code = compile(src, self.module.__name__, 'exec')
    specs = self.toolspecBySrc(src)
    exec(code, self.module.__dict__)
    for spec in specs:
      print(spec)
      name = spec['function']['name']
      func = getattr(self.module, name)
      logs += self.addTool(func, spec, src)
    return logs

  #
  # --- API keys & profiles (core) ---
  #
  def _update_env_var(self, key, value):
    os.environ[key] = value
    env_path = os.path.join(os.getcwd(), ".env")
    try:
      lines = []
      if os.path.exists(env_path):
        with open(env_path, "r") as f:
          lines = f.read().splitlines()

      prefix = key + "="
      found = False
      new_lines = []
      for line in lines:
        if line.startswith(prefix):
          new_lines.append(f"{key}={value}")
          found = True
        else:
          new_lines.append(line)

      if not found:
        new_lines.append(f"{key}={value}")

      with open(env_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

      return True
    except Exception as e:
      print(f"Warning: could not persist {key} to .env: {e}")
      return False

  def _tools_for_responses(self, tools):
    if not tools:
      return []

    converted = []
    for t in tools:
      if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
        fn = t["function"] or {}
        converted.append({
          "type": "function",
          "name": fn.get("name"),
          "description": fn.get("description", ""),
          "parameters": fn.get("parameters", {
            "type": "object",
            "properties": {},
            "required": [],
          }),
        })
      else:
        converted.append(t)
    return converted

  def _messages_to_responses_input(self, messages):
    input_items = []

    for m in messages:
      if not isinstance(m, dict):
        continue

      role = m.get("role")

      # /responses does NOT accept role "tool"
      if role == "tool":
        continue

      # Skip assistant stub that only carries tool_calls
      if role == "assistant" and m.get("tool_calls"):
        continue

      content = m.get("content", "")

      if isinstance(content, str):
        text = content
      elif isinstance(content, list):
        parts = []
        for part in content:
          if isinstance(part, dict) and "text" in part:
            parts.append(part["text"])
        text = "\n".join(parts) if parts else ""
      else:
        text = str(content)

      if role not in ("system", "user", "assistant", "developer"):
        role = "user"

      if role == "assistant":
        ctype = "output_text"
      else:
        ctype = "input_text"

      input_items.append({
        "role": role,
        "content": [{
          "type": ctype,
          "text": text,
        }],
      })

    return input_items

  @toolspec(
    desc="Set or update API keys for external services (OpenAI, Shodan, SerpAPI).",
    args={
      "service": {
        "type": "string",
        "description": "Service name: 'openai', 'shodan', or 'serpapi'."
      },
      "api_key": {
        "type": "string",
        "description": "API key or token for the given service."
      }
    },
    reqs=["service", "api_key"]
  )
  def setApiKey(self, service, api_key):
    svc = service.lower()
    persisted = False

    if svc == "openai":
      self.openai_api_key = api_key
      persisted = self._update_env_var("OPENAI_API_KEY", api_key)
      client_kwargs = {"api_key": api_key}
      if getattr(self, "openai_base_url", None):
        client_kwargs["base_url"] = self.openai_base_url
      self.openai = OpenAI(**client_kwargs)

    elif svc == "shodan":
      self.shodan_api_key = api_key
      persisted = self._update_env_var("SHODAN_API_KEY", api_key)
      if hasattr(self, "shodan"):
        self.shodan = shodan.Shodan(api_key)

    elif svc == "serpapi":
      persisted = self._update_env_var("SERPAPI_API_KEY", api_key)
      if hasattr(self, "serpapi"):
        self.serpapi = serpapi.Client(api_key=api_key)

    else:
      return json.dumps({
        "status": "error",
        "error": f"Unknown service '{service}'. Use one of: openai, shodan, serpapi."
      })

    return json.dumps({
      "status": "success",
      "service": svc,
      "persisted": persisted
    })

  def _apply_model_profile(self, profile_name: str) -> bool:
    profile_key = profile_name.strip().lower()
    if profile_key not in self.model_profiles:
      self.logger.warning("Unknown model profile '%s'", profile_name)
      return False

    prof = self.model_profiles[profile_key]

    if "chat" in prof:
      self.openai_chat_model = prof["chat"]
    if "vision" in prof:
      self.openai_vision_model = prof["vision"]
    if "research" in prof:
      self.openai_research_model = prof["research"]
    if "stt" in prof:
      self.openai_stt_model = prof["stt"]

    self.current_model_profile = profile_key
    self.logger.info(
      "Switched model profile to '%s' (chat=%s, vision=%s, research=%s, stt=%s)",
      profile_key,
      self.openai_chat_model,
      self.openai_vision_model,
      self.openai_research_model,
      self.openai_stt_model,
    )
    return True

  @toolspec(
    desc="Switch between predefined LLM model profiles (e.g. 'legacy', 'current'). "
         "Each profile sets chat/vision/research/STT models as a bundle.",
    args={
      "profile": {
        "type": "string",
        "description": "Name of the model profile to activate, e.g. 'legacy' or 'current'."
      }
    },
    reqs=["profile"]
  )
  def setModelProfile(self, profile):
    ok = self._apply_model_profile(profile)
    if not ok:
      return {
        "status": "error",
        "error": f"Unknown profile '{profile}'. Available: {list(self.model_profiles.keys())}"
      }
    return {
      "status": "success",
      "profile": self.current_model_profile,
      "chat_model": self.openai_chat_model,
      "vision_model": self.openai_vision_model,
      "research_model": self.openai_research_model,
      "stt_model": self.openai_stt_model,
    }

  def llm_call(self, messages, tools=None, tool_choice="auto", **kwargs):
    """
    Unified LLM call.

    Returns:
      {
        "backend": "chat" | "responses",
        "raw": <raw SDK response>,
        "finish_reason": "stop" | "tool_calls" | <other> | None,
        "message": <primary assistant message-like object>,
      }
    """
    if not getattr(self, "openai", None):
      raise RuntimeError("OpenAI client not initialized")

    backend = getattr(self, "llm_backend", "chat").lower()
    tools = tools if tools is not None else self.toolMessage()

    if "tools" in kwargs:
      if not tools:
        tools = kwargs["tools"]
      kwargs.pop("tools")

    # ---------------- CHAT COMPLETIONS ----------------
    if backend == "completions":
      res = self.openai.chat.completions.create(
        model=self.openai_chat_model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
      )
      choice = res.choices[0]
      return {
        "backend": "chat",
        "raw": res,
        "finish_reason": choice.finish_reason,
        "message": choice.message,
      }

    # ---------------- RESPONSES ----------------
    if backend == "responses":
        # /responses ignores tool_choice, so drop it
        kwargs.pop("tool_choice", None)

        input_items = self._messages_to_responses_input(messages)
        tools_for_responses = self._tools_for_responses(tools)

        res = self.openai.responses.create(
            model=self.openai_chat_model,
            input=input_items,
            tools=tools_for_responses,
            **kwargs,
        )

        # --------- extract text in a robust way ----------
        text = None
        # 1) Preferred: use the convenience attribute if present
        text = getattr(res, "output_text", None)

        # 2) Fallback: manually walk res.output
        if not text:
            chunks = []
            output = getattr(res, "output", None) or []
            for item in output:
                # We only care about message items, skip reasoning, etc.
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if hasattr(c, "text") and c.text:
                            chunks.append(c.text)
            text = "\n".join(chunks) if chunks else ""

        # --------- normalize stop_reason to your old semantics ----------
        finish_reason = "stop"   # sensible default
        output = getattr(res, "output", None) or []
        if output:
            first = output[0]
            raw_reason = getattr(first, "stop_reason", None)
            if raw_reason == "tool_use":
                finish_reason = "tool_calls"
            else:
                # end_turn, max_output, etc → treat as stop
                finish_reason = "stop"

        # Build assistant-like message wrapper so modelOne can use it
        from types import SimpleNamespace as AttrDict
        message_like = AttrDict({
            "role": "assistant",
            "content": text,
        })

        return {
            "backend": "responses",
            "raw": res,
            "finish_reason": finish_reason,
            "message": message_like,
        }

    raise ValueError(f"Unknown llm_backend: {backend!r}")

#
# BaseToolkit (system toolkit: decorator mgmt + IO + console + clipboard, etc.)
#

class BaseToolkit(BaseCoreToolkit):
  def __init__(self):
    super().__init__()
    # System-level extras
    self.data.stt = None
    self.shodan = shodan.Shodan(self.shodan_api_key)
    self.serpapi = serpapi.Client()
    self.chain_enabled = True

  @toolspec(
    desc="Get input from console. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, text, console",
    args={},
    reqs=[]
  )
  def read(self):
    self.trace.info("ACTION: Reading text from console (input()).")
    return input()

  def input(self):
    text = None
    if 'listen' in self._toolspec and self._toolspec.listen.state == "enabled":
      self.trace.info("ACTION: Listening to your voice (speech-to-text).")
      self.listen()
      text = self.stt()
      self.trace.info("ACTION: Transcribed your voice input.")
    else:
      self.trace.info("ACTION: Waiting for your console text input.")
      text = self.read()
      self.trace.info("ACTION: Received your text input from console.")

    self.data.prompt = text
    self.data.screenshot = None
    self.data.clipboard = None


    if 'clipboardRead' in self._toolspec and self._toolspec.clipboardRead.state == "enabled":
      self.trace.info("ACTION: Reading clipboard snapshot.")
      self.clipboardRead()

    if 'screenshot' in self._toolspec and self._toolspec.screenshot.state == "enabled":
      self.trace.info("ACTION: Capturing screenshot snapshot.")
      self.screenshot()

    return text

  def userPrompt(self):
    return self.data.prompt

  def reset(self):
    print("Resetting Toolkit")

  #
  # System "web-ish" tools
  #
  @toolspec(
    desc="Open URL in default web browser. Can be a local path with file:/// URL",
    args={"url": {"type": "string", "description": "URL to be opened"}},
    reqs=["url"]
  )
  def browse(self, url):
    webbrowser.open(url, new=2)
    return "{status: success}"

  @toolspec(
    desc="Downloads file from URL. Returns local path of downloaded file.",
    args={
      "url": {"type": "string", "description": "File to download"},
      "filename": {"type": "string", "description": "Optional filename/path to save as."}
    },
    reqs=["url"]
  )
  def download(self, url, filename=None):
    file, _ = urllib.request.urlretrieve(url, filename)
    return f'{{"status": "success", "file": {json.dumps(file)}}}'

  @toolspec(
    desc="Search the Internet. Returns top 10 results: {url, title, description}",
    args={
      "phrase": {"type": "string", "description": "Phrase to search for"},
      "limit": {"type": "integer", "description": "Number of results. Default: 10"}
    },
    reqs=["phrase"],
    state="disabled",
  )
  def webSearch(self, phrase, limit=10):
    res = self.serpapi.search({'engine': 'google', 'q': phrase})
    arr = [
      {'url': r['link'], 'title': r['title'], 'description': r['snippet']}
      for r in res.get('organic_results', [])[:limit]
    ]
    return json.dumps({"status": "success", "content": arr})

  #
  # Clipboard & misc system controls
  #
  @toolspec(
    desc="Write text into users clipboard. Should be used to output code, json, csv, commands to run, or data to fill a form. Category: output, text, copy-paste",
    args={"text": {"type": "string", "description": "Text to be written into clipboard"}},
    reqs=["text"]
  )
  def clipboardWrite(self, text):
    self.trace.info("ACTION: Writing text to your clipboard.")
    pyperclip.copy(text)
    return "{status: success}"

  @toolspec(
    desc="Read contents of users clipboard. Returns {status:<status>, type:<type of content>, content: <text content>}. Category: input, text, copy-paste",
    args={},
    reqs=[]
  )
  def clipboardRead(self):
    self.trace.info("ACTION: Attempting to read your clipboard.")
    img = None

    # Try image clipboard first
    try:
      img = ImageGrab.grabclipboard()
    except NotImplementedError as e:
      print(f"Image clipboard not supported on this system: {e}")
    except Exception as e:
      print(f"Error grabbing image from clipboard: {e}")

    if isinstance(img, Image.Image):
      self.data.clipboard = img
      return '{"status": "success", "type": "image"}'

    # Fallback to text via pyperclip
    try:
      text = pyperclip.paste()
      self.data.clipboard = text
      return '{"status": "success", "type": "text", "content": ' + json.dumps(text) + '}'
    except Exception as e:
      print(f"Clipboard text read failed: {e}")
      return '{"status": "error", "reason": "Clipboard not accessible"}'

  @toolspec(
    desc="Change which OpenAI model the toolkit uses at runtime.",
    args={
      "target": {
        "type": "string",
        "description": "Which model to change: one of 'chat', 'vision', 'research', 'stt'."
      },
      "model": {
        "type": "string",
        "description": "New OpenAI model name, e.g. 'gpt-4.1-mini' or 'gpt-4.1'."
      }
    },
    reqs=["target", "model"]
  )
  def setOpenAIModel(self, target, model):
    target_l = target.lower()
    if target_l == "chat":
      self.openai_chat_model = model
    elif target_l == "vision":
      self.openai_vision_model = model
    elif target_l == "research":
      self.openai_research_model = model
    elif target_l == "stt":
      self.openai_stt_model = model
    else:
      return json.dumps({
        "status": "error",
        "error": f"Unknown target '{target}'. Use one of: chat, vision, research, stt."
      })

    return json.dumps({
      "status": "success",
      "target": target_l,
      "model": model
    })

  @toolspec(
    desc="Change logging level at runtime. Useful to increase or decrease verbosity (debug/info/warning/error/critical).",
    args={
      "level": {
        "type": "string",
        "description": "New log level: one of 'debug', 'info', 'warning', 'error', 'critical'."
      },
      "logger_name": {
        "type": "string",
        "description": "Optional logger name, default 'echo'. For fine-grained control you can use 'echo.llm' or 'echo.toolkit.BaseToolkit'."
      }
    },
    reqs=["level"]
  )
  def setLogLevel(self, level, logger_name="echo"):
    lvl_str = level.upper()
    mapping = {
      "DEBUG": logging.DEBUG,
      "INFO": logging.INFO,
      "WARNING": logging.WARNING,
      "ERROR": logging.ERROR,
      "CRITICAL": logging.CRITICAL,
    }
    if lvl_str not in mapping:
      return {
        "status": "error",
        "error": f"Unknown level '{level}'. Use one of: debug, info, warning, error, critical."
      }

    lvl = mapping[lvl_str]
    logger = logging.getLogger(logger_name)
    logger.setLevel(lvl)

    if logger_name == "echo":
      logging.getLogger().setLevel(lvl)

    return {
      "status": "success",
      "logger": logger_name,
      "level": lvl_str
    }

  @toolspec(
    desc="Enable or disable remembering previous conversation turns when answering. When disabled, only the latest user prompt is sent to the model.",
    args={
      "enabled": {
        "type": "string",
        "description": "Set to 'true' to enable history chaining, or 'false' to disable it."
      }
    },
    reqs=["enabled"]
  )
  def setHistoryChain(self, enabled):
    val = enabled.strip().lower()
    on = val in ("true", "1", "yes", "y", "on")

    self.chain_enabled = on

    return {
      "status": "success",
      "chain_enabled": self.chain_enabled
    }


#
# Extra toolkit (Shodan, research, arxiv, exploit-db, NVD, etc.)
#

class Toolkit(BaseToolkit):
  @toolspec(
    desc="Search arxiv for publications. Returns {url:<permalink>, title:<title>, authors:<authors>, summary:<summary>}",
    args={
      "query": {"type": "string", "description": "Arxiv query."},
      "limit": {"type": "integer", "description": "Optional. Number of results. Default: 10"}
    },
    reqs=["query"]
  )
  def arxivSearch(self, query, limit=10):
    print(f"{query}")
    client = arxiv.Client()
    res = client.results(arxiv.Search(
      query=query,
      max_results=limit
    ))
    entries = []
    for r in res:
      entries.append({
        'url': r.entry_id,
        'title': r.title,
        'authors': r.authors,
        'summary': r.summary
      })
    return json.dumps({"status": "success", "results": entries})

  @toolspec(
    desc="""Run a research model. Research model can access files and run code.
        Multiple files can be passes in with "files" argument. Supports local files and Arxiv permalinks.
        Pass research_id to continue research. Creates new research thread if empty.
      """,
    args={
      "query": {"type": "string", "description": "Research query."},
      "files": {"type": "array",
                "description": "Optional. Array of strings. List of files to include in research. Can be local files or Arxiv permalinks.",
                "items": {"type": "string"}},
      "research_id": {"type": "string",
                      "description": "Optional. Research thread id. If empty, a new research thread will be created."},
    },
    reqs=["query"],
    prompt="When researching better results are achieved by reusing existing research thread and uploading multiple files to one thread."
  )
  def research(self, query, files=None, research_id=None):
    if files is None:
      files = []
    ass = None
    thr = None
    if not research_id:
      ass = self.openai.beta.assistants.create(
        instructions="""
                You are a research assistant.
                Your job is to process scientific papers.
                Display mathematical formulas using MathJax \\[ markdown \\] blocks.
              """,
        name="Echo research",
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
        model=self.openai_research_model
      )
      thr = self.openai.beta.threads.create(metadata={'aid': ass.id})
      print(f"New research context: {thr.id}")
    else:
      thr = self.openai.beta.threads.retrieve(research_id)
      ass = self.openai.beta.assistants.retrieve(thr.metadata['aid'])
      print(f"Loaded research context: {thr.id}")

    for file in files:
      print(f"Loading file: {file}")
      if not os.path.isfile(file):
        file_id = urllib.parse.urlparse(file).path.rsplit("/", 1)[-1]
        res = arxiv.Search(id_list=[file_id])
        pdf = next(res.results())
        file = pdf.download_pdf(dirpath="./downloads/")
      with open(file, "rb") as f:
        fid = self.openai.files.create(file=f, purpose="assistants")
        self.openai.beta.assistants.files.create(assistant_id=ass.id, file_id=fid.id)

    print(f"Research query: {query}")
    ts_s = timer()
    self.openai.beta.threads.messages.create(thread_id=thr.id, role="user", content=query)
    run = self.openai.beta.threads.runs.create(assistant_id=ass.id, thread_id=thr.id)

    while run.status != "completed":
      time.sleep(1)
      run = self.openai.beta.threads.runs.retrieve(run_id=run.id, thread_id=run.thread_id)

    msg = self.openai.beta.threads.messages.list(thread_id=run.thread_id, limit=1).data[0].content[0].text.value
    ts_e = timer()
    print(f"... took {ts_e - ts_s}s")
    return {'research_id': thr.id, 'message': msg}

  @toolspec(
    desc="Interact with Shodan API. Search for internet-connected devices.",
    args={
      "query": {"type": "string", "description": "The search query to pass to Shodan."},
    },
    reqs=["query"]
  )
  def shodanSearch(self, query):
    results = self.shodan.search(query)
    return json.dumps({"status": "success", "results": results})

  @toolspec(
    desc="Interact with Shodan API. Get host info for an IP address.",
    args={
      "ip_address": {"type": "string", "description": "IP address to get host information for."}
    },
    reqs=["ip_address"]
  )
  def shodanHostInfo(self, ip_address):
    host_info = self.shodan.host(ip_address)
    return json.dumps({"status": "success", "result": host_info})

  @toolspec(
    desc=(
            "Search Exploit-DB (exploit-db.com) for exploits by keyword or CVE. "
            "Returns a list of exploits with id, description, type, platform, date, "
            "verified flag, port, tags, author, and link. Uses pyxploitdb."
    ),
    args={
      "query": {
        "type": "string",
        "description": "Search term (product, version, or CVE like 'CVE-2021-44228')."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return (default 10)."
      }
    },
    reqs=["query"]
  )
  def exploitdbSearch(self, query, limit=10):
    if pyxploitdb is None:
      return json.dumps({
        "status": "error",
        "error": "pyxploitdb not installed; run `pip install pyxploitdb`"
      })

    try:
      results = pyxploitdb.searchEDB(query, _print=False, nb_results=limit)
      payload = []
      for e in results:
        payload.append({
          "id": getattr(e, "id", None),
          "description": getattr(e, "description", None),
          "type": getattr(e, "type", None),
          "platform": getattr(e, "platform", None),
          "date_published": getattr(e, "date_published", None),
          "verified": getattr(e, "verified", None),
          "port": getattr(e, "port", None),
          "tags": getattr(e, "tag_if_any", None),
          "author": getattr(e, "author", None),
          "link": getattr(e, "link", None),
        })

      return json.dumps({
        "status": "success",
        "query": query,
        "results": payload
      })
    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"Exploit-DB search failed: {ex}"
      })

  @toolspec(
    desc=(
            "Search Exploit-DB specifically by CVE identifier "
            "(e.g. 'CVE-2006-1234', 'CVE-2021-44228'). "
            "Uses pyxploitdb.searchCVE under the hood."
    ),
    args={
      "cve": {
        "type": "string",
        "description": "CVE identifier to search for."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results (default 10)."
      }
    },
    reqs=["cve"]
  )
  def exploitdbSearchCVE(self, cve, limit=10):
    if pyxploitdb is None:
      return json.dumps({
        "status": "error",
        "error": "pyxploitdb not installed; run `pip install pyxploitdb`"
      })

    try:
      results = pyxploitdb.searchCVE(cve, _print=False)
      results = results[:limit]

      payload = []
      for e in results:
        payload.append({
          "id": getattr(e, "id", None),
          "description": getattr(e, "description", None),
          "type": getattr(e, "type", None),
          "platform": getattr(e, "platform", None),
          "date_published": getattr(e, "date_published", None),
          "verified": getattr(e, "verified", None),
          "port": getattr(e, "port", None),
          "tags": getattr(e, "tag_if_any", None),
          "author": getattr(e, "author", None),
          "link": getattr(e, "link", None),
        })

      return json.dumps({
        "status": "success",
        "cve": cve,
        "results": payload
      })
    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"Exploit-DB CVE search failed: {ex}"
      })

  @toolspec(
    desc=(
            "Query the U.S. National Vulnerability Database (NVD) for vulnerabilities. "
            "Supports keyword, CVE ID, product name, vendor name, etc. "
            "Requires NVD_API_KEY in .env. Returns normalized list of CVE records."
    ),
    args={
      "query": {
        "type": "string",
        "description": "Search string or CVE ID (e.g., 'Apache', 'OpenSSL', 'CVE-2021-44228')."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return (default 10)."
      }
    },
    reqs=["query"]
  )
  def nvdSearch(self, query, limit=10):
    api_key = self.nvd_api_key
    if not api_key:
      return json.dumps({
        "status": "error",
        "error": "NVD_API_KEY not set in .env"
      })

    base = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
      "keywordSearch": query,
      "resultsPerPage": limit,
      "apiKey": api_key
    }

    try:
      res = requests.get(base, params=params, timeout=12)
      res.raise_for_status()
      data = res.json()

      out = []
      for v in data.get("vulnerabilities", []):
        cve = v.get("cve", {})
        out.append({
          "id": cve.get("id"),
          "published": cve.get("published"),
          "lastModified": cve.get("lastModified"),
          "description": self._extractNvdDescription(cve),
          "cvss": self._extractNvdCvss(cve),
          "weaknesses": self._extractNvdWeaknesses(cve),
          "references": self._extractNvdReferences(cve)
        })

      return json.dumps({
        "status": "success",
        "query": query,
        "results": out
      })

    except Exception as ex:
      return json.dumps({
        "status": "error",
        "error": f"NVD query failed: {ex}"
      })

  # ---- helper methods for NVD -------
  def _extractNvdDescription(self, cve):
    descs = cve.get("descriptions", [])
    for d in descs:
      if d.get("lang") == "en":
        return d.get("value")
    return None

  def _extractNvdCvss(self, cve):
    metrics = cve.get("metrics", {})
    if "cvssMetricV31" in metrics:
      item = metrics["cvssMetricV31"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    if "cvssMetricV30" in metrics:
      item = metrics["cvssMetricV30"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    if "cvssMetricV2" in metrics:
      item = metrics["cvssMetricV2"][0]
      return {
        "baseScore": item["cvssData"]["baseScore"],
        "vector": item["cvssData"]["vectorString"]
      }
    return None

  def _extractNvdWeaknesses(self, cve):
    entries = []
    for w in cve.get("weaknesses", []):
      for desc in w.get("description", []):
        entries.append(desc.get("value"))
    return entries

  def _extractNvdReferences(self, cve):
    refs = []
    for r in cve.get("references", []):
      refs.append({
        "url": r.get("url"),
        "source": r.get("source"),
        "tags": r.get("tags")
      })
    return refs


#
# InterAction (External) – same as before, but based on BaseToolkit
#

class BaseToolkitHID(BaseToolkit):
  def __init__(self):
    super().__init__()

  @toolspec(
    desc="""
        Speak text using text-to-speech. Keep it short and entertaining. Jarvis style banter is welcome.
        Speak should only be used for very short communication - single sentence summary, remark or progress update.
        Category: output, audio
        """,
    args={"text": {"type": "string", "description": "Text to be spoken. Keep short, one sentence."}},
    reqs=["text"],
    prompt="When user says 'say','tell' etc use speak.",
    state="disabled"
  )
  def speak(self, text):
    self.trace.info("ACTION: Speaking short response via TTS.")
    threading.Thread(target=self.localtts, kwargs={'text': text}).start()
    return "{status: success}"

  def stt(self, file=None):
    if file is None:
      file = self.data.stt.file
    with open(file, "rb") as f:
      return self.openai.audio.transcriptions.create(
        model=self.openai_stt_model,
        file=f,
        response_format="text"
      )

  def localtts(self, text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "{status: success}"

  @toolspec(
    desc="Get input from speech-to-text. Used for primary prompt but can also be called for clarifications/followups/how-to-proceed advice. Category: input, audio",
    args={},
    reqs=[]
  )
  def listen(self):
    self.trace.info("ACTION: Starting microphone capture for speech input.")
    if self.data.stt is None:
      rec = sr.Recognizer()
      mic = sr.Microphone()
      self.data.stt = AttrDict({'rec': rec, 'mic': mic, 'file': './stt.mp3'})
      with mic:
        rec.adjust_for_ambient_noise(mic)
    with self.data.stt.mic:
      audio = self.data.stt.rec.listen(self.data.stt.mic)
    with open(self.data.stt.file, "wb") as f:
      f.write(audio.get_wav_data(convert_rate=44100))


class BaseToolkitOSID(BaseToolkit):
  def __init__(self):
    super().__init__()

  def screenshot(self, title=None):
    self.trace.info("ACTION: Capturing your screen (primary monitor).")
    with mss.mss() as sct:
      monitor = sct.monitors[1]  # 0 = all, 1 = primary
      img = sct.grab(monitor)
      img_pil = Image.frombytes("RGB", img.size, img.rgb)
      self.data.screenshot = img_pil
    return "{status: success}"

  def selectImage(self, image=None):
    if image is None:
      try:
        image = self.data.clipboard
        if not isinstance(image, Image.Image):
          image = Image.open(image)
      except:
        image = None
    if image is None:
      image = self.data.screenshot
    return image

  @toolspec(
    desc="Optical character recognition to extract text from image. Category: input, image",
    args={"image": {"type": "string",
                    "description": "Image file to OCR. If not specified, clipboard or screenshot will be used automatically."}},
    reqs=[]
  )
  def ocr(self, image=None):
    image = self.selectImage(image)
    return f"{{status: success, content:{pytesseract.image_to_string(image)}}}"

  @toolspec(
    desc="""
        Performs image processing using vision model. 
        Clipboard image or screenshot will be used automatically.
        Category: input, image""",
    args={"prompt": {"type": "string",
                     "description": "Prompt for vision model. User prompt will also be available for context."}},
    reqs=["prompt"],
    prompt="Plan: If clipboard data seems short or not suitable, consider calling vision instead."
  )
  def vision(self, prompt, img=None):
    img = self.selectImage(img)
    ocr = self.ocr(img)
    res = self.openai.chat.completions.create(
      model=self.openai_vision_model,
      max_tokens=500,
      messages=[{
        "role": "system",
        "content": f"""
              You are a subordinate function of an assistant called Echo.
              Echo determined that users request is related to this image and called you.
              You are not talking to the user directly. Be succint. Avoid pleasentries, appologizing and trivial explanations.
              OCR data of the image is provided below. 
              For context the user request to Echo was: {{{self.data.prompt}}}
              If user request is about textual data take a guess on what's important, extract it from OCR and return it verbatim.
              If user request is not about text or if OCR data is not useful to the request, proceed as you see fit yourself.
              <ocr>
              {ocr}
              </ocr>
            """
      },
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64(img)}"}
          ]
        }]
    )
    return res.choices[0].message.content

class FullToolkit(Toolkit, BaseToolkitHID, BaseToolkitOSID):
  def __init__(self):
    super().__init__()  # MRO will walk: FullToolkit → Toolkit → BaseToolkitHID → BaseToolkitOSID → BaseToolkit → BaseCoreToolkit

def create_toolkit(mode: str = "system"):
  """
  Small factory to get a toolkit instance:
  - system  -> BaseToolkit (minimal, safe)
  - extra   -> Toolkit (BaseToolkit + security/research tools)
  - hid     -> BaseToolkitHID (audio I/O)
  - os      -> BaseToolkitOSID (screen/vision/OCR)
  - full    -> FullToolkit (everything)
  """
  m = (mode or "system").lower()
  if m == "system":
    return BaseToolkit()
  if m == "extra":
    return Toolkit()
  if m == "hid":
    return BaseToolkitHID()
  if m == "os":
    return BaseToolkitOSID()
  if m == "full":
    return FullToolkit()
  # fallback
  return BaseToolkit()