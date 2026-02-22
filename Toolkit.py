import inspect
import json
import secrets
import traceback
from types import ModuleType
from openai import OpenAI
from dotenv import load_dotenv
from timeit import default_timer as timer
import logging

import atexit
try:
  import readline   # on Windows, install: pip install pyreadline3
except ImportError:
  readline = None

# Tool imports
import time
import os
import re
import webbrowser
import requests
import base64
import urllib
import urllib.parse
import pyperclip


from io import BytesIO

import subprocess
import shlex

import pywinctl as pwc
import pyautogui


# Tool imports (External)
import shodan
import serpapi
import arxiv
import pyxploitdb

#Tool import tools
from echo_config import HISTORY_ENTRIES_LIMIT

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
    self.redact_mode = os.getenv("REDACT_MODE", "off").lower() in (
      "1", "true", "yes", "y", "on"
    )

    # Load .env
    load_dotenv()

    # --- API keys ---
    self.shodan_api_key = os.getenv("SHODAN_API_KEY", "Missing Key")
    self.nvd_api_key = os.getenv("NVD_API_KEY")

    # --- Load LLM Providers and Model Mapping ---
    from echo_config import parse_llm_providers, parse_model_provider_map
    self.llm_providers = parse_llm_providers()
    self.model_provider_map = parse_model_provider_map()

    # Create reverse mapping: modelIdentifier -> modelName and providerName
    self.model_identifier_map = {}
    for model_name, mapping in self.model_provider_map.items():
      model_id = mapping.get("modelIdentifier", model_name)
      self.model_identifier_map[model_id] = {
        "modelName": model_name,
        "providerName": mapping.get("providerName")
      }

    # Cache for provider-specific OpenAI clients
    self.provider_clients = {}

    # --- LLM Backend configuration ---
    # Try new variable name first, then fall back to legacy OPENAI_LLM_BACKEND
    self.llm_backend = os.getenv("LLM_BACKEND",
                                  os.getenv("OPENAI_LLM_BACKEND", "completions")).lower()

    # --- Default fallback model for assistants tool choice ---
    self.default_assistants_fallback_model = os.getenv("DEFAULT_ASSISTANTS_FALLBACK_MODEL", None)

    # --- Base model values (used for 'current' profile by default) ---
    # Try new variable names first, then fall back to legacy OPENAI_ prefixed names
    default_chat = os.getenv("CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"))
    default_vision = os.getenv("VISION_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-5.0"))
    default_research = os.getenv("RESEARCH_MODEL", os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.1"))
    default_stt = os.getenv("STT_MODEL", os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"))

    # Store active values (will be overridden by profile)
    self.chat_model = default_chat
    self.vision_model = default_vision
    self.research_model = default_research
    self.stt_model = default_stt

    # --- Named model profiles (template sets) ---
    self.model_profiles = {
      "legacy": {
        "chat": os.getenv("CHAT_MODEL_LEGACY",
                         os.getenv("OPENAI_CHAT_MODEL_LEGACY", "gpt-4-turbo-preview")),
        "vision": os.getenv("VISION_MODEL_LEGACY",
                           os.getenv("OPENAI_VISION_MODEL_LEGACY", "gpt-4-vision-preview")),
        "research": os.getenv("RESEARCH_MODEL_LEGACY",
                             os.getenv("OPENAI_RESEARCH_MODEL_LEGACY", "gpt-4-turbo-preview")),
        "stt": os.getenv("STT_MODEL_LEGACY",
                        os.getenv("OPENAI_STT_MODEL_LEGACY", "whisper-1")),
      },
      "current": {
        "chat": os.getenv("CHAT_MODEL_CURRENT",
                         os.getenv("OPENAI_CHAT_MODEL_CURRENT", default_chat)),
        "vision": os.getenv("VISION_MODEL_CURRENT",
                           os.getenv("OPENAI_VISION_MODEL_CURRENT", default_vision)),
        "research": os.getenv("RESEARCH_MODEL_CURRENT",
                             os.getenv("OPENAI_RESEARCH_MODEL_CURRENT", default_research)),
        "stt": os.getenv("STT_MODEL_CURRENT",
                        os.getenv("OPENAI_STT_MODEL_CURRENT", default_stt)),
      },
    }

    # Select starting profile
    # Try new variable name first, then fall back to legacy OPENAI_MODEL_PROFILE
    self.current_model_profile = os.getenv("MODEL_PROFILE",
                                           os.getenv("OPENAI_MODEL_PROFILE", "current")).lower()

    # REDACT_MAP
    raw_redact_map = os.getenv("REDACT_MAP", "").strip()
    try:
      self.redact_map = json.loads(raw_redact_map) if raw_redact_map else {}
      if not isinstance(self.redact_map, dict):
        self.redact_map = {}
    except Exception:
      self.logger.exception("Failed to parse REDACT_MAP from env; using empty map.")
      self.redact_map = {}

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
    client, _, _ = self._get_client_for_model(self.chat_model)
    if not client:
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
    if self.llm_backend == "completions":
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
  def _resolve_model_identifier(self, model_identifier_or_name):
    """
    Resolve a model identifier to the actual model name and provider.

    Args:
      model_identifier_or_name: Can be either a modelIdentifier (e.g., "openai-gpt5-mini")
                                or a modelName (e.g., "gpt-5-mini")

    Returns:
      tuple: (modelName, providerName) or (model_identifier_or_name, None) if not found
    """
    # First check if it's a modelIdentifier
    if model_identifier_or_name in self.model_identifier_map:
      mapping = self.model_identifier_map[model_identifier_or_name]
      return mapping["modelName"], mapping["providerName"]

    # Then check if it's already a modelName
    if model_identifier_or_name in self.model_provider_map:
      mapping = self.model_provider_map[model_identifier_or_name]
      return model_identifier_or_name, mapping.get("providerName")

    # Not found in mapping, return as-is (for backward compatibility with unmapped models)
    return model_identifier_or_name, None

  def _get_model_config(self, model_identifier_or_name):
    """
    Get the model configuration from the model mappings.

    Args:
      model_identifier_or_name: Either a modelIdentifier or modelName

    Returns:
      dict: Model config or None if not found
    """
    # First check if it's a modelIdentifier
    if model_identifier_or_name in self.model_identifier_map:
      return self.model_identifier_map[model_identifier_or_name]

    # Then check if it's already a modelName
    if model_identifier_or_name in self.model_provider_map:
      return self.model_provider_map[model_identifier_or_name]

    return None

  def _get_client_for_model(self, model_identifier_or_name):
    """
    Get the appropriate client for the given model.
    Creates provider-specific clients (OpenAI, Anthropic, Google).

    Args:
      model_identifier_or_name: Either a modelIdentifier or modelName

    Returns: (client, provider_info, actual_model_name)
    """
    # Resolve the identifier to actual model name and provider
    model_name, provider_name = self._resolve_model_identifier(model_identifier_or_name)

    if provider_name and provider_name in self.llm_providers:
      # Check if we already have a cached client for this provider
      if provider_name in self.provider_clients:
        return self.provider_clients[provider_name], self.llm_providers[provider_name], model_name

      # Create a new client for this provider
      provider_config = self.llm_providers[provider_name]
      api_key = provider_config.get("apiKey")
      endpoint = provider_config.get("endpoint")

      try:
        client = None

        # Create provider-specific client
        if provider_name == "anthropic":
          from anthropic import Anthropic
          client_kwargs = {}
          if api_key:
            client_kwargs["api_key"] = api_key
          if endpoint:
            client_kwargs["base_url"] = endpoint
          client = Anthropic(**client_kwargs) if client_kwargs else None

        elif provider_name == "google":
          import google.generativeai as genai
          if api_key:
            genai.configure(api_key=api_key)
            client = genai  # Store the module itself as client

        else:
          # Default: OpenAI-compatible API
          client_kwargs = {}
          if api_key:
            client_kwargs["api_key"] = api_key
          if endpoint:
            client_kwargs["base_url"] = endpoint
          if client_kwargs:
            client = OpenAI(**client_kwargs)

        if client:
          self.provider_clients[provider_name] = client
          self.logger.info(f"Created new client for provider: {provider_name}")
          return client, provider_config, model_name
        else:
          self.logger.warning(f"No API key configured for provider: {provider_name}")

      except Exception as e:
        self.logger.error(f"Failed to create client for provider {provider_name}: {e}")

    # Fall back to OpenAI provider if configured
    if "openai" in self.llm_providers:
      if "openai" not in self.provider_clients:
        provider_config = self.llm_providers["openai"]
        api_key = provider_config.get("apiKey")
        endpoint = provider_config.get("endpoint")

        client_kwargs = {}
        if api_key:
          client_kwargs["api_key"] = api_key
        if endpoint:
          client_kwargs["base_url"] = endpoint

        if client_kwargs:
          self.provider_clients["openai"] = OpenAI(**client_kwargs)
          self.logger.info("Created fallback OpenAI client")

      if "openai" in self.provider_clients:
        return self.provider_clients["openai"], self.llm_providers["openai"], model_name

    self.logger.error(f"No provider configured for model: {model_identifier_or_name}")
    return None, None, None

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

  def _expand_redacted_placeholders(self, text: str) -> str:
    if not isinstance(text, str):
      return text
    if not getattr(self, "redact_mode", False):
      return text
    if not getattr(self, "redact_map", None):
      return text

    out = text
    for placeholder, real_value in self.redact_map.items():
      try:
        if isinstance(real_value, (dict, list)):
          real_value = json.dumps(real_value)
        out = out.replace(str(placeholder), str(real_value))
      except Exception:
        continue
    return out

  @toolspec(
    desc="Set or update API keys for LLM providers or external services. For LLM providers, updates the LLM_PROVIDERS configuration.",
    args={
      "service": {
        "type": "string",
        "description": "Service/provider name: any provider from LLM_PROVIDERS (e.g., 'openai', 'anthropic', 'google'), or 'shodan', 'serpapi'."
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

    # Check if it's an LLM provider
    if svc in self.llm_providers:
      # Update the provider's API key in memory
      self.llm_providers[svc]["apiKey"] = api_key

      # Invalidate cached client for this provider so it gets recreated with new key
      if svc in self.provider_clients:
        del self.provider_clients[svc]

      self.logger.info(f"Updated API key for LLM provider: {svc}")
      return json.dumps({
        "status": "success",
        "service": svc,
        "message": f"API key updated for provider '{svc}'. Note: changes are in-memory only and won't persist to .env file."
      })

    # Handle non-LLM services
    elif svc == "shodan":
      self.shodan_api_key = api_key
      persisted = self._update_env_var("SHODAN_API_KEY", api_key)
      if hasattr(self, "shodan"):
        self.shodan = shodan.Shodan(api_key)
      return json.dumps({
        "status": "success",
        "service": svc,
        "persisted": persisted
      })

    elif svc == "serpapi":
      persisted = self._update_env_var("SERPAPI_API_KEY", api_key)
      if hasattr(self, "serpapi"):
        self.serpapi = serpapi.Client(api_key=api_key)
      return json.dumps({
        "status": "success",
        "service": svc,
        "persisted": persisted
      })

    else:
      available_providers = list(self.llm_providers.keys())
      return json.dumps({
        "status": "error",
        "error": f"Unknown service '{service}'. Available LLM providers: {available_providers}. Other services: shodan, serpapi."
      })

  def _apply_model_profile(self, profile_name: str) -> bool:
    profile_key = profile_name.strip().lower()
    if profile_key not in self.model_profiles:
      self.logger.warning("Unknown model profile '%s'", profile_name)
      return False

    prof = self.model_profiles[profile_key]

    if "chat" in prof:
      self.chat_model = prof["chat"]
    if "vision" in prof:
      self.vision_model = prof["vision"]
    if "research" in prof:
      self.research_model = prof["research"]
    if "stt" in prof:
      self.stt_model = prof["stt"]

    self.current_model_profile = profile_key
    self.logger.info(
      "Switched model profile to '%s' (chat=%s, vision=%s, research=%s, stt=%s)",
      profile_key,
      self.chat_model,
      self.vision_model,
      self.research_model,
      self.stt_model,
    )
    return True

  @toolspec(
    desc="Show current model profile and provider information for all models (chat, vision, research, STT).",
    args={},
    reqs=[]
  )
  def showModelProfile(self):
    # Get provider info for each model
    def get_provider_info(model_name):
      if model_name in self.model_provider_map:
        mapping = self.model_provider_map[model_name]
        provider_name = mapping.get("providerName", "unknown")
        model_id = mapping.get("modelIdentifier", model_name)
        if provider_name in self.llm_providers:
          provider_config = self.llm_providers[provider_name]
          provider_display = provider_config.get("name", provider_name)
          endpoint = provider_config.get("endpoint", "N/A")
          return {
            "provider": provider_display,
            "providerName": provider_name,
            "modelIdentifier": model_id,
            "endpoint": endpoint
          }
        return {
          "provider": provider_name,
          "providerName": provider_name,
          "modelIdentifier": model_id,
          "endpoint": "N/A"
        }
      return {
        "provider": "default",
        "providerName": "default",
        "modelIdentifier": model_name,
        "endpoint": "legacy configuration"
      }

    return {
      "current_profile": self.current_model_profile,
      "available_profiles": list(self.model_profiles.keys()),
      "backend": self.llm_backend,
      "models": {
        "chat": {
          "model": self.chat_model,
          **get_provider_info(self.chat_model)
        },
        "vision": {
          "model": self.vision_model,
          **get_provider_info(self.vision_model)
        },
        "research": {
          "model": self.research_model,
          **get_provider_info(self.research_model)
        },
        "stt": {
          "model": self.stt_model,
          **get_provider_info(self.stt_model)
        }
      },
      "total_providers": len(self.llm_providers),
      "total_model_mappings": len(self.model_provider_map)
    }

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

    # Get provider info for each model
    def get_provider_info(model_name):
      if model_name in self.model_provider_map:
        mapping = self.model_provider_map[model_name]
        provider_name = mapping.get("providerName", "unknown")
        model_id = mapping.get("modelIdentifier", model_name)
        if provider_name in self.llm_providers:
          provider_display = self.llm_providers[provider_name].get("name", provider_name)
          return f"{provider_display} ({model_id})"
        return f"{provider_name} ({model_id})"
      return "default (legacy)"

    return {
      "status": "success",
      "profile": self.current_model_profile,
      "chat_model": self.chat_model,
      "chat_provider": get_provider_info(self.chat_model),
      "vision_model": self.vision_model,
      "vision_provider": get_provider_info(self.vision_model),
      "research_model": self.research_model,
      "research_provider": get_provider_info(self.research_model),
      "stt_model": self.stt_model,
      "stt_provider": get_provider_info(self.stt_model),
    }

  def _convert_to_anthropic_messages(self, messages):
    """Convert OpenAI format messages to Anthropic format."""
    # Anthropic doesn't support 'system' in messages array, extract it
    system_content = None
    anthropic_messages = []

    for msg in messages:
      if not isinstance(msg, dict):
        continue

      role = msg.get("role")
      content = msg.get("content", "")

      # Extract system message
      if role == "system":
        system_content = content
        continue

      # Skip tool messages for now (Anthropic handles tools differently)
      if role == "tool":
        continue

      # Skip assistant messages with only tool_calls
      if role == "assistant" and msg.get("tool_calls") and not content:
        continue

      anthropic_messages.append({
        "role": role,
        "content": content
      })

    return system_content, anthropic_messages

  def _convert_to_gemini_messages(self, messages):
    """Convert OpenAI format messages to Gemini format."""
    # Gemini uses a different structure with 'parts'
    gemini_messages = []
    system_instruction = None

    for msg in messages:
      if not isinstance(msg, dict):
        continue

      role = msg.get("role")
      content = msg.get("content", "")

      # Extract system message as system_instruction
      if role == "system":
        system_instruction = content
        continue

      # Skip tool messages for now
      if role == "tool":
        continue

      # Map roles: assistant -> model, user -> user
      gemini_role = "model" if role == "assistant" else "user"

      # Skip assistant messages with only tool_calls
      if role == "assistant" and msg.get("tool_calls") and not content:
        continue

      gemini_messages.append({
        "role": gemini_role,
        "parts": [{"text": content}]
      })

    return system_instruction, gemini_messages

  def llm_call(self, messages, tools=None, tool_choice="auto", **kwargs):
    """
    Unified LLM call supporting multiple providers (OpenAI, Anthropic, Google).

    Returns:
      {
        "backend": "completions" | "responses" | "anthropic" | "gemini",
        "raw": <raw SDK response>,
        "finish_reason": "stop" | "tool_calls" | <other> | None,
        "message": <primary assistant message-like object>,
        "provider": <provider info dict>,
      }
    """
    # Get the appropriate client for the current model and resolve to actual model name
    client, provider_info, actual_model_name = self._get_client_for_model(self.chat_model)

    if not client:
      raise RuntimeError(f"No client available for model: {self.chat_model}")

    provider_name = provider_info.get("providerName", "default")
    self.logger.info(f"Using provider '{provider_info.get('name', 'Unknown')}' for model: {self.chat_model} (resolved to: {actual_model_name})")

    backend = getattr(self, "llm_backend", "completions").lower()
    tools = tools if tools is not None else self.toolMessage()

    if "tools" in kwargs:
      if not tools:
        tools = kwargs["tools"]
      kwargs.pop("tools")

    # ---------------- ANTHROPIC (CLAUDE) ----------------
    if provider_name == "anthropic":
      system_content, anthropic_messages = self._convert_to_anthropic_messages(messages)

      call_kwargs = {"model": actual_model_name, "messages": anthropic_messages, "max_tokens": kwargs.get("max_tokens", 4096)}
      if system_content:
        call_kwargs["system"] = system_content

      res = client.messages.create(**call_kwargs)

      # Convert response to unified format
      content = ""
      for block in res.content:
        if hasattr(block, "text"):
          content += block.text

      from types import SimpleNamespace
      message_like = SimpleNamespace(role="assistant", content=content)

      return {
        "backend": "anthropic",
        "raw": res,
        "finish_reason": res.stop_reason or "stop",
        "message": message_like,
        "provider": provider_info,
      }

    # ---------------- GOOGLE (GEMINI) ----------------
    elif provider_name == "google":
      system_instruction, gemini_messages = self._convert_to_gemini_messages(messages)

      model = client.GenerativeModel(actual_model_name, system_instruction=system_instruction)

      # Convert to single conversation
      if gemini_messages:
        # Start chat with history
        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        last_message = gemini_messages[-1] if gemini_messages else {"parts": [{"text": ""}]}
        res = chat.send_message(last_message["parts"][0]["text"])
      else:
        res = model.generate_content("")

      from types import SimpleNamespace
      message_like = SimpleNamespace(role="assistant", content=res.text if hasattr(res, "text") else "")

      return {
        "backend": "gemini",
        "raw": res,
        "finish_reason": "stop",
        "message": message_like,
        "provider": provider_info,
      }

    # ---------------- OPENAI CHAT COMPLETIONS ----------------
    elif backend == "completions":
      res = client.chat.completions.create(
        model=actual_model_name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
      )
      choice = res.choices[0]
      return {
        "backend": "completions",
        "raw": res,
        "finish_reason": choice.finish_reason,
        "message": choice.message,
        "provider": provider_info,
      }

    # ---------------- OPENAI RESPONSES ----------------
    elif backend == "responses":
        kwargs.pop("tool_choice", None)

        input_items = self._messages_to_responses_input(messages)
        tools_for_responses = self._tools_for_responses(tools)

        res = client.responses.create(
            model=actual_model_name,
            input=input_items,
            tools=tools_for_responses,
            **kwargs,
        )

        text = None
        text = getattr(res, "output_text", None)

        # 2) Fallback: manually walk res.output
        if not text:
            chunks = []
            output = getattr(res, "output", None) or []
            for item in output:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if hasattr(c, "text") and c.text:
                            chunks.append(c.text)
            text = "\n".join(chunks) if chunks else ""

        # --------- normalize stop_reason to your old semantics ----------
        finish_reason = "stop"
        output = getattr(res, "output", None) or []
        if output:
            first = output[0]
            raw_reason = getattr(first, "stop_reason", None)
            if raw_reason == "tool_use":
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

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
            "provider": provider_info,
        }

    raise ValueError(f"Unknown provider or backend: {provider_name} / {backend}")

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

    self._setup_shell_history()
    self.history_user = []

  # -------------------
  # Redaction utilities
  # -------------------
  def _redact_text(self, text: str) -> str:
    """
    Best-effort redaction of sensitive data from plain text.
    Runs ONLY when self.redact_mode is True.

    Redacts:
      - IPv4 addresses
      - email addresses
      - GPS-like coordinates
      - hostnames/domains
      - recon/shodan structured fields (Location, ISP/Org, Org, ASN, etc.)
      - parenthetical geo hints after IP lines (e.g. "(Krasnogorsk, Russian Federation)")
    """
    if not isinstance(text, str):
      return text
    if not getattr(self, "redact_mode", False):
      return text

    red = text

    # 1) IPv4
    red = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]", red)

    # 1.1) IPv6 (full + compressed, incl ::)
    red = re.sub(
      r"\b(?:"  # word boundary
      r"(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}|"  # full
      r"(?:[A-Fa-f0-9]{1,4}:){1,7}:|"  # :: short
      r"(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}|"
      r"(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}|"
      r"(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}|"
      r"(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}|"
      r"(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}|"
      r"[A-Fa-f0-9]{1,4}:(?:(?::[A-Fa-f0-9]{1,4}){1,6})|"
      r":(?:(?::[A-Fa-f0-9]{1,4}){1,7}|:)"
      r")\b",
      "[REDACTED_IPv6]",
      red
    )

    # 2) Emails
    red = re.sub(
      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
      "[REDACTED_EMAIL]",
      red,
    )

    # 3) GPS-like coordinates:
    red = re.sub(
      r"\b-?\d{1,2}\.\d+,\s*-?\d{1,3}\.\d+\b",
      "[REDACTED_COORDINATES]",
      red,
    )

    # 4) Structured fields (supports slashes in labels)
    red = re.sub(
      r"(?im)^(\s*[-*]?\s*"
      r"(Location|ISP\s*/\s*Org|ISP|Org|Organization|ASN|AS|City|Country|Region)"
      r"\s*:\s*)(.+)$",
      r"\1[REDACTED_VALUE]",
      red
    )

    # 5) Parenthetical geo/org leaks after IP lines
    red = re.sub(
      r"(?im)^(\s*[-*]?\s*IP\s*:\s*.*?)(\s*\([^)]*\))\s*$",
      r"\1 ([REDACTED_LOCATION])",
      red
    )

    # 6) Hostnames/domains (rough heuristic; can also hide org-like identifiers)
    # Keep this late so it doesn't interfere with the field-label rules above.
    red = re.sub(
      r"\b([a-zA-Z0-9-]+\.){1,}[a-zA-Z]{2,}\b",
      "[REDACTED_HOST]",
      red,
    )

    return red

  @toolspec(
    desc="Redact sensitive data (IP addresses, emails, coordinates, hostnames/organisations) from the given text.",
    args={
      "text": {
        "type": "string",
        "description": "Plain text to redact."
      }
    },
    reqs=["text"]
  )
  def redactSensitive(self, text):
    return self._redact_text(text)

  @toolspec(
    desc=(
            "Read a line of text from the console. "
            "If a prompt string is provided, it will be displayed and used by readline, "
            "allowing correct shell history navigation with ↑ and ↓ keys. "
            "This function is normally used internally by the toolkit; "
            "it is suitable for follow-up questions or clarifications where plain text input is required."
    ),
    args={
      "prompt": {
        "type": "string",
        "description": "Optional prompt to show before reading user input."
      }
    },
    reqs=[]
  )
  def read(self, prompt=""):
    self.trace.info("ACTION: Reading text from console (input()).")
    return input(prompt)

  def input(self, prompt=""):
    text = None
    if 'listen' in self._toolspec and self._toolspec.listen.state == "enabled":
      self.trace.info("ACTION: Listening to your voice (speech-to-text).")
      self.listen()
      text = self.stt()
      self.trace.info("ACTION: Transcribed your voice input.")
    else:
      self.trace.info("ACTION: Waiting for your console text input.")
      text = self.read(prompt)
      self.trace.info("ACTION: Received your text input from console.")

    self.data.prompt_raw = text

    # Apply redacted placeholder expansion (only if redacted mode is ON)
    effective = self._expand_redacted_placeholders(text)

    self.data.prompt = effective
    self.add_user_history(text)

    # Reset snapshot data
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

  def add_user_history(self, text):
    if not hasattr(self, "history_user"):
      self.history_user = []

    self.history_user.append(text)
    # apply limit from config
    try:
      from echo_config import HISTORY_ENTRIES_LIMIT
      self.history_user = self.history_user[-HISTORY_ENTRIES_LIMIT:]
    except Exception:
      pass

  def _setup_shell_history(self):
    if readline is None:
      self.logger.info("Readline not available; shell-like history disabled.")
      return

    histfile = os.path.join(os.path.expanduser("~"), ".echo_history")

    try:
      readline.read_history_file(histfile)
    except FileNotFoundError:
      pass
    except Exception as e:
      self.logger.warning("Could not read history file %s: %s", histfile, e)

    try:
      readline.set_history_length(HISTORY_ENTRIES_LIMIT)
    except Exception as e:
      self.logger.warning("Could not set history length: %s", e)

    def _save_history():
      try:
        readline.write_history_file(histfile)
      except Exception as e:
        self.logger.warning("Could not write history file %s: %s", histfile, e)

    atexit.register(_save_history)

  def get_user_history(self):
    return list(self.history_user)

  def clear_user_history(self):
    self.history_user = []


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
    desc="Change which model the toolkit uses at runtime for a specific purpose.",
    args={
      "target": {
        "type": "string",
        "description": "Which model to change: one of 'chat', 'vision', 'research', 'stt'."
      },
      "model": {
        "type": "string",
        "description": "New model identifier or name, e.g. 'openai-gpt5-mini' or 'claude-sonnet-3.5'."
      }
    },
    reqs=["target", "model"]
  )
  def setModel(self, target, model):
    target_l = target.lower()
    if target_l == "chat":
      self.chat_model = model
    elif target_l == "vision":
      self.vision_model = model
    elif target_l == "research":
      self.research_model = model
    elif target_l == "stt":
      self.stt_model = model
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

  @toolspec(
    desc="Enable or disable redacted mode. When enabled, placeholders from the redaction map are expanded for input and outputs are post-redacted.",
    args={
      "enabled": {
        "type": "string",
        "description": "Set to 'true'/'on' to enable, 'false'/'off' to disable."
      }
    },
    reqs=["enabled"]
  )
  def setRedactMode(self, enabled):
    val = (enabled or "").strip().lower()
    on = val in ("true", "1", "yes", "y", "on")
    self.redact_mode = on
    return {
      "status": "success",
      "redact_mode": self.redact_mode
    }

  @toolspec(
    desc="Replace the JSON redaction map. Keys are placeholders (as typed by the user), values are real sensitive tokens.",
    args={
      "mapping_json": {
        "type": "string",
        "description": "JSON object, e.g. {\"redactedIP\": \"127.0.0.1\"}."
      }
    },
    reqs=["mapping_json"]
  )
  def setRedactMap(self, mapping_json):
    try:
      m = json.loads(mapping_json)
      if not isinstance(m, dict):
        raise ValueError("mapping_json must be a JSON object")
    except Exception as e:
      return {
        "status": "error",
        "error": f"Invalid mapping_json: {e}"
      }

    self.redact_map = m
    return {
      "status": "success",
      "size": len(self.redact_map)
    }

#
# Extra toolkit (Shodan, research, arxiv, exploit-db, NVD, etc.)
#

class Toolkit(BaseToolkit):
  def __init__(self):
    super().__init__()
    self.echo = logging.getLogger("echo")
    self.echo_toolkit = logging.getLogger("echo.toolkit")
    self.trace = logging.getLogger("echo.trace")

    # Load env variables here
    load_dotenv()
    self.wpscan_api_token = os.getenv("WPSCAN_API_TOKEN")

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
      client, provider_info, actual_model_name = self._get_client_for_model(self.research_model)
      if not client:
        return json.dumps({"status": "error", "error": "Research model client not available"})

      # Prepare extra params for providers requiring custom_llm_provider (e.g., LiteLLM)
      extra_params = {}
      if provider_info and provider_info.get("requireCustomLlmProvider"):
        # Extract custom_llm_provider from model name
        if "_" in actual_model_name:
          # Format: provider_model (e.g., pcss_gpt_oss_120b)
          potential_provider = actual_model_name.split("_")[0]
          extra_params["extra_headers"] = {"custom-llm-provider": potential_provider}

      ass = client.beta.assistants.create(
        instructions="""
                You are a research assistant.
                Your job is to process scientific papers.
                Display mathematical formulas using MathJax \\[ markdown \\] blocks.
              """,
        name="Echo research",
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
        model=actual_model_name,
        **extra_params
      )
      thr = client.beta.threads.create(metadata={'aid': ass.id})
      print(f"New research context: {thr.id}")
    else:
      client, _, _ = self._get_client_for_model(self.research_model)
      if not client:
        return json.dumps({"status": "error", "error": "Research model client not available"})
      thr = client.beta.threads.retrieve(research_id)
      ass = client.beta.assistants.retrieve(thr.metadata['aid'])
      print(f"Loaded research context: {thr.id}")

    for file in files:
      print(f"Loading file: {file}")
      if not os.path.isfile(file):
        file_id = urllib.parse.urlparse(file).path.rsplit("/", 1)[-1]
        res = arxiv.Search(id_list=[file_id])
        pdf = next(res.results())
        file = pdf.download_pdf(dirpath="./downloads/")
      with open(file, "rb") as f:
        fid = client.files.create(file=f, purpose="assistants")
        client.beta.assistants.files.create(assistant_id=ass.id, file_id=fid.id)

    print(f"Research query: {query}")
    ts_s = timer()
    client.beta.threads.messages.create(thread_id=thr.id, role="user", content=query)

    # Prepare run params - check if model has explicit assistantsToolChoiceOverride or use supportsToolChoice
    model_config = self._get_model_config(self.research_model)
    run_params = {"assistant_id": ass.id, "thread_id": thr.id}

    # Check if we should use a fallback model for tool choice decisions
    use_fallback_for_tools = False
    fallback_model = None
    if model_config:
      # Check for model-specific fallback first
      if "assistantsFallbackModel" in model_config:
        fallback_model = model_config["assistantsFallbackModel"]
        use_fallback_for_tools = True
        print(f"Using model-specific fallback '{fallback_model}' for tool choice decisions")
      # If model doesn't support tools and has no specific fallback, use default
      elif model_config.get("supportsToolChoice") is False and self.default_assistants_fallback_model:
        fallback_model = self.default_assistants_fallback_model
        use_fallback_for_tools = True
        print(f"Using default fallback model '{fallback_model}' for tool choice decisions")

    if model_config and not use_fallback_for_tools:
      # If explicit assistantsToolChoiceOverride is set, use it (allows manual override)
      if "assistantsToolChoiceOverride" in model_config:
        run_params["tool_choice"] = model_config["assistantsToolChoiceOverride"]
      # Otherwise, fall back to supportsToolChoice flag
      elif model_config.get("supportsToolChoice") is False:
        run_params["tool_choice"] = "none"

    # If using fallback model for tools, ask it to decide
    if use_fallback_for_tools:
      # Use fallback model to determine which tools to use
      fallback_client, _, fallback_model_name = self._get_client_for_model(fallback_model)
      if fallback_client:
        tool_decision_prompt = f"""Analyze this research query and decide if tools are needed:

Query: {query}
Available files: {files if files else 'None'}

Available tools:
1. code_interpreter - For running Python code, calculations, data analysis
2. retrieval - For searching and extracting information from uploaded files

Respond in JSON format:
{{"needs_tools": true/false, "tools_needed": ["code_interpreter", "retrieval"], "reasoning": "why these tools are needed"}}"""

        try:
          tool_decision = self.llm_call(
            messages=[{"role": "user", "content": tool_decision_prompt}],
            response_format={"type": "json_object"},
            model_override=fallback_model
          )

          # Parse the decision
          if self.llm_backend == "completions":
            decision_content = tool_decision["raw"].choices[0].message.content
          else:
            first_step = tool_decision["raw"].output[0]
            text_block = next(c for c in first_step.content if getattr(c, "type", None) in ("output_text", "message", None))
            decision_content = getattr(text_block, "text", str(text_block))

          decision = json.loads(decision_content)
          print(f"Fallback model decision: {decision}")

          # If tools are needed, use "auto", otherwise "none"
          if decision.get("needs_tools"):
            run_params["tool_choice"] = "auto"
          else:
            run_params["tool_choice"] = "none"
        except Exception as e:
          print(f"Error getting tool decision from fallback model: {e}")
          # Fall back to disabling tools
          run_params["tool_choice"] = "none"

    run = client.beta.threads.runs.create(**run_params)

    while run.status != "completed":
      time.sleep(1)
      run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=run.thread_id)

    msg = client.beta.threads.messages.list(thread_id=run.thread_id, limit=1).data[0].content[0].text.value
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
      #"apiKey": api_key
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

  @toolspec(
    desc=(
            "Run a WordPress security scan using the WPScan Docker image. "
            "Command executed: docker run -it --rm --network host wpscanteam/wpscan --url {SITE} -e vp  --plugins-detection mixed  --enumerate u --api-token {API_TOKEN}"
            "It executes `docker run wpscanteam/wpscan` against the given site URL, "
            "parses the output, and returns a structured JSON summary instead of raw text. "
            "The WPScan API token is loaded from the WPSCAN_API_TOKEN variable in .env and "
            "is never logged or returned in clear text."
    ),
    args={
      "url": {
        "type": "string",
        "description": "Target WordPress site URL, e.g. 'https://example.com'."
      },
      "extra_args": {
        "type": "string",
        "description": "Optional extra CLI arguments passed verbatim to WPScan."
      }
    },
    reqs=["url"]
  )
  def wordpressScan(self, url, extra_args=""):
    """
    Run a WPScan Docker-based WordPress scan and return a structured JSON summary.

    The command executed is approximately:

      docker run --rm wpscanteam/wpscan \\
        --url <url> -e vp --plugins-detection mixed \\
        --api-token <WPSCAN_API_TOKEN_FROM_ENV>

    Important:
    - The real API token is never written to logs or returned to the caller.
    - The returned JSON contains a parsed summary (version, themes, plugins, users, meta, etc.)
      along with the raw stdout/stderr for debugging.
    """
    logger = self.echo_toolkit
    trace = self.trace

    api_token = getattr(self, "wpscan_api_token", None)
    if not api_token:
      msg = "Missing WPSCAN_API_TOKEN in .env (required for wordpressScan)."
      logger.error(msg)
      trace.error(f"ACTION: {msg}")
      return json.dumps({"status": "error", "error": msg})

    base_cmd = [
      "docker", "run", "--rm", "--network", "host",
      "wpscanteam/wpscan",
      "--url", url,
      "-e", "vp",
      "--plugins-detection", "mixed",
      "--api-token", api_token,
    ]

    if extra_args:
      base_cmd.extend(shlex.split(extra_args))

    redacted_cmd = list(base_cmd)
    try:
      idx = redacted_cmd.index("--api-token")
      if idx + 1 < len(redacted_cmd):
        redacted_cmd[idx + 1] = "****REDACTED****"
    except ValueError:
      pass

    logger.info("Starting WPScan for url=%s", url)
    logger.debug("WPScan command (redacted): %r", redacted_cmd)
    trace.info("ACTION: Running WPScan Docker scan for %s", url)

    try:
      ts_s = timer()
      proc = subprocess.run(
        base_cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # safety timeout: 1 hour
      )
      ts_e = timer()
      elapsed = ts_e - ts_s

      logger.info("WPScan finished for %s with returncode=%s (%.2fs)", url, proc.returncode, elapsed)
      if proc.returncode != 0:
        logger.warning("WPScan reported non-zero exit status %s for url=%s", proc.returncode, url)

      parsed = self._parseWpscanOutput(proc.stdout or "", url=url)

      result = {
        "status": "success" if proc.returncode == 0 else "error",
        "url": url,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": redacted_cmd,  # <-- TOKEN CENSORED HERE
        "parsed": parsed,
        "raw": {
          "stdout": proc.stdout,
          "stderr": proc.stderr,
        },
      }
      return json.dumps(result)

    except subprocess.TimeoutExpired as ex:
      msg = f"WPScan timeout for url={url}: {ex}"
      logger.error(msg)
      trace.error(f"ACTION: {msg}")
      return json.dumps({
        "status": "error",
        "url": url,
        "error": f"WPScan timed out: {ex}",
      })
    except FileNotFoundError as ex:
      msg = f"WPScan failed for url={url}: docker executable not found"
      logger.error("%s (%s)", msg, ex)
      trace.error("ACTION: WPScan failed – docker executable not found.")
      return json.dumps({
        "status": "error",
        "url": url,
        "error": "docker executable not found. Make sure Docker is installed and in PATH.",
      })
    except Exception as ex:
      logger.exception("WPScan unexpected error for url=%s", url)
      trace.error(f"ACTION: WPScan raised unexpected exception: {ex}")
      return json.dumps({
        "status": "error",
        "url": url,
        "error": f"Unexpected error while running WPScan: {ex}",
      })

# Helpers --

  # ---- helper for WPScan parsing ----
  def _stripAnsi(self, text):
    """Remove ANSI colour codes from WPScan output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

  def _parseWpscanOutput(self, stdout, url=None):
    """Parse WPScan human-readable output into a structured JSON-friendly dict.

    The output is free-form text; this helper extracts the most useful parts for the LLM:
    - WordpressVersion (version, description, is_latest - when detectable)
    - ScanDate
    - Plugins: [{name, foundVulnerability, vulnerabilities}]
    - Templates: [{name, version, latestVersion, foundVulnerability, vulnerabilities}]
    - Users: list of usernames
    - InterestingFindings: list of simple strings
    - Meta: duration, requests, data sent/received, memory usage, etc. (when present)
    """
    clean = self._stripAnsi(stdout or "")
    lines = [ln.rstrip() for ln in clean.splitlines()]

    wordpress_version = None
    wp_version_desc = None
    scan_date = None
    finished_at = None
    plugins = []
    templates = []
    users = []
    interesting = []
    meta = {}

    def parse_plugin_block(name, block_lines):
      text = "\n".join(block_lines)
      version = None
      latest = None
      vulns = []
      found_vuln = False

      # Version & latest version
      m_ver = re.search(r"Version:\s*([^\s]+)", text)
      if m_ver:
        version = m_ver.group(1).strip().strip(".")
      m_latest = re.search(r"latest version is\s*([0-9\.]+)", text, re.IGNORECASE)
      if m_latest:
        latest = m_latest.group(1).strip()

      # Any explicit warnings / vulnerabilities
      for ln in block_lines:
        if "[!]" in ln or "vulnerab" in ln.lower():
          found_vuln = True
          vulns.append(ln.strip())

      return {
        "name": name,
        "version": version,
        "latestVersion": latest,
        "foundVulnerability": found_vuln,
        "vulnerabilities": vulns,
      }

    def parse_template_block(name, block_lines):
      # Theme/template parsing is similar to plugin parsing
      return parse_plugin_block(name, block_lines)

    i = 0
    n = len(lines)
    while i < n:
      line = lines[i].strip()

      if not line:
        i += 1
        continue

      # URL
      if line.startswith("[+] URL:"):
        # format: [+] URL: https://example.com/ [1.2.3.4]
        try:
          part = line.split("URL:", 1)[1].strip()
          # Drop trailing [IP] if present
          if " [" in part:
            part = part.split(" [", 1)[0].strip()
          url = url or part
        except Exception:
          pass

      # Scan start/finish
      if line.startswith("[+] Started:"):
        scan_date = line.split("Started:", 1)[1].strip()
      if line.startswith("[+] Finished:"):
        finished_at = line.split("Finished:", 1)[1].strip()

      # WordPress version
      if "WordPress version" in line and "identified" in line:
        m = re.search(r"WordPress version\s*([0-9\.]+)", line)
        if m:
          wordpress_version = m.group(1)
        wp_version_desc = line.strip()

      # Theme / template block
      if "WordPress theme in use:" in line:
        # Start of a template/theme section
        try:
          name = line.split(":", 1)[1].strip()
        except Exception:
          name = line.strip()

        block = []
        i += 1
        while i < n and not lines[i].strip().startswith("[+] "):
          block.append(lines[i])
          i += 1
        templates.append(parse_template_block(name, block))
        continue  # already advanced i

      # Generic [+] blocks – can be plugins, findings, etc.
      if line.startswith("[+] "):
        name = line[4:].strip()
        block = []
        j = i + 1
        while j < n and not lines[j].strip().startswith("[+] "):
          block.append(lines[j])
          j += 1

        block_text = "\n".join(block)

        # Heuristic: plugin blocks mention wp-content/plugins
        if "wp-content/plugins" in block_text:
          plugins.append(parse_plugin_block(name, block))
        elif name.lower().startswith("wpscan db api"):
          # Skip plan/usage info; we'll extract summary later if needed
          pass
        else:
          # Treat as generic interesting finding
          combined = name
          if block_text.strip():
            combined = name + "\n" + block_text
          interesting.append(combined)

        i = j
        continue

      # Users: '[i] User(s) Identified:' then '[+] username' lines
      if "User(s) Identified" in line:
        j = i + 1
        while j < n:
          l2 = lines[j].strip()
          if not l2:
            j += 1
            continue
          if l2.startswith("[+] "):
            users.append(l2[4:].strip())
            j += 1
            continue
          if l2.startswith("[+] Finished:") or l2.startswith("[+] URL:"):
            break
          j += 1

      # Meta summary near the end
      if line.startswith("[+] Requests Done:"):
        m = re.search(r"Requests Done:\s*(\d+)", line)
        if m:
          meta["requestsDone"] = int(m.group(1))
      if line.startswith("[+] Cached Requests:"):
        m = re.search(r"Cached Requests:\s*(\d+)", line)
        if m:
          meta["cachedRequests"] = int(m.group(1))
      if line.startswith("[+] Data Sent:"):
        meta["dataSent"] = line.split("Data Sent:", 1)[1].strip()
      if line.startswith("[+] Data Received:"):
        meta["dataReceived"] = line.split("Data Received:", 1)[1].strip()
      if line.startswith("[+] Memory used:"):
        meta["memoryUsed"] = line.split("Memory used:", 1)[1].strip()
      if line.startswith("[+] Elapsed time:"):
        meta["elapsedText"] = line.split("Elapsed time:", 1)[1].strip()

      i += 1

    return {
      "WordpressVersion": {
        "version": wordpress_version,
        "description": wp_version_desc,
      },
      "ScanDate": scan_date,
      "FinishedAt": finished_at,
      "URL": url,
      "Plugins": plugins,
      "Templates": templates,
      "Users": users,
      "InterestingFindings": interesting,
      "Meta": meta,
    }


ENABLE_LISTEN = os.getenv("ENABLE_LISTEN", "false").lower() == "true"
ENABLE_SPEAK = os.getenv("ENABLE_SPEAK", "false").lower() == "true"
ENABLE_VISUAL_PERCEPTION = os.getenv("ENABLE_VISUAL_PERCEPTION", "false").lower() == "true"

if ENABLE_LISTEN or ENABLE_SPEAK:
    from ToolkitVoice import BaseToolkitVoice
else:
    class BaseToolkitVoice(BaseToolkit):
        def __init__(self):
            super().__init__()

if ENABLE_VISUAL_PERCEPTION:
    from ToolkitVisualPerception import BaseToolkitVisualPerception
else:
    class BaseToolkitVisualPerception(BaseToolkit):
        def __init__(self):
            super().__init__()

class FullToolkit(Toolkit, BaseToolkitVoice, BaseToolkitVisualPerception):
  def __init__(self):
    super().__init__()  # MRO will walk: FullToolkit → Toolkit → BaseToolkitVoice → BaseToolkitVisualPerception → BaseToolkit → BaseCoreToolkit

def create_toolkit(mode: str = "system"):
  """
  Small factory to get a toolkit instance:
  - system  -> BaseToolkit (minimal, safe)
  - extra   -> Toolkit (BaseToolkit + security/research tools)
  - hid     -> BaseToolkitVoice (audio I/O)
  - os      -> BaseToolkitVisualPerception (screen/vision/OCR)
  - full    -> FullToolkit (everything)
  """
  m = (mode or "system").lower()
  if m == "system":
    return BaseToolkit()
  if m == "extra":
    return Toolkit()
  if m == "hid":
    return BaseToolkitVoice()
  if m == "os":
    return BaseToolkitVisualPerception()
  if m == "full":
    return FullToolkit()
  # fallback
  return BaseToolkit()